# Standard library imports
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
# NOTE: `h2o` is imported LAZILY, in the one branch that uses it
# (`H2OScoringParameter._score`). It is a heavyweight import and the common scoring path —
# small batches, and every pipeline with no H2O layer at all — never touches it.
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local application/library-specific imports
from typing import List, Dict, Callable, Optional, Union
from mlflow.pyfunc import PythonModel, PythonModelContext

from .binning import apply_thermometer_encoding

__all__ = [
    "BasePricingLayer",
    "ColumnCalculation",
    "CustomLayer",
    "DropColumns",
    "GlmScoring",
    "H2OScoringParameter",
    "OnnxScoring",
    "RatingTablesLayer",
    "RenameColumns",
    "SourceColumn",
    "TableAggregation",
    "TableDrop",
    "TableJoin",
    "TableTransform",
    "DependencyError",
    "LayerCalculationError",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DependencyError(Exception):
    """Raised when layer dependency validation fails."""
    pass


class LayerCalculationError(Exception):
    """Raised when a layer's calculation expression or callable fails."""
    pass


# ---------------------------------------------------------------------------
# Aggregate helper (exposed inside ColumnCalculation eval namespace)
# ---------------------------------------------------------------------------

_ALLOWED_AGG_FUNCS = frozenset(
    {"sum", "mean", "count", "min", "max", "median", "std", "nunique", "first", "last"}
)


def _resolve_structural_join(
    data_structure: List[Dict],
    target_table: Optional[str],
    source: str,
) -> tuple:
    """Look up the direct parent/child join between ``target_table`` and
    ``source`` in the algorithm's frozen data-structure snapshot, and return
    ``(target_key, source_key)`` — the column to read on the target table and
    the column to group on in the source table, respectively.

    Two directions are supported:

    1. ``target_table`` is a child of ``source`` (broadcasting a parent
       attribute down onto child rows): use the target node's
       ``child_key`` on the target side and ``parent_key`` on the source side.
    2. ``source`` is a child of ``target_table`` (aggregating child rows up
       onto the parent): use the source node's ``parent_key`` on the target
       side and ``child_key`` on the source side.

    Only direct parent/child links are resolved — indirect (grandparent /
    grandchild) relationships raise so the caller has to be explicit.
    """
    if not target_table:
        raise LayerCalculationError(
            "aggregate(): cannot auto-detect group_by because the layer has no "
            "target_table bound. Pass group_by= explicitly."
        )
    if not data_structure:
        raise LayerCalculationError(
            f"aggregate(): cannot auto-detect group_by between '{target_table}' "
            f"and '{source}' because no data_structure snapshot is attached to "
            f"this algorithm. Re-create the algorithm from a data structure, or "
            f"pass group_by= explicitly."
        )

    by_name = {node.get("name"): node for node in data_structure}
    target_node = by_name.get(target_table)
    source_node = by_name.get(source)

    if target_node and target_node.get("parent") == source:
        return target_node.get("child_key"), target_node.get("parent_key")
    if source_node and source_node.get("parent") == target_table:
        return source_node.get("parent_key"), source_node.get("child_key")

    raise LayerCalculationError(
        f"aggregate(): no direct parent/child relationship between "
        f"target_table='{target_table}' and source='{source}' in the algorithm's "
        f"data structure. Pass group_by= explicitly, or restructure the data "
        f"structure so the two tables are directly related."
    )


def _broadcast_aggregate(
    df: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    source: str,
    agg_func: str,
    *,
    target_table: Optional[str] = None,
    data_structure: Optional[List[Dict]] = None,
    source_column: Optional[str] = None,
    group_by: Optional[str] = None,
    fill_value: Optional[Union[int, float]] = None,
) -> pd.Series:
    """
    Compute a per-group aggregate on ``data[source]`` and broadcast it onto
    rows of ``df``. Returns a Series of length ``len(df)`` aligned to
    ``df.index`` — drop-in for a ColumnCalculation result.

    The join can be specified two ways:

    - **Auto-detect (default)**: when ``group_by`` is not passed, the helper
      looks up the direct parent/child link between ``target_table`` (the
      layer's table) and ``source`` in ``data_structure`` (the algorithm's
      snapshot of its data structure). Pulls ``parent_key`` and ``child_key``
      from that node — which may legitimately differ (e.g.
      ``parent_key='policy.policy_number'``, ``child_key='risk.policy_number'``).
    - **Explicit override**: pass ``group_by=`` to force a single column name
      that exists on both source and target. Useful for non-structural
      groupings (e.g. group by ``region`` to aggregate something other than
      the structural join key).

    Exposed inside :class:`ColumnCalculation` expression strings as
    ``aggregate(source, agg_func, source_column=None, group_by=None,
    fill_value=None)`` with ``df``, ``data``, ``target_table`` and
    ``data_structure`` pre-bound from the current layer / pipeline.

    Note: this helper is purely a read of ``data[source]``; it does not mutate
    any table in ``data``. It is safe to call from multiple ColumnCalculation
    layers running in parallel within the same wave.

    The pipeline's DAG ordering is still driven by declared layer-name
    dependencies — when the source table is produced or modified by an
    upstream layer, that layer must be listed in ``dependencies=[...]`` on
    the consuming ColumnCalculation.
    """
    if source not in data:
        raise LayerCalculationError(
            f"aggregate(): source table '{source}' not in data. "
            f"Available tables: {sorted(data.keys())}"
        )
    if agg_func not in _ALLOWED_AGG_FUNCS:
        raise LayerCalculationError(
            f"aggregate(): agg_func must be one of {sorted(_ALLOWED_AGG_FUNCS)}, "
            f"got {agg_func!r}"
        )

    if group_by is not None:
        if not isinstance(group_by, str):
            raise LayerCalculationError(
                f"aggregate(): group_by must be a single column name (string); "
                f"multi-column group_by is not yet supported. Got {group_by!r}"
            )
        target_key = source_key = group_by
    else:
        target_key, source_key = _resolve_structural_join(
            data_structure or [], target_table, source,
        )

    if target_key not in df.columns:
        raise LayerCalculationError(
            f"aggregate(): join key {target_key!r} missing on target table "
            f"{target_table!r}"
        )

    source_df = data[source]
    if source_key not in source_df.columns:
        raise LayerCalculationError(
            f"aggregate(): join key {source_key!r} missing on source table '{source}'"
        )

    grouped = source_df.groupby(source_key, sort=False)

    if agg_func == "count":
        if source_column is None:
            agg_series = grouped.size()
        else:
            if source_column not in source_df.columns:
                raise LayerCalculationError(
                    f"aggregate(): source_column '{source_column}' missing on source table '{source}'"
                )
            agg_series = grouped[source_column].count()
        default_fill: Union[int, float] = 0
    else:
        if source_column is None:
            raise LayerCalculationError(
                f"aggregate(): source_column is required for agg_func={agg_func!r}"
            )
        if source_column not in source_df.columns:
            raise LayerCalculationError(
                f"aggregate(): source_column '{source_column}' missing on source table '{source}'"
            )
        agg_series = getattr(grouped[source_column], agg_func)()
        default_fill = np.nan

    fill = default_fill if fill_value is None else fill_value
    mapped = df[target_key].map(agg_series)
    return pd.Series(mapped.values, index=df.index).fillna(fill)


# ---------------------------------------------------------------------------
# BasePricingLayer
# ---------------------------------------------------------------------------

class BasePricingLayer(ABC):
    """Abstract base class for all pricing pipeline layers."""

    def __init__(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.name = name
        self.dependencies: List[str] = dependencies or []
        self.description = description
        self.parameters: Dict = {}

    @abstractmethod
    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        """Execute this layer and return the (mutated) data dict."""
        ...

    def _layer_params(self) -> dict:
        """Return subclass-specific JSON-serializable fields for to_dict()."""
        return {}

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of this layer."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "dependencies": self.dependencies,
            "description": self.description,
            **self._layer_params(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BasePricingLayer":
        """Reconstruct a layer from a dict produced by to_dict()."""
        from .layer_registry import LAYER_REGISTRY
        layer_type = d.get("type")
        layer_cls = LAYER_REGISTRY.get(layer_type)
        if layer_cls is None:
            raise ValueError(f"Unknown layer type: {layer_type!r}")
        if not hasattr(layer_cls, "_from_dict"):
            raise ValueError(
                f"Layer type {layer_type!r} requires a callable and cannot be "
                "reconstructed from a dict. Create it programmatically instead."
            )
        return layer_cls._from_dict(d)


# ---------------------------------------------------------------------------
# ColumnCalculation
# ---------------------------------------------------------------------------

class ColumnCalculation(BasePricingLayer):
    """
    A single pricing layer that computes **one new column** and writes it
    to a specified table within the dataset.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key of the table the computed column is written to. Also the table
        that ``df`` refers to inside calculations.
    output_column : str
        Name of the column to create (or overwrite) on the target table.
    calculation : str | Callable
        Either a Python expression string (where ``df`` is ``data[target_table]``
        and ``data``, ``catalog``, ``options`` are also in scope), or a callable
        with signature
        ``(df: pd.DataFrame, data: dict, catalog: dict, options: dict) -> pd.Series``.

        Expression strings additionally have access to an
        ``aggregate(source, agg_func, source_column=None, group_by=None, fill_value=None)``
        helper for per-group aggregation broadcast onto the target table — a
        single-layer replacement for the ``TableAggregation`` + ``TableJoin``
        pattern. Supported ``agg_func`` values: ``"sum"``, ``"mean"``,
        ``"count"``, ``"min"``, ``"max"``, ``"median"``, ``"std"``,
        ``"nunique"``, ``"first"``, ``"last"``. ``source_column`` may be
        omitted only when ``agg_func="count"`` (defaults to a row count).
        Default ``fill_value`` is ``0`` for ``"count"`` and ``NaN`` otherwise.

        When ``group_by`` is omitted, the join keys are auto-derived from the
        algorithm's data-structure snapshot using the direct parent/child link
        between ``target_table`` and ``source`` — supports keys that differ
        per side (e.g. ``parent_key='policy.policy_number'``,
        ``child_key='risk.policy_number'``). Pass ``group_by=`` explicitly
        only when grouping by a non-structural column. Example::

            ColumnCalculation(
                name="policy_vehicle_count",
                target_table="policy",
                output_column="policy_vehicle_count",
                calculation="aggregate(source='risk', agg_func='count')",
            )

        When the source table is produced or modified by an upstream layer,
        that layer must be listed in ``dependencies=[...]`` — the DAG resolves
        ordering from declared layer names, not from columns referenced
        inside the expression.
    dependencies : list[str], optional
        Names of other layers that must run before this one.
    description : str, optional
        Human-readable description of what this layer computes.
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        output_column: str,
        calculation: Union[str, Callable],
        dependencies: Optional[List[str]] = None,
        description: str = "",
        override_source: bool = False,
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)

        if not isinstance(target_table, str) or not target_table.strip():
            raise ValueError("target_table must be a non-empty string")
        if not isinstance(output_column, str) or not output_column.strip():
            raise ValueError("output_column must be a non-empty string")
        if not callable(calculation) and not isinstance(calculation, str):
            raise TypeError(
                "calculation must be a callable or a Python expression string"
            )

        self.target_table = target_table
        self.output_column = output_column
        self.calculation = calculation
        # When True this layer overrides a source-data column of the same name.
        # The pipeline snapshots the raw fed-in value as ``<output_column>{}``
        # before any wave runs (see PyfuncPricingPipeline.predict), so both the
        # raw and the overridden value stay addressable.
        self.override_source = bool(override_source)
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "output_column": output_column,
            "calculation": calculation,
            "dependencies": dependencies,
            "description": description,
            "override_source": self.override_source,
        }

    @staticmethod
    def _safe_builtins() -> dict:
        return {
            "int": int, "float": float, "str": str,
            "bool": bool, "len": len, "abs": abs,
            "round": round, "min": min, "max": max,
            "sum": sum, "zip": zip, "map": map,
            "enumerate": enumerate, "range": range,
            "list": list, "dict": dict, "tuple": tuple,
            "set": set, "isinstance": isinstance,
            "True": True, "False": False, "None": None,
        }

    def _compute_result(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> pd.Series:
        """Compute the output Series without writing to ``data``."""
        target_df = data[self.target_table]
        df = target_df
        target_table = self.target_table
        data_structure = options.get("data_structure", []) if options else []

        try:
            if callable(self.calculation):
                result = self.calculation(df, data, catalog, options)
            else:
                result = eval(  # noqa: S307
                    self.calculation,
                    {
                        "__builtins__": self._safe_builtins(),
                        "np": np,
                        "pd": pd,
                        "df": df,
                        "data": data,
                        "catalog": catalog,
                        "options": options,
                        "aggregate": (
                            lambda source, agg_func, source_column=None, group_by=None, fill_value=None:
                            _broadcast_aggregate(
                                df, data, source, agg_func,
                                target_table=target_table,
                                data_structure=data_structure,
                                source_column=source_column,
                                group_by=group_by,
                                fill_value=fill_value,
                            )
                        ),
                    },
                )
        except LayerCalculationError:
            raise
        except Exception as exc:
            raise LayerCalculationError(
                f"Layer '{self.name}': error computing '{self.output_column}' "
                f"on table '{self.target_table}': {exc}"
            ) from exc

        if not isinstance(result, pd.Series):
            try:
                result = pd.Series(result, index=target_df.index)
            except Exception as exc:
                raise LayerCalculationError(
                    f"Layer '{self.name}': calculation must return a pd.Series or a "
                    f"scalar broadcastable to one. Got {type(result)}: {exc}"
                ) from exc

        if len(result) != len(target_df):
            raise LayerCalculationError(
                f"Layer '{self.name}': returned Series length {len(result)} does not "
                f"match table '{self.target_table}' length {len(target_df)}"
            )

        return result

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute ``output_column`` and write it to ``data[target_table]``."""
        result = self._compute_result(data, catalog, options, context)
        data[self.target_table] = data[self.target_table].copy()
        data[self.target_table][self.output_column] = result.values
        return data

    def _layer_params(self) -> dict:
        is_string = isinstance(self.calculation, str)
        return {
            "target_table": self.target_table,
            "output_column": self.output_column,
            "calculation": self.calculation if is_string else None,
            "calculation_type": "string" if is_string else "callable",
            "override_source": self.override_source,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "ColumnCalculation":
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            output_column=d["output_column"],
            calculation=d.get("calculation") or "",
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
            override_source=d.get("override_source", False),
        )

    def __repr__(self) -> str:  # pragma: no cover
        calc_repr = (
            f"'{self.calculation}'"
            if isinstance(self.calculation, str)
            else getattr(self.calculation, "__name__", repr(self.calculation))
        )
        return (
            f"ColumnCalculation(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"output={self.output_column!r}, "
            f"calc={calc_repr})"
        )


# ---------------------------------------------------------------------------
# TableTransform
# ---------------------------------------------------------------------------

class TableTransform(BasePricingLayer):
    """
    Creates any new table in the data dict by applying a callable to a source table.

    The callable receives the source DataFrame plus the full ``data`` dict, so it
    can filter, reshape, or combine tables in any way and return a new DataFrame.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    source_table : str
        Key of the table passed as ``df`` into ``calculation``.
    output_table : str
        Key under which the result is stored in ``data``.
    calculation : Callable
        ``(df: pd.DataFrame, data: dict, catalog: dict, options: dict) -> pd.DataFrame``
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        TableTransform(
            name="high_value_risks",
            source_table="risk",
            output_table="high_value_risks",
            calculation=lambda df, data, catalog, options: df[df['vehicle_market_value'] > 50_000],
        )
    """

    def __init__(
        self,
        name: str,
        source_table: str,
        output_table: str,
        calculation: Callable,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)

        if not callable(calculation):
            raise TypeError(f"Layer '{name}': calculation must be a callable")

        self.source_table = source_table
        self.output_table = output_table
        self.calculation = calculation
        self.parameters = {
            "name": name,
            "source_table": source_table,
            "output_table": output_table,
            "calculation": calculation,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        result = self.calculation(data[self.source_table], data, catalog, options)
        data[self.output_table] = result.reset_index(drop=True)
        return data

    def _layer_params(self) -> dict:
        return {
            "source_table": self.source_table,
            "output_table": self.output_table,
            "calculation_type": "callable",
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TableTransform(name={self.name!r}, "
            f"source={self.source_table!r}, "
            f"output={self.output_table!r})"
        )


# ---------------------------------------------------------------------------
# TableAggregation
# ---------------------------------------------------------------------------

class TableAggregation(TableTransform):
    """
    Shorthand for groupby aggregations. Subclasses :class:`TableTransform`.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    source_table : str
        Key of the table to aggregate.
    output_table : str
        Key under which the aggregated DataFrame is stored in ``data``.
    groupby_key : str | list[str]
        Column(s) to group by.
    aggregations : dict
        Pandas named-aggregation format:
        ``{output_col: (source_col, agg_func)}`` or ``{col: agg_func}``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        TableAggregation(
            name="risk_agg",
            source_table="risk",
            output_table="risk_agg",
            groupby_key="policy_number",
            aggregations={"total_pred_claim_cost": ("pred_claim_cost", "sum")},
            dependencies=["pred_claim_cost"],
        )
    """

    def __init__(
        self,
        name: str,
        source_table: str,
        output_table: str,
        groupby_key: Union[str, List[str]],
        aggregations: Dict,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        def _agg(df, data, catalog, options):
            return df.groupby(groupby_key, as_index=False).agg(
                **{k: v for k, v in aggregations.items()}
            )

        super().__init__(
            name=name,
            source_table=source_table,
            output_table=output_table,
            calculation=_agg,
            dependencies=dependencies,
            description=description,
        )
        self.groupby_key = groupby_key
        self.aggregations = aggregations
        self.parameters = {
            "name": name,
            "source_table": source_table,
            "output_table": output_table,
            "groupby_key": groupby_key,
            "aggregations": aggregations,
            "dependencies": dependencies,
            "description": description,
        }

    def _layer_params(self) -> dict:
        return {
            "source_table": self.source_table,
            "output_table": self.output_table,
            "groupby_key": self.groupby_key,
            "aggregations": self.aggregations,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "TableAggregation":
        agg = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d["aggregations"].items()
        }
        return cls(
            name=d["name"],
            source_table=d["source_table"],
            output_table=d["output_table"],
            groupby_key=d["groupby_key"],
            aggregations=agg,
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TableAggregation(name={self.name!r}, "
            f"source={self.source_table!r}, "
            f"output={self.output_table!r}, "
            f"groupby={self.groupby_key!r})"
        )


# ---------------------------------------------------------------------------
# TableJoin
# ---------------------------------------------------------------------------

class TableJoin(BasePricingLayer):
    """
    Merges ``right_table`` onto ``left_table`` and writes the result back to
    ``output_table`` (defaults to ``left_table``).

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    left_table : str
        Key of the base table (e.g. ``"policy"``).
    right_table : str
        Key of the table to merge in.
    right_source : str
        Where to look up ``right_table`` — ``"data"`` (default) or ``"catalog"``.
    left_on : str | list[str]
        Column(s) in ``left_table`` to join on.
    right_on : str | list[str]
        Column(s) in ``right_table`` to join on. Defaults to ``left_on``
        when both sides share the same key name.
    output_table : str, optional
        Where to store the result. Defaults to ``left_table`` (in-place enrichment).
    columns : list[str], optional
        Columns from ``right_table`` to bring across. ``None`` brings all columns.
    how : str
        Merge strategy — ``"left"``, ``"inner"``, ``"outer"``, ``"right"``.
        Defaults to ``"left"``.
    suffixes : tuple
        Suffixes applied to overlapping column names. Defaults to ``("", "_right")``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        # Same key name on both sides
        TableJoin(
            name="join_risk_agg",
            left_table="policy",
            right_table="risk_agg",
            left_on="policy_number",
            columns=["total_pred_claim_cost"],
            dependencies=["risk_agg"],
        )

        # Different key names
        TableJoin(
            name="join_risk_agg",
            left_table="policy",
            right_table="risk_agg",
            left_on="policy_number",
            right_on="unique_id",
            columns=["total_pred_claim_cost"],
            dependencies=["risk_agg"],
        )
    """

    def __init__(
        self,
        name: str,
        left_table: str,
        right_table: str,
        left_on: Union[str, List[str]],
        right_on: Optional[Union[str, List[str]]] = None,
        output_table: Optional[str] = None,
        columns: Optional[List[str]] = None,
        how: str = "left",
        suffixes: tuple = ("", "_right"),
        right_source: str = "data",
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        if right_source not in ("data", "catalog"):
            raise ValueError(f"right_source must be 'data' or 'catalog', got {right_source!r}")
        super().__init__(name=name, dependencies=dependencies, description=description)
        self.left_table = left_table
        self.right_table = right_table
        self.right_source = right_source
        self.left_on = left_on
        self.right_on = right_on if right_on is not None else left_on
        self.output_table = output_table or left_table
        self.columns = columns
        self.how = how
        self.suffixes = suffixes
        self.parameters = {
            "name": name,
            "left_table": left_table,
            "right_table": right_table,
            "left_on": left_on,
            "right_on": right_on,
            "output_table": output_table,
            "columns": columns,
            "how": how,
            "suffixes": suffixes,
            "right_source": right_source,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        left = data[self.left_table]
        source = data if self.right_source == "data" else catalog
        right = source[self.right_table]

        if self.columns is not None:
            right_key_cols = [self.right_on] if isinstance(self.right_on, str) else self.right_on
            right = right[right_key_cols + self.columns]

        merged = left.merge(
            right,
            left_on=self.left_on,
            right_on=self.right_on,
            how=self.how,
            suffixes=self.suffixes,
        )
        data[self.output_table] = merged.reset_index(drop=True)
        return data

    def _layer_params(self) -> dict:
        return {
            "left_table": self.left_table,
            "right_table": self.right_table,
            "right_source": self.right_source,
            "left_on": self.left_on,
            "right_on": self.right_on,
            "output_table": self.output_table,
            "columns": self.columns,
            "how": self.how,
            "suffixes": list(self.suffixes),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "TableJoin":
        return cls(
            name=d["name"],
            left_table=d["left_table"],
            right_table=d["right_table"],
            left_on=d["left_on"],
            right_on=d.get("right_on"),
            output_table=d.get("output_table"),
            columns=d.get("columns"),
            how=d.get("how", "left"),
            suffixes=tuple(d.get("suffixes", ["", "_right"])),
            right_source=d.get("right_source", "data"),
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TableJoin(name={self.name!r}, "
            f"left={self.left_table!r}, "
            f"right={self.right_table!r}, "
            f"left_on={self.left_on!r}, "
            f"right_on={self.right_on!r}, "
            f"output={self.output_table!r})"
        )


# ---------------------------------------------------------------------------
# RenameColumns
# ---------------------------------------------------------------------------

class RenameColumns(BasePricingLayer):
    """
    Renames columns on a table in the data dict.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key of the table whose columns will be renamed.
    columns : dict
        Mapping of ``{old_name: new_name}``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        RenameColumns(
            name="rename_agg_cols",
            target_table="risk_agg",
            columns={"pred_claim_cost": "total_pred_claim_cost"},
            dependencies=["risk_agg"],
        )
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        columns: Dict[str, str],
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)
        if not columns:
            raise ValueError(f"Layer '{name}': columns mapping must not be empty")
        self.target_table = target_table
        self.columns = columns
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "columns": columns,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        data[self.target_table] = data[self.target_table].rename(columns=self.columns)
        return data

    def _layer_params(self) -> dict:
        return {
            "target_table": self.target_table,
            "columns": self.columns,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "RenameColumns":
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            columns=d["columns"],
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RenameColumns(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"columns={self.columns!r})"
        )


# ---------------------------------------------------------------------------
# DropColumns
# ---------------------------------------------------------------------------

class DropColumns(BasePricingLayer):
    """
    Drops columns from a table in the data dict.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key of the table whose columns will be dropped.
    columns : list[str]
        Column names to drop.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        DropColumns(
            name="drop_temp_cols",
            target_table="risk_agg",
            columns=["tmp_flag", "tmp_id"],
        )
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        columns: List[str],
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)
        if not columns:
            raise ValueError(f"Layer '{name}': columns list must not be empty")
        self.target_table = target_table
        self.columns = columns
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "columns": columns,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        data[self.target_table] = data[self.target_table].drop(columns=self.columns)
        return data

    def _layer_params(self) -> dict:
        return {
            "target_table": self.target_table,
            "columns": self.columns,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "DropColumns":
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            columns=d["columns"],
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DropColumns(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"columns={self.columns!r})"
        )


# ---------------------------------------------------------------------------
# TableDrop
# ---------------------------------------------------------------------------

class TableDrop(BasePricingLayer):
    """
    Removes one or more intermediate tables from the data dict.

    Useful for cleaning up temporary aggregation tables after they have been
    joined onto their target table.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    tables : str | list[str]
        Table key(s) to remove from ``data``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        TableDrop(
            name="drop_risk_agg",
            tables="risk_agg",
            dependencies=["join_risk_agg"],
        )
    """

    def __init__(
        self,
        name: str,
        tables: Union[str, List[str]],
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)
        self.tables = [tables] if isinstance(tables, str) else tables
        self.parameters = {
            "name": name,
            "tables": tables,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        for table in self.tables:
            if table not in data:
                raise KeyError(
                    f"Layer '{self.name}': table '{table}' not found in data."
                )
            del data[table]
        return data

    def _layer_params(self) -> dict:
        return {"tables": self.tables}

    @classmethod
    def _from_dict(cls, d: dict) -> "TableDrop":
        return cls(
            name=d["name"],
            tables=d["tables"],
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"TableDrop(name={self.name!r}, tables={self.tables!r})"


# ---------------------------------------------------------------------------
# SourceColumn
# ---------------------------------------------------------------------------

class SourceColumn(BasePricingLayer):
    """
    Declares a single source-data column that enters the pipeline from the
    score-time input payload.

    This layer is an **identity / no-op** at execution time — the column
    already exists in ``data[source_table]``. It exists so the dependency
    graph can attribute ``df['<source_column>']`` references back to a real,
    registered layer (rather than treating source columns as ambient and
    invisible to dependency detection).

    Parameters
    ----------
    name : str
        Unique identifier for this layer (conventionally ``"<table>.<column>"``).
    source_table : str
        Key of the table this column lives on in the score-time payload.
    source_column : str
        Name of the source column.
    dtype : str | None, optional
        Classification of the variable — one of
        ``"numeric"``, ``"categorical"``, ``"date"``, ``"boolean"``,
        ``"text"``, or ``None`` (unspecified). Descriptive metadata; will
        inform future aging transforms (date vs numeric aging).
    transformation : str | None, optional
        RESERVED FOR FUTURE USE (multi-year policy aging). When set, a Python
        expression (``df``/``data``/``np``/``pd`` in scope) recomputing the
        column in place. Currently serialized but **not applied** — ``apply``
        is always identity. See ``open_rating_engine.data_processing.data_aging``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        SourceColumn(
            name="risk.driver_youngest_age",
            source_table="risk",
            source_column="driver_youngest_age",
            dtype="numeric",
        )
    """

    ALLOWED_DTYPES = ("numeric", "categorical", "date", "boolean", "text")

    def __init__(
        self,
        name: str,
        source_table: str,
        source_column: str,
        dtype: Optional[str] = None,
        transformation: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)
        if not isinstance(source_table, str) or not source_table.strip():
            raise ValueError(f"Layer '{name}': source_table must be a non-empty string")
        if not isinstance(source_column, str) or not source_column.strip():
            raise ValueError(f"Layer '{name}': source_column must be a non-empty string")
        if dtype is not None and dtype not in self.ALLOWED_DTYPES:
            raise ValueError(
                f"Layer '{name}': dtype must be None or one of "
                f"{self.ALLOWED_DTYPES!r}, got {dtype!r}"
            )
        if transformation is not None and not isinstance(transformation, str):
            raise ValueError(
                f"Layer '{name}': transformation must be a string when supplied"
            )
        self.source_table = source_table
        self.source_column = source_column
        self.dtype = dtype
        self.transformation = transformation  # reserved; not applied yet
        self.parameters = {
            "name": name,
            "source_table": source_table,
            "source_column": source_column,
            "dtype": dtype,
            "transformation": transformation,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        # Identity no-op: the column is supplied in the score-time payload.
        # FUTURE: if self.transformation, recompute
        # data[self.source_table][self.source_column] here (data aging).
        return data

    def _layer_params(self) -> dict:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "dtype": self.dtype,
            "transformation": self.transformation,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "SourceColumn":
        return cls(
            name=d["name"],
            source_table=d["source_table"],
            source_column=d["source_column"],
            dtype=d.get("dtype"),
            transformation=d.get("transformation"),
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SourceColumn(name={self.name!r}, "
            f"source={self.source_table!r}.{self.source_column!r}, "
            f"dtype={self.dtype!r})"
        )


# ---------------------------------------------------------------------------
# RatingTablesLayer
# ---------------------------------------------------------------------------

class RatingTablesLayer(ColumnCalculation):
    """
    Applies a set of rating tables to produce a combined output column.

    Each rating table is a DataFrame with one or more key columns (whose names
    must match columns in the target table) and a numeric ``value`` column.
    For each row, the target table is left-joined to every rating table on the
    matching key columns. By default the resulting ``value`` columns are
    multiplied together; alternatively, a ``formula`` may be supplied to
    combine them via any Python expression (e.g. mixing multiplicative and
    additive factors).

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key of the table the output column is written to.
    output_column : str
        Name of the new column holding the combined result.
    rating_tables : dict[str, pd.DataFrame]
        ``{table_name: DataFrame}``.  Every DataFrame must contain a ``value``
        column plus at least one key column whose name exists in
        ``data[target_table]``.
    formula : str, optional
        Python expression combining the rating tables. Each rating table is
        in scope under its own name as a one-column DataFrame with column
        ``"value"``, so refer to its looked-up Series via ``tbl['value']``.
        If omitted, all tables' ``value`` columns are multiplied together.
    default_values : dict[str, float], optional
        Per-table fill value used when a row's key is absent from a rating
        table. Use ``1`` for multiplicative identity (the default for any
        table not listed) and ``0`` for additive identity. Keys must match
        entries in ``rating_tables``.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        RatingTablesLayer(
            name="demand_multiplier",
            target_table="risk",
            output_column="demand_multiplier",
            rating_tables={
                "risk_age_band_mult": pd.DataFrame({
                    "risk_age_band": ["15-24", "25-34", "35+"],
                    "value":         [1.2,     1.0,     1.0],
                }),
                "customer_tenure_band_mult": pd.DataFrame({
                    "customer_tenure_band": ["0-5", "6-10", "10+"],
                    "value":                [1.0,   1.0,    1.0],
                }),
                "customer_tenure_band_add": pd.DataFrame({
                    "customer_tenure_band": ["0-5", "6-10", "10+"],
                    "value":                [0.0,   0.1,    0.0],
                }),
            },
            formula=(
                "risk_age_band_mult['value'] * customer_tenure_band_mult['value']"
                " + customer_tenure_band_add['value']"
            ),
            default_values={
                "risk_age_band_mult": 1,
                "customer_tenure_band_mult": 1,
                "customer_tenure_band_add": 0,
            },
        )
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        output_column: str,
        rating_tables: Dict[str, pd.DataFrame],
        formula: Optional[str] = None,
        default_values: Optional[Dict[str, float]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(
            name=name,
            target_table=target_table,
            output_column=output_column,
            calculation=lambda *_: None,  # never called; _compute_result is overridden
            dependencies=dependencies,
            description=description,
        )
        if not isinstance(rating_tables, dict) or not rating_tables:
            raise ValueError(f"Layer '{name}': rating_tables must be a non-empty dict")
        for tbl_name, tbl_df in rating_tables.items():
            if not isinstance(tbl_df, pd.DataFrame):
                raise ValueError(
                    f"Layer '{name}': rating_tables['{tbl_name}'] must be a pd.DataFrame"
                )
            if "value" not in tbl_df.columns:
                raise ValueError(
                    f"Layer '{name}': rating_tables['{tbl_name}'] must contain a 'value' column"
                )
            # Check that value column is numeric
            if not pd.api.types.is_numeric_dtype(tbl_df["value"]):
                raise ValueError(
                    f"Layer '{name}': rating_tables['{tbl_name}']['value'] must be numeric"
                )
            key_cols = [c for c in tbl_df.columns if c != "value"]
            if not key_cols:
                raise ValueError(
                    f"Layer '{name}': rating_tables['{tbl_name}'] must have at least one key column besides 'value'"
                )
            # Check that key columns are categorical
            for key_col in key_cols:
                if not (isinstance(tbl_df[key_col].dtype, pd.CategoricalDtype) or tbl_df[key_col].dtype == object):
                    raise ValueError(
                        f"Layer '{name}': rating_tables['{tbl_name}']['{key_col}'] must be categorical"
                    )
            # Check for no duplicate values across key_cols
            if tbl_df.duplicated(subset=key_cols).any():
                raise ValueError(
                    f"Layer '{name}': rating_tables['{tbl_name}'] has duplicate values across key columns {key_cols}"
                )
        self.rating_tables = rating_tables

        # --- formula validation
        if formula is not None:
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError(
                    f"Layer '{name}': formula must be a non-empty string when provided"
                )
            reserved = {"np", "pd"}
            for tbl_name in rating_tables:
                if not tbl_name.isidentifier() or tbl_name in reserved:
                    raise ValueError(
                        f"Layer '{name}': rating table name '{tbl_name}' is not a valid "
                        f"Python identifier or shadows a reserved name (np, pd)"
                    )
            try:
                self._formula_code = compile(formula, "<formula>", "eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"Layer '{name}': formula has invalid Python syntax: {exc}"
                ) from exc
        else:
            self._formula_code = None
        self.formula = formula

        # --- default_values validation + normalisation
        if default_values is not None:
            if not isinstance(default_values, dict):
                raise ValueError(
                    f"Layer '{name}': default_values must be a dict if provided"
                )
            unknown = set(default_values) - set(rating_tables)
            if unknown:
                raise ValueError(
                    f"Layer '{name}': default_values contains unknown rating tables: "
                    f"{sorted(unknown)}"
                )
            for k, v in default_values.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(
                        f"Layer '{name}': default_values['{k}'] must be numeric"
                    )
        normalised_defaults = {
            tbl_name: float(default_values[tbl_name]) if default_values and tbl_name in default_values else 1.0
            for tbl_name in rating_tables
        }
        self.default_values = normalised_defaults

        self.parameters = {
            "name": name,
            "target_table": target_table,
            "output_column": output_column,
            "rating_tables": rating_tables,
            "formula": formula,
            "default_values": normalised_defaults,
            "dependencies": dependencies,
            "description": description,
        }

    def _compute_result(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> pd.Series:
        df = data[self.target_table]
        lookup_frames: Dict[str, pd.DataFrame] = {}
        try:
            for tbl_name, tbl_df in self.rating_tables.items():
                key_cols = [c for c in tbl_df.columns if c != "value"]
                missing = [c for c in key_cols if c not in df.columns]
                if missing:
                    raise LayerCalculationError(
                        f"Layer '{self.name}': columns {missing} from rating table '{tbl_name}' "
                        f"are missing in '{self.target_table}'"
                    )
                merged = df[key_cols].merge(
                    tbl_df[key_cols + ["value"]],
                    on=key_cols,
                    how="left",
                )
                value_series = pd.Series(
                    merged["value"].values, index=df.index
                ).fillna(self.default_values[tbl_name])
                lookup_frames[tbl_name] = pd.DataFrame({"value": value_series})

            if self._formula_code is not None:
                eval_globals = {
                    "__builtins__": ColumnCalculation._safe_builtins(),
                    "np": np,
                    "pd": pd,
                }
                combined = eval(self._formula_code, eval_globals, dict(lookup_frames))  # noqa: S307
                if not isinstance(combined, pd.Series):
                    combined = pd.Series(combined, index=df.index)
            else:
                combined = pd.concat(
                    [frame["value"] for frame in lookup_frames.values()], axis=1
                ).prod(axis=1)
        except LayerCalculationError:
            raise
        except Exception as exc:
            raise LayerCalculationError(
                f"Layer '{self.name}': failed computing rating factor result: {exc}"
            ) from exc

        if len(combined) != len(df):
            raise LayerCalculationError(
                f"Layer '{self.name}': formula produced length {len(combined)}, "
                f"expected {len(df)}"
            )
        return combined

    def _layer_params(self) -> dict:
        return {
            "target_table": self.target_table,
            "output_column": self.output_column,
            "rating_tables": {
                k: df.to_dict(orient="records")
                for k, df in self.rating_tables.items()
            },
            "formula": self.formula,
            "default_values": self.default_values,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "RatingTablesLayer":
        rating_tables = {
            k: pd.DataFrame(records)
            for k, records in d["rating_tables"].items()
        }
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            output_column=d["output_column"],
            rating_tables=rating_tables,
            formula=d.get("formula"),
            default_values=d.get("default_values"),
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        formula_repr = f", formula={self.formula!r}" if self.formula else ""
        return (
            f"RatingTablesLayer(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"output={self.output_column!r}, "
            f"factors={list(self.rating_tables.keys())!r}{formula_repr})"
        )


# ---------------------------------------------------------------------------
# CustomLayer
# ---------------------------------------------------------------------------

class CustomLayer(BasePricingLayer):
    """
    A free-form layer that delegates entirely to a user-supplied callable.

    The callable receives the full ``data`` dict plus ``catalog``, ``options``,
    and an optional ``context``, and must return the (mutated or replaced)
    ``data`` dict.  Unlike :class:`TableTransform` or :class:`ColumnCalculation`,
    no table or column is prescribed — the function can do anything.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    func : Callable
        ``(data: dict, catalog: dict, options: dict, context) -> dict``
        The returned dict becomes the new ``data`` for downstream layers.
    dependencies : list[str], optional
    description : str, optional

    Example
    -------
    ::

        def add_combined_premium(data, catalog, options, context=None):
            data["policy"]["combined_premium"] = (
                data["policy"]["base_premium"] * data["policy"]["demand_multiplier"]
            )
            return data

        CustomLayer(
            name="combined_premium",
            func=add_combined_premium,
            dependencies=["demand_multiplier"],
        )
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        super().__init__(name=name, dependencies=dependencies, description=description)
        if not callable(func):
            raise TypeError(f"Layer '{name}': func must be a callable")
        self.func = func
        self.parameters = {
            "name": name,
            "func": func,
            "dependencies": dependencies,
            "description": description,
        }

    def apply(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> Dict[str, pd.DataFrame]:
        try:
            result = self.func(data, catalog, options, context)
        except Exception as exc:
            raise LayerCalculationError(
                f"Layer '{self.name}': custom function raised an error: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise LayerCalculationError(
                f"Layer '{self.name}': func must return a dict, got {type(result)}"
            )
        return result

    def _layer_params(self) -> dict:
        return {
            "func_name": getattr(self.func, "__name__", repr(self.func)),
            "calculation_type": "callable",
        }

    def __repr__(self) -> str:  # pragma: no cover
        func_name = getattr(self.func, "__name__", repr(self.func))
        return f"CustomLayer(name={self.name!r}, func={func_name!r})"


# ---------------------------------------------------------------------------
# Score models
# ---------------------------------------------------------------------------
def _deepcopy_sharing(obj, memo, shared_attr: str):
    """Deep-copy ``obj`` but keep ``shared_attr`` (a loaded model) by reference.

    ``__getstate__`` drops the model so the layer pickles, and ``deepcopy`` would
    otherwise go through it too, making every copy reload the model from disk. The
    optimisers deep-copy the pipeline once per grid point / objective evaluation, so
    they share the read-only loaded model instead.
    """
    clone = obj.__class__.__new__(obj.__class__)
    memo[id(obj)] = clone
    for key, value in obj.__dict__.items():
        setattr(clone, key, value if key == shared_attr else deepcopy(value, memo))
    return clone


def _row_to_h2o_payload(row: pd.Series) -> dict[str, str]:
    """Stringify a row for H2O RowData, omitting missing values.

    EasyPredictModelWrapper interprets an absent column as NA. Passing an explicit
    Java null instead triggers a NullPointerException deep in genmodel, so NaN/None
    values must be dropped, not forwarded.
    """
    return {col: str(val) for col, val in row.items() if not pd.isna(val)}


def _canonical_level_label(value):
    """Canonical string label for a categorical level, mirroring
    the fit pipeline's level stringification so score-time ``group_levels`` labels
    match the labels the MOJO was trained on (e.g. a count column loaded as float64
    must read ``"2"`` not ``"2.0"``). Returns ``None`` for missing."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    return str(value)


class H2OScoringParameter(ColumnCalculation):
    """
    A pipeline layer that scores rows in a dataset table using an H2O MOJO model
    and writes the predictions as a new column.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key in the ``data`` dict whose rows will be scored.
    output_column : str
        Name of the column written with the model's predictions.
    column_map : dict[str, str]
        Mapping ``{required_model_column: source_column_in_target_table}``.
        Source columns are selected from ``data[target_table]`` and renamed to
        the required MOJO names before scoring. The keys are the canonical
        required-column list for the MOJO.
    mojo_artifact_key : str
        Key in ``context.artifacts`` pointing to the exported MOJO zip file.
    genmodel_artifact_key : str
        Key in ``context.artifacts`` pointing to the H2O ``genmodel.jar``.
    model_type : str
        One of ``"regression"``, ``"binomial"``, or ``"multinomial"``.
    n_threads : int
        Number of threads used for parallel row scoring.
    dependencies : list[str], optional
    description : str, optional
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        output_column: str,
        column_map: Dict[str, str],
        mojo_artifact_key: str,
        genmodel_artifact_key: str,
        model_type: str = "binomial",
        n_threads: int = 4,
        derived: Optional[List[Dict]] = None,
        preprocessing: Optional[List[Dict]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        if not column_map or not isinstance(column_map, dict):
            raise ValueError("column_map must be a non-empty {required: source} dict")

        super().__init__(
            name=name,
            target_table=target_table,
            output_column=output_column,
            calculation=lambda _df, _data, _catalog, _options: None,  # never called
            dependencies=dependencies,
            description=description,
        )

        self.model_type = model_type
        self.n_threads = n_threads
        self.mojo_artifact_key = mojo_artifact_key
        self.genmodel_artifact_key = genmodel_artifact_key
        self.column_map = column_map
        # ``derived`` + ``preprocessing`` are the modelling-time recipe baked into
        # registry-imported layers; when both are empty the layer falls back to
        # the legacy contract (column_map keyed by MOJO required columns, no
        # in-layer transforms).
        self.derived = list(derived or [])
        self.preprocessing = list(preprocessing or [])
        self.wrapper = None
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "output_column": output_column,
            "mojo_artifact_key": mojo_artifact_key,
            "genmodel_artifact_key": genmodel_artifact_key,
            "model_type": model_type,
            "n_threads": n_threads,
            "column_map": column_map,
            "derived": self.derived,
            "preprocessing": self.preprocessing,
            "dependencies": dependencies,
            "description": description,
        }

    @property
    def model_columns(self) -> List[str]:
        return list(self.column_map.keys())

    @staticmethod
    def _load_model(mojo_zip_path: str, genmodel_jar_path: str):
        import jpype
        import jpype.imports
        import sys

        # Imports must happen eagerly here to avoid JPype lazy-loading issues in multi-threaded scoring.
        from hex.genmodel.easy import EasyPredictModelWrapper
        from hex.genmodel import MojoModel
        return EasyPredictModelWrapper(MojoModel.load(mojo_zip_path))

    def init_from_context(self, context: PythonModelContext) -> None:
        self.wrapper = self._load_model(context.artifacts[self.mojo_artifact_key], context.artifacts[self.genmodel_artifact_key])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["wrapper"] = None
        return state

    def __deepcopy__(self, memo):
        return _deepcopy_sharing(self, memo, "wrapper")

    @staticmethod
    def _mojo_predict_pandas(
        dataframe: pd.DataFrame,
        wrapper,
        model_type: str = "regression",
        n_threads: int = 4,
    ) -> pd.DataFrame:
        from hex.genmodel.easy import RowData

        def score_row(row):
            r = RowData()
            for col, val in _row_to_h2o_payload(row).items():
                r.put(col, val)
            if model_type == "regression":
                return {"predict": wrapper.predictRegression(r).value}
            elif model_type == "binomial":
                probs = wrapper.predictBinomial(r).classProbabilities
                return {"predict": wrapper.predictBinomial(r).label, "p0": probs[0], "p1": probs[1]}
            elif model_type == "multinomial":
                probs = list(wrapper.predictMultinomial(r).classProbabilities)
                return {"predict": wrapper.predictMultinomial(r).label, **{f"p{i}": p for i, p in enumerate(probs)}}
            else:
                raise ValueError(f"Unknown model_type: {model_type!r}. Use 'regression', 'binomial', or 'multinomial'.")

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(score_row, [row for _, row in dataframe.iterrows()]))
        return pd.DataFrame(results, index=dataframe.index)

    def _score(self, data_flattened: pd.DataFrame, options: Dict, context: PythonModelContext) -> pd.Series:
        if data_flattened.shape[0] <= options.get("h2o_jvm_threshold", 100):
            result_df = self._mojo_predict_pandas(
                dataframe=data_flattened,
                wrapper=self.wrapper,
                model_type=self.model_type,
                n_threads=self.n_threads,
            )
        else:
            # Deliberately inside the `else`, not at the top of `_score`: the JPype branch
            # above is the common one and must not pay for this import.
            import h2o

            result_df = h2o.mojo_predict_pandas(
                dataframe=data_flattened,
                mojo_zip_path=context.artifacts[self.mojo_artifact_key],
                genmodel_jar_path=context.artifacts[self.genmodel_artifact_key],
            )
        if self.model_type == "regression":
            return result_df["predict"]
        elif self.model_type == "binomial":
            return result_df["p1"]
        else:  # multinomial
            return result_df.iloc[:, -1]

    def _compute_result(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: PythonModelContext = None,
    ) -> pd.Series:
        if self.wrapper is None and context is not None:
            self.wrapper = self._load_model(context.artifacts[self.mojo_artifact_key], context.artifacts[self.genmodel_artifact_key])

        target_df = data[self.target_table]

        required = list(self.column_map.keys())
        sources = [self.column_map[c] for c in required]
        missing = [
            f"{src} (for required '{req}')"
            for req, src in zip(required, sources)
            if src not in target_df.columns
        ]
        if missing:
            raise LayerCalculationError(
                f"Layer '{self.name}': source columns missing from '{self.target_table}': {missing}"
            )
        # Pull raw inputs onto a working frame and apply the modelling-time
        # preprocessing recipe (formulas + in-place per-column transforms) before
        # handing the frame to the MOJO. The recipe is empty for layers created
        # via manual MOJO upload — those paths score directly off ``column_map``.
        scoring_df = pd.DataFrame(index=target_df.index)
        for req, src in zip(required, sources):
            scoring_df[req] = target_df[src]
        scoring_df = self._apply_derived(scoring_df)
        scoring_df = self._apply_preprocessing(scoring_df)

        try:
            result = self._score(data_flattened=scoring_df, options=options, context=context)
        except Exception as exc:
            raise LayerCalculationError(
                f"Layer '{self.name}': H2O scoring failed: {exc}"
            ) from exc

        result = result.reset_index(drop=True)
        result.index = target_df.index
        return result

    def _apply_derived(self, working: pd.DataFrame) -> pd.DataFrame:
        """Evaluate formula-derived columns in order, mirroring fit-time
        derived-column evaluation (bare column names +
        ``np`` / ``pd``). New columns are appended to the frame so subsequent
        derived columns and preprocessing entries can reference them."""
        if not self.derived:
            return working
        out = working.copy() if not working.empty else working
        for d in self.derived:
            dname, expr = d["name"], d["expression"]
            namespace = {col: out[col] for col in out.columns}
            namespace["np"] = np
            namespace["pd"] = pd
            namespace["df"] = out
            try:
                result = eval(  # noqa: S307 - restricted builtins, governance-gated
                    expr, {"__builtins__": ColumnCalculation._safe_builtins()}, namespace,
                )
            except Exception as exc:  # noqa: BLE001
                raise LayerCalculationError(
                    f"Layer '{self.name}': formula column '{dname}': {exc}"
                ) from exc
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=out.index)
            out[dname] = result.to_numpy()
        return out

    def _apply_preprocessing(self, working: pd.DataFrame) -> pd.DataFrame:
        """Apply each preprocessing entry in order. Transforms are in-place — the
        output column has the same name as the source — matching how the
        modelling pipeline mutates columns before pushing to H2O at fit time
        (preprocessing + binning in the training pipeline)."""
        if not self.preprocessing:
            return working
        for p in self.preprocessing:
            kind = p.get("type")
            col = p.get("column")
            if col not in working.columns:
                raise LayerCalculationError(
                    f"Layer '{self.name}': preprocessing column '{col}' missing from working frame"
                )
            if kind == "bin":
                edges = list(p.get("edges") or [])
                oor = p.get("out_of_range_label", "out_of_range")
                numeric = pd.to_numeric(working[col], errors="coerce")
                binned = pd.cut(numeric, bins=edges, include_lowest=True)
                labels = binned.astype("object").astype("string").fillna(oor)
                working[col] = labels.to_numpy()
            elif kind == "bin_thermometer":
                # Monotone-across-bins: the MOJO was trained on cumulative
                # indicator columns ({col}__ge_k), not the bin label. Regenerate
                # them here with the shared encoder so score == train. The raw
                # ``col`` is left in place; the MOJO only references the indicators.
                edges = list(p.get("edges") or [])
                working = apply_thermometer_encoding(working, col, edges)
            elif kind == "log_transform":
                working[col] = np.log(pd.to_numeric(working[col], errors="coerce"))
            elif kind == "group_levels":
                # Mirror fit-time ``derived.apply_level_combining``: canonicalise each
                # value, then map it to its group name (or pass the canonical label
                # through if unmapped). Missing → None so the MOJO reads it as NA.
                mapping = p.get("mapping") or {}

                def _remap(value, _m=mapping):
                    key = _canonical_level_label(value)
                    if key is None:
                        return None
                    return _m.get(key, key)

                working[col] = working[col].map(_remap)
            else:
                raise LayerCalculationError(
                    f"Layer '{self.name}': unknown preprocessing type {kind!r}"
                )
        return working

    def _layer_params(self) -> dict:
        params: dict = {
            "target_table": self.target_table,
            "output_column": self.output_column,
            "mojo_artifact_key": self.mojo_artifact_key,
            "genmodel_artifact_key": self.genmodel_artifact_key,
            "model_type": self.model_type,
            "n_threads": self.n_threads,
            "column_map": self.column_map,
        }
        if self.derived:
            params["derived"] = self.derived
        if self.preprocessing:
            params["preprocessing"] = self.preprocessing
        return params

    @classmethod
    def _from_dict(cls, d: dict) -> "H2OScoringParameter":
        column_map = d.get("column_map")
        # Legacy: derive identity column_map from model_columns if present.
        if column_map is None and d.get("model_columns"):
            column_map = {c: c for c in d["model_columns"]}
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            output_column=d["output_column"],
            column_map=column_map,
            mojo_artifact_key=d["mojo_artifact_key"],
            genmodel_artifact_key=d["genmodel_artifact_key"],
            model_type=d.get("model_type", "binomial"),
            n_threads=d.get("n_threads", 4),
            derived=d.get("derived"),
            preprocessing=d.get("preprocessing"),
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"H2OScoringParameter(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"output={self.output_column!r}, "
            f"model_type={self.model_type!r})"
        )


# ---------------------------------------------------------------------------
# OnnxScoring
# ---------------------------------------------------------------------------

# Map ONNX-runtime input type strings (e.g. "tensor(float)") to numpy dtypes.
# Each source column is cast to the graph's declared input dtype before scoring;
# onnxruntime raises if a fed array's dtype doesn't match the input spec.
_ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
    "tensor(string)": np.str_,
}


class OnnxScoring(ColumnCalculation):
    """
    A pipeline layer that scores rows in a dataset table with an ONNX model and
    writes the predictions as a new column.

    The model runs under :mod:`onnxruntime`. Source columns are selected from
    ``data[target_table]`` and renamed to the model's input names via
    ``column_map`` before scoring — so the layer works even when the live source
    data names a feature differently from how it was named at training time.

    Parameters
    ----------
    name : str
        Unique identifier for this layer.
    target_table : str
        Key in the ``data`` dict whose rows will be scored.
    output_column : str
        Name of the column written with the model's predictions.
    column_map : dict[str, str]
        Mapping ``{model_input_name: source_column_in_target_table}``. The keys
        are the ONNX graph's input names (the canonical required-column list);
        the values are the columns read from ``target_table``.
    model_artifact_key : str
        Key in ``context.artifacts`` pointing to the exported ``.onnx`` file.
    dependencies : list[str], optional
    description : str, optional

    Notes
    -----
    The first model output (``session.run(None, feed)[0]``) is taken as the
    prediction and flattened to one value per row. Models exported with the link
    function baked into the graph (e.g. XGBoost via onnxmltools) therefore return the
    final prediction as their first output.
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        output_column: str,
        column_map: Dict[str, str],
        model_artifact_key: str,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        if not column_map or not isinstance(column_map, dict):
            raise ValueError("column_map must be a non-empty {model_input: source} dict")

        super().__init__(
            name=name,
            target_table=target_table,
            output_column=output_column,
            calculation=lambda _df, _data, _catalog, _options: None,  # never called
            dependencies=dependencies,
            description=description,
        )

        self.model_artifact_key = model_artifact_key
        self.column_map = column_map
        self.session = None
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "output_column": output_column,
            "model_artifact_key": model_artifact_key,
            "column_map": column_map,
            "dependencies": dependencies,
            "description": description,
        }

    @property
    def model_columns(self) -> List[str]:
        return list(self.column_map.keys())

    @staticmethod
    def _load_model(onnx_path: str):
        import onnxruntime as ort

        # CPUExecutionProvider keeps scoring deterministic and dependency-light.
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def init_from_context(self, context: PythonModelContext) -> None:
        self.session = self._load_model(context.artifacts[self.model_artifact_key])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["session"] = None
        return state

    def __deepcopy__(self, memo):
        return _deepcopy_sharing(self, memo, "session")

    def _score(
        self, scoring_df: pd.DataFrame, context: PythonModelContext = None
    ) -> pd.Series:
        # Lazy-load on first use (mirrors H2O's wrapper fallback) so the layer
        # works whether or not the pipeline called init_from_context.
        if self.session is None:
            if context is None:
                raise LayerCalculationError(
                    f"Layer '{self.name}': ONNX session not loaded and no context provided"
                )
            self.session = self._load_model(context.artifacts[self.model_artifact_key])

        feed = {}
        for inp in self.session.get_inputs():
            np_dtype = _ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
            feed[inp.name] = (
                scoring_df[inp.name].to_numpy().astype(np_dtype).reshape(-1, 1)
            )

        outputs = self.session.run(None, feed)
        return pd.Series(np.asarray(outputs[0]).reshape(-1), index=scoring_df.index)

    def _compute_result(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: PythonModelContext = None,
    ) -> pd.Series:
        target_df = data[self.target_table]

        required = list(self.column_map.keys())
        sources = [self.column_map[c] for c in required]
        missing = [
            f"{src} (for input '{req}')"
            for req, src in zip(required, sources)
            if src not in target_df.columns
        ]
        if missing:
            raise LayerCalculationError(
                f"Layer '{self.name}': source columns missing from '{self.target_table}': {missing}"
            )
        scoring_df = target_df[sources].rename(
            columns={self.column_map[c]: c for c in required}
        )

        try:
            result = self._score(scoring_df=scoring_df, context=context)
        except LayerCalculationError:
            raise
        except Exception as exc:
            raise LayerCalculationError(
                f"Layer '{self.name}': ONNX scoring failed: {exc}"
            ) from exc

        result = result.reset_index(drop=True)
        result.index = target_df.index
        return result

    def _layer_params(self) -> dict:
        return {
            "target_table": self.target_table,
            "output_column": self.output_column,
            "model_artifact_key": self.model_artifact_key,
            "column_map": self.column_map,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "OnnxScoring":
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            output_column=d["output_column"],
            column_map=d.get("column_map"),
            model_artifact_key=d["model_artifact_key"],
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OnnxScoring(name={self.name!r}, "
            f"target={self.target_table!r}, "
            f"output={self.output_column!r}, "
            f"model={self.model_artifact_key!r})"
        )


# ---------------------------------------------------------------------------
# GlmScoring
# ---------------------------------------------------------------------------

# Label for binned values outside the outer edges. Must match the value used at
# fit time so bin labels line up with the trained coefficient levels — source of
# truth is the training pipeline's out-of-range label.
_GLM_OUT_OF_RANGE_LABEL = "out_of_range"
_GLM_VALID_LINKS = ("log", "identity", "logit", "inverse")
_GLM_VALID_TREATMENTS = ("raw", "log_transform", "as_factor", "binned")


class GlmScoring(ColumnCalculation):
    """Score a GLM (or any additive model) as a closed-form function — no MOJO,
    no JVM at runtime.

    A fitted GLM is ``mu = link^-1(intercept + sum_j term_j)`` where each term is
    a per-predictor contribution evaluated from the **raw** input:

    - ``raw``           -> ``coef * x``
    - ``log_transform`` -> ``coef * log(x)``
    - ``as_factor``     -> ``levels[str(x)]``        (one coefficient per level)
    - ``binned``        -> ``levels[bin_label(x)]``  where the bin label is
      reproduced with the **exact** fit-time ``pd.cut(..., include_lowest=True)``
      semantics (out-of-range -> ``out_of_range_label``).

    The layer takes only raw inputs and bins internally, so the binning can't
    drift from training. Unseen categorical levels / out-of-range bins with no
    learned coefficient contribute ``0`` (the model's reference/baseline).

    ``terms`` is intentionally polymorphic on ``treatment`` so a future
    ``spline`` term (GAM) is purely additive — the intercept / inverse-link /
    offset / breakdown machinery is shared across all term types.

    Parameters
    ----------
    name, target_table, output_column : str
    intercept : float
        The GLM intercept (``beta_0``), on the link scale.
    link : str
        Inverse link applied to the linear predictor: one of
        ``"log"``, ``"identity"``, ``"logit"``, ``"inverse"``.
    terms : list[dict]
        One entry per predictor (see treatments above). ``raw``/``log_transform``
        carry a numeric ``coef``; ``as_factor``/``binned`` carry a ``levels``
        map; ``binned`` also carries ``bin_edges``.
    column_map : dict[str, str], optional
        ``{raw_input_column: source_column}`` mapping the GLM's raw leaf inputs
        (predictor columns *before* any formula/binning, plus the columns
        referenced inside formulas) to source columns on ``target_table``. When
        omitted, terms are read directly from ``target_table`` by name (legacy).
    derived : list[dict], optional
        Formula columns ``[{"name", "expression"}]`` evaluated in order from the
        mapped raw inputs *before* binning — mirroring fit-time
        derived-column evaluation (bare column names +
        ``np``/``pd``). A term whose ``name`` is a derived column reads the
        computed value.
    offset_column : str | None
        If set, this column (on the link scale, e.g. ``log(exposure)``) is added
        to the linear predictor — matching how the GLM was fit with an offset.
    out_of_range_label : str
        Label for out-of-range binned values (must match fit time).
    unseen_policy : str
        Behaviour for unseen levels / out-of-range bins. Only ``"baseline"``
        (contribute 0) is implemented; reserved for a future refer/decline path.
    dependencies, description : optional
    """

    def __init__(
        self,
        name: str,
        target_table: str,
        output_column: str,
        intercept: float,
        link: str,
        terms: List[Dict],
        column_map: Optional[Dict[str, str]] = None,
        derived: Optional[List[Dict]] = None,
        offset_column: Optional[str] = None,
        out_of_range_label: str = _GLM_OUT_OF_RANGE_LABEL,
        unseen_policy: str = "baseline",
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ):
        if link not in _GLM_VALID_LINKS:
            raise ValueError(f"link must be one of {_GLM_VALID_LINKS}, got {link!r}")
        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        for t in terms:
            self._validate_term(t)
        derived = derived or []
        for d in derived:
            if not isinstance(d, dict) or not d.get("name") or not isinstance(d.get("expression"), str):
                raise ValueError("each derived entry needs a 'name' and a string 'expression'")

        super().__init__(
            name=name,
            target_table=target_table,
            output_column=output_column,
            calculation=lambda _df, _data, _catalog, _options: None,  # never called
            dependencies=dependencies,
            description=description,
        )

        self.intercept = float(intercept)
        self.link = link
        self.terms = terms
        self.column_map = dict(column_map or {})
        self.derived = list(derived)
        self.offset_column = offset_column
        self.out_of_range_label = out_of_range_label
        self.unseen_policy = unseen_policy
        self.parameters = {
            "name": name,
            "target_table": target_table,
            "output_column": output_column,
            "intercept": self.intercept,
            "link": link,
            "terms": terms,
            "column_map": self.column_map,
            "derived": self.derived,
            "offset_column": offset_column,
            "out_of_range_label": out_of_range_label,
            "unseen_policy": unseen_policy,
            "dependencies": dependencies,
            "description": description,
        }

    @staticmethod
    def _validate_term(t: Dict) -> None:
        if not isinstance(t, dict):
            raise ValueError("each term must be a dict")
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each term needs a non-empty 'name'")
        treatment = t.get("treatment")
        if treatment not in _GLM_VALID_TREATMENTS:
            raise ValueError(
                f"term '{name}': treatment must be one of {_GLM_VALID_TREATMENTS}, got {treatment!r}"
            )
        if treatment == "binned":
            edges = t.get("bin_edges")
            if not isinstance(edges, (list, tuple)) or len(edges) < 2:
                raise ValueError(f"term '{name}': binned term needs at least two bin_edges")
            if not isinstance(t.get("levels"), dict):
                raise ValueError(f"term '{name}': binned term needs a 'levels' map")
        elif treatment == "as_factor":
            if not isinstance(t.get("levels"), dict):
                raise ValueError(f"term '{name}': as_factor term needs a 'levels' map")
        else:  # raw / log_transform
            if not isinstance(t.get("coef"), (int, float)):
                raise ValueError(f"term '{name}': {treatment} term needs a numeric 'coef'")

    def _build_working_frame(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve the GLM's raw inputs onto a working frame, then evaluate any
        formula (derived) columns in order — mirroring fit-time
        ``apply_derived_columns`` (bare column names + ``np``/``pd``). The result
        carries every column a term reads (raw leaves + derived columns)."""
        if self.column_map:
            missing = [
                f"{src} (for '{leaf}')"
                for leaf, src in self.column_map.items()
                if src not in target_df.columns
            ]
            if missing:
                raise LayerCalculationError(
                    f"Layer '{self.name}': source columns missing from "
                    f"'{self.target_table}': {missing}"
                )
            working = pd.DataFrame(index=target_df.index)
            for leaf, src in self.column_map.items():
                working[leaf] = target_df[src]
        else:
            # Legacy / no mapping: read terms directly from the target table by name.
            working = target_df.copy() if self.derived else target_df

        for d in self.derived:
            dname, expr = d["name"], d["expression"]
            namespace = {col: working[col] for col in working.columns}
            namespace["np"] = np
            namespace["pd"] = pd
            namespace["df"] = working
            try:
                result = eval(  # noqa: S307 - restricted builtins, governance-gated
                    expr, {"__builtins__": ColumnCalculation._safe_builtins()}, namespace
                )
            except Exception as exc:  # noqa: BLE001
                raise LayerCalculationError(
                    f"Layer '{self.name}': formula column '{dname}': {exc}"
                ) from exc
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=working.index)
            working[dname] = result.to_numpy()

        return working

    def _term_contribution(self, term: Dict, src: pd.Series) -> pd.Series:
        treatment = term["treatment"]
        if treatment == "raw":
            return pd.to_numeric(src, errors="coerce") * float(term["coef"])
        if treatment == "log_transform":
            return np.log(pd.to_numeric(src, errors="coerce")) * float(term["coef"])
        if treatment == "as_factor":
            return src.astype("string").map(term["levels"])
        if treatment == "binned":
            # Reproduce fit-time binning verbatim
            # so labels line up with the trained coefficient levels.
            numeric = pd.to_numeric(src, errors="coerce")
            binned = pd.cut(numeric, bins=list(term["bin_edges"]), include_lowest=True)
            labels = binned.astype("object").astype("string")
            labels = labels.fillna(self.out_of_range_label)
            return labels.map(term["levels"])
        raise LayerCalculationError(
            f"Layer '{self.name}': unknown term treatment {treatment!r}"
        )

    @staticmethod
    def _inverse_link(eta: pd.Series, link: str) -> pd.Series:
        if link == "identity":
            return eta
        with np.errstate(over="ignore", divide="ignore"):
            if link == "log":
                return np.exp(eta)
            if link == "logit":
                return 1.0 / (1.0 + np.exp(-eta))
            if link == "inverse":
                return 1.0 / eta
        raise LayerCalculationError(f"Unsupported link {link!r}")

    def _compute_result(
        self,
        data: Dict[str, pd.DataFrame],
        catalog: Dict,
        options: Dict,
        context: "PythonModelContext" = None,
    ) -> pd.Series:
        target_df = data[self.target_table]
        working = self._build_working_frame(target_df)

        if self.offset_column and self.offset_column not in target_df.columns:
            raise LayerCalculationError(
                f"Layer '{self.name}': offset column '{self.offset_column}' missing "
                f"from '{self.target_table}'"
            )
        missing_terms = [t["name"] for t in self.terms if t["name"] not in working.columns]
        if missing_terms:
            raise LayerCalculationError(
                f"Layer '{self.name}': term inputs missing after mapping/formulas: {missing_terms}"
            )

        eta = pd.Series(self.intercept, index=target_df.index, dtype="float64")
        contribs: Dict[str, pd.Series] = {}
        for term in self.terms:
            src = working[term["name"]]
            contrib = pd.to_numeric(self._term_contribution(term, src), errors="coerce").fillna(0.0)
            contribs[term["name"]] = contrib
            eta = eta + contrib

        if self.offset_column:
            offset = pd.to_numeric(target_df[self.offset_column], errors="coerce").fillna(0.0)
            eta = eta + offset

        mu = self._inverse_link(eta, self.link)
        mu = pd.Series(np.asarray(mu, dtype="float64"), index=target_df.index)

        # Premium-breakdown seam (off by default — v1 output is unchanged). When
        # enabled, write per-term contributions + the linear predictor alongside.
        if (options or {}).get("emit_breakdown"):
            out = self.output_column
            data[self.target_table] = target_df = target_df.copy()
            target_df[f"{out}__contrib__intercept"] = self.intercept
            for tname, contrib in contribs.items():
                target_df[f"{out}__contrib__{tname}"] = contrib.values
            target_df[f"{out}__eta"] = eta.values

        return mu

    def _layer_params(self) -> dict:
        return {
            "target_table": self.target_table,
            "output_column": self.output_column,
            "intercept": self.intercept,
            "link": self.link,
            "terms": self.terms,
            "column_map": self.column_map,
            "derived": self.derived,
            "offset_column": self.offset_column,
            "out_of_range_label": self.out_of_range_label,
            "unseen_policy": self.unseen_policy,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "GlmScoring":
        return cls(
            name=d["name"],
            target_table=d["target_table"],
            output_column=d["output_column"],
            intercept=d.get("intercept", 0.0),
            link=d.get("link", "log"),
            terms=d.get("terms", []),
            column_map=d.get("column_map"),
            derived=d.get("derived"),
            offset_column=d.get("offset_column"),
            out_of_range_label=d.get("out_of_range_label", _GLM_OUT_OF_RANGE_LABEL),
            unseen_policy=d.get("unseen_policy", "baseline"),
            dependencies=d.get("dependencies"),
            description=d.get("description", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GlmScoring(name={self.name!r}, target={self.target_table!r}, "
            f"output={self.output_column!r}, link={self.link!r}, "
            f"n_terms={len(self.terms)})"
        )
