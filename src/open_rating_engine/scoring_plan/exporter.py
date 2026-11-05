"""Export a pricing pipeline as a scoring plan for the Rust scorer.

The plan is a JSON description of the pipeline: one step per layer, in the order
the wave executor runs them, with every expression compiled to a small tree (see
:mod:`.expressions`). A plan bundle is a directory holding ``plan.json`` and the
ONNX files the plan scores with::

    bundle/
        plan.json
        models/<artifact_key>.onnx

Layers the plan cannot express are reported as :class:`Blocker` objects instead of
being guessed at. A pipeline with any blocker has no plan and keeps the Python
scoring image.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..rating_engine.pricing_layers import (
    ColumnCalculation,
    CustomLayer,
    DropColumns,
    GlmScoring,
    H2OScoringParameter,
    OnnxScoring,
    RatingTablesLayer,
    RenameColumns,
    SourceColumn,
    TableAggregation,
    TableDrop,
    TableJoin,
    TableTransform,
    _ALLOWED_AGG_FUNCS,
)
from .expressions import (
    ExpressionScope,
    UnsupportedExpression,
    compile_expression,
    const,
    encode_float,
)

__all__ = [
    "PLAN_FORMAT",
    "PLAN_VERSION",
    "Blocker",
    "ExportResult",
    "PlanExportError",
    "export_plan",
    "write_plan_bundle",
]

PLAN_FORMAT = "open-rating-plan"
PLAN_VERSION = 1


@dataclass(frozen=True)
class Blocker:
    """A layer (or pipeline setting) the scoring plan cannot express."""

    layer: Optional[str]
    layer_type: Optional[str]
    reason: str

    def __str__(self) -> str:
        where = f"{self.layer} ({self.layer_type})" if self.layer else "pipeline"
        return f"{where}: {self.reason}"


@dataclass
class ExportResult:
    """Outcome of :func:`export_plan`: a plan, or the reasons there is none."""

    plan: Optional[dict]
    blockers: List[Blocker] = field(default_factory=list)
    model_files: Dict[str, str] = field(default_factory=dict)

    @property
    def exportable(self) -> bool:
        return self.plan is not None


class PlanExportError(Exception):
    """Raised by :func:`write_plan_bundle` when the pipeline has blockers."""

    def __init__(self, blockers: List[Blocker]):
        self.blockers = blockers
        lines = "\n".join(f"  - {b}" for b in blockers)
        super().__init__(f"The pipeline cannot be exported as a scoring plan:\n{lines}")


def export_plan(pipeline, artifacts: Optional[Dict[str, str]] = None) -> ExportResult:
    """Compile ``pipeline`` into a scoring plan.

    Parameters
    ----------
    pipeline : PyfuncPricingPipeline
    artifacts : dict[str, str], optional
        ``{artifact_key: local_path}`` for the ONNX models. Defaults to
        ``pipeline.context.artifacts``.

    Returns
    -------
    ExportResult
        ``plan`` is ``None`` when any blocker was found; ``blockers`` lists every
        one, not just the first.
    """
    artifacts = dict(artifacts if artifacts is not None else getattr(pipeline.context, "artifacts", {}) or {})
    blockers: List[Blocker] = []
    if getattr(pipeline, "pydantic_model", None) is not None:
        blockers.append(Blocker(None, None, "pydantic_model schema validation is not supported"))
    if pipeline.options.get("emit_breakdown"):
        blockers.append(Blocker(None, None, "options['emit_breakdown'] is not supported"))

    steps: List[dict] = []
    model_files: Dict[str, str] = {}
    exporter = _LayerExporter(pipeline, artifacts, model_files)
    for name in pipeline._canonical_order():
        layer = pipeline.layers[name]
        try:
            step = exporter.export(layer)
        except UnsupportedExpression as exc:
            blockers.append(Blocker(name, type(layer).__name__, str(exc)))
            continue
        if step is not None:
            steps.append(step)

    if blockers:
        return ExportResult(plan=None, blockers=blockers)

    accessors = [
        [layer.source_table, layer.source_column]
        for layer in pipeline.layers.values()
        if type(layer) is SourceColumn
    ]
    plan = {
        "format": PLAN_FORMAT,
        "version": PLAN_VERSION,
        "source_accessors": accessors,
        "models": {
            key: {"path": f"models/{key}.onnx", "sha256": _sha256(Path(path))}
            for key, path in sorted(model_files.items())
        },
        "steps": steps,
    }
    return ExportResult(plan=plan, model_files=model_files)


def write_plan_bundle(
    pipeline,
    out_dir: Union[str, Path],
    artifacts: Optional[Dict[str, str]] = None,
) -> Path:
    """Export ``pipeline`` and write ``plan.json`` plus its models into ``out_dir``.

    Raises :class:`PlanExportError` listing every blocker when the pipeline cannot
    be exported. Returns the path of ``plan.json``.
    """
    result = export_plan(pipeline, artifacts)
    if not result.exportable:
        raise PlanExportError(result.blockers)
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    for key, src in result.model_files.items():
        shutil.copyfile(src, out_dir / result.plan["models"][key]["path"])
    plan_path = out_dir / "plan.json"
    plan_path.write_text(
        json.dumps(result.plan, indent=1, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return plan_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_table(df: pd.DataFrame) -> dict:
    """A DataFrame as a plan table: typed columns, in order."""
    columns = []
    for name in df.columns:
        if not isinstance(name, str):
            raise UnsupportedExpression(f"table column {name!r} must be named by a string")
        columns.append({"name": name, **encode_column(df[name])})
    return {"columns": columns}


def encode_column(series: pd.Series) -> dict:
    """A Series as ``{"dtype", "values"}`` in the plan's column encoding."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return {"dtype": "bool", "values": [bool(v) for v in series]}
    if pd.api.types.is_integer_dtype(dtype):
        return {"dtype": "i64", "values": [int(v) for v in series]}
    if dtype == np.float32:
        return {"dtype": "f32", "values": [encode_float(v) for v in series]}
    if pd.api.types.is_float_dtype(dtype):
        return {"dtype": "f64", "values": [encode_float(v) for v in series]}
    values = []
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            values.append(None)
        elif isinstance(v, str):
            values.append(v)
        else:
            raise UnsupportedExpression(
                f"column {series.name!r} mixes strings with {type(v).__name__} values"
            )
    return {"dtype": "str", "values": values}


def _key_value(value):
    """A rating-table key as JSON, keeping its Python type."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return encode_float(value)
    if isinstance(value, str):
        return value
    raise UnsupportedExpression(f"rating table key {value!r} of type {type(value).__name__} is not supported")


class _LayerExporter:
    def __init__(self, pipeline, artifacts: Dict[str, str], model_files: Dict[str, str]):
        self.pipeline = pipeline
        self.artifacts = artifacts
        self.model_files = model_files
        self.data_structure = list(pipeline.options.get("data_structure", []) or [])

    def export(self, layer) -> Optional[dict]:
        kind = type(layer)
        handler = {
            ColumnCalculation: self._column,
            RatingTablesLayer: self._rating_tables,
            OnnxScoring: self._onnx,
            GlmScoring: self._glm,
            TableJoin: self._join,
            TableAggregation: self._aggregation,
            RenameColumns: self._rename,
            DropColumns: self._drop_columns,
            TableDrop: self._drop_tables,
            SourceColumn: lambda _layer: None,
        }.get(kind)
        if handler is not None:
            return handler(layer)
        if kind is H2OScoringParameter:
            raise UnsupportedExpression(
                "H2O MOJO models need a JVM; convert the model to ONNX or GlmScoring"
            )
        if kind is CustomLayer:
            raise UnsupportedExpression("CustomLayer runs arbitrary Python")
        if kind is TableTransform:
            raise UnsupportedExpression("TableTransform runs an arbitrary Python function")
        raise UnsupportedExpression(f"layer type {kind.__name__} is not supported")

    def _column(self, layer: ColumnCalculation) -> dict:
        if not isinstance(layer.calculation, str):
            raise UnsupportedExpression(
                "calculation is a Python function; write it as an expression string to export it"
            )
        scope = ExpressionScope(
            "column",
            catalog=self.pipeline.catalog,
            options=self.pipeline.options,
            target_table=layer.target_table,
        )
        return {
            "op": "column",
            "layer": layer.name,
            "table": layer.target_table,
            "column": layer.output_column,
            "expr": compile_expression(layer.calculation, scope, self.data_structure),
        }

    def _rating_tables(self, layer: RatingTablesLayer) -> dict:
        tables = []
        for name, frame in layer.rating_tables.items():
            keys = [c for c in frame.columns if c != "value"]
            value = frame["value"]
            if pd.api.types.is_integer_dtype(value.dtype) and not pd.api.types.is_bool_dtype(value.dtype):
                value_dtype, values = "i64", [int(v) for v in value]
            else:
                value_dtype, values = "f64", [encode_float(v) for v in value.astype("float64")]
            tables.append({
                "name": name,
                "keys": keys,
                "key_rows": [[_key_value(v) for v in row] for row in frame[keys].itertuples(index=False)],
                "value_dtype": value_dtype,
                "values": values,
                "default": encode_float(layer.default_values[name]),
            })
        formula = None
        if layer.formula is not None:
            scope = ExpressionScope("rating_formula", rating_tables=layer.rating_tables.keys())
            formula = compile_expression(layer.formula, scope)
        return {
            "op": "rating_tables",
            "layer": layer.name,
            "table": layer.target_table,
            "column": layer.output_column,
            "tables": tables,
            "formula": formula,
        }

    def _onnx(self, layer: OnnxScoring) -> dict:
        path = self.artifacts.get(layer.model_artifact_key)
        if not path or not Path(path).is_file():
            raise UnsupportedExpression(
                f"ONNX artifact {layer.model_artifact_key!r} was not found locally ({path!r})"
            )
        self.model_files[layer.model_artifact_key] = str(path)
        step = {
            "op": "onnx",
            "layer": layer.name,
            "table": layer.target_table,
            "column": layer.output_column,
            "model": layer.model_artifact_key,
            "inputs": [[model_input, source] for model_input, source in layer.column_map.items()],
        }
        if layer.derived:
            # Formula columns, compiled exactly as GlmScoring's (same evaluator in pandas).
            # With them, `inputs` maps working-frame names (leaves) to source columns and
            # every model input is read from the working frame by name. Emitted only when
            # present, so a plan without formulas is byte-identical to before.
            scope = ExpressionScope("glm_derived")
            step["derived"] = [
                {"name": d["name"], "expr": compile_expression(d["expression"], scope)}
                for d in layer.derived
            ]
        return step

    def _glm(self, layer: GlmScoring) -> dict:
        if layer.unseen_policy != "baseline":
            raise UnsupportedExpression(f"unseen_policy {layer.unseen_policy!r} is not supported")
        scope = ExpressionScope("glm_derived")
        derived = [
            {"name": d["name"], "expr": compile_expression(d["expression"], scope)}
            for d in layer.derived
        ]
        eta = const(layer.intercept, strong=True)
        for term in layer.terms:
            eta = {"k": "arith", "op": "add", "l": eta, "r": self._glm_term(layer, term)}
        return {
            "op": "glm",
            "layer": layer.name,
            "table": layer.target_table,
            "column": layer.output_column,
            "inputs": [[leaf, src] for leaf, src in layer.column_map.items()] if layer.column_map else None,
            "derived": derived,
            "eta": eta,
            "offset": layer.offset_column,
            "link": layer.link,
        }

    @staticmethod
    def _glm_term(layer: GlmScoring, term: dict) -> dict:
        """One term's contribution, as ``GlmScoring._compute_result`` computes it:
        the treatment, then ``to_numeric(...).fillna(0.0)``."""
        x = {"k": "col", "name": term["name"]}
        numeric = {"k": "call", "f": "to_numeric", "args": [x]}
        treatment = term["treatment"]
        if treatment == "raw":
            contrib = {"k": "arith", "op": "mul", "l": numeric, "r": const(float(term["coef"]))}
        elif treatment == "log_transform":
            log = {"k": "call", "f": "log", "args": [numeric]}
            contrib = {"k": "arith", "op": "mul", "l": log, "r": const(float(term["coef"]))}
        elif treatment == "as_factor":
            contrib = {
                "k": "str_lookup",
                "x": x,
                "mapping": {str(k): encode_float(v) for k, v in term["levels"].items()},
            }
        else:  # binned
            edges = [float(e) for e in term["bin_edges"]]
            if any(not a < b for a, b in zip(edges, edges[1:])):
                raise UnsupportedExpression(f"term {term['name']!r}: bin edges must be strictly increasing")
            # Bin labels exactly as pandas formats them, from the same bins object the
            # layer passes to pd.cut (integer and float edges format differently).
            empty = pd.cut(pd.Series([], dtype="float64"), bins=list(term["bin_edges"]), include_lowest=True)
            labels = [str(interval) for interval in empty.cat.categories]
            levels = term["levels"]
            missing = levels.get(layer.out_of_range_label, np.nan)
            contrib = {
                "k": "bin_lookup",
                "x": numeric,
                "edges": [encode_float(e) for e in edges],
                "right": True,
                "include_lowest": True,
                "values": [encode_float(levels.get(label, np.nan)) for label in labels],
                "missing": encode_float(missing),
            }
        contrib = {"k": "call", "f": "to_numeric", "args": [contrib]}
        return {"k": "call", "f": "fillna", "args": [contrib, const(0.0)]}

    def _join(self, layer: TableJoin) -> dict:
        if layer.how != "left":
            raise UnsupportedExpression(f"how={layer.how!r} joins are not supported; only 'left'")
        left_on = [layer.left_on] if isinstance(layer.left_on, str) else list(layer.left_on)
        right_on = [layer.right_on] if isinstance(layer.right_on, str) else list(layer.right_on)
        if len(left_on) != len(right_on):
            raise UnsupportedExpression("left_on and right_on must have the same length")
        if layer.right_source == "catalog":
            table = self.pipeline.catalog.get(layer.right_table)
            if not isinstance(table, pd.DataFrame):
                raise UnsupportedExpression(f"catalog table {layer.right_table!r} is not a DataFrame")
            right = {"catalog": encode_table(table)}
        else:
            right = {"table": layer.right_table}
        suffixes = list(layer.suffixes)
        if len(suffixes) != 2 or not all(isinstance(s, str) for s in suffixes):
            raise UnsupportedExpression("suffixes must be two strings")
        return {
            "op": "join",
            "layer": layer.name,
            "left": layer.left_table,
            "right": right,
            "left_on": left_on,
            "right_on": right_on,
            "output": layer.output_table,
            "columns": list(layer.columns) if layer.columns is not None else None,
            "suffixes": suffixes,
        }

    def _aggregation(self, layer: TableAggregation) -> dict:
        keys = [layer.groupby_key] if isinstance(layer.groupby_key, str) else list(layer.groupby_key)
        aggs = []
        for out, spec in layer.aggregations.items():
            if not (isinstance(spec, (tuple, list)) and len(spec) == 2):
                raise UnsupportedExpression(f"aggregation {out!r} must be (column, function)")
            column, func = spec
            if not isinstance(func, str) or func not in _ALLOWED_AGG_FUNCS:
                raise UnsupportedExpression(f"aggregation function {func!r} is not supported")
            aggs.append([out, column, func])
        return {
            "op": "aggregate_table",
            "layer": layer.name,
            "source": layer.source_table,
            "output": layer.output_table,
            "keys": keys,
            "aggs": aggs,
        }

    @staticmethod
    def _rename(layer: RenameColumns) -> dict:
        return {
            "op": "rename",
            "layer": layer.name,
            "table": layer.target_table,
            "columns": [[old, new] for old, new in layer.columns.items()],
        }

    @staticmethod
    def _drop_columns(layer: DropColumns) -> dict:
        return {"op": "drop_columns", "layer": layer.name, "table": layer.target_table, "columns": list(layer.columns)}

    @staticmethod
    def _drop_tables(layer: TableDrop) -> dict:
        return {"op": "drop_tables", "layer": layer.name, "tables": list(layer.tables)}
