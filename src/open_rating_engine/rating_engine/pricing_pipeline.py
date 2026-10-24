# Standard library imports
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

# Local application/library-specific imports
from typing import List, Dict, Callable, Optional, Union
from mlflow.pyfunc import PythonModel, PythonModelContext

# NOTE: no `import h2o` here. It was unused in this module — the H2O work all happens in
# `pricing_layers.H2OScoringParameter`, and MOJO scoring itself goes through JPype, not the
# H2O Python client. It cost every consumer a heavyweight import for nothing, on the cold
# start of every scale-to-zero scoring container.

from .pricing_layers import *

import tempfile
from urllib.parse import urlparse

def download_blob_to_temp(blob_url: str, conn_str: str = None) -> str:
    """Download a blob from Azure Blob Storage to a local temp directory.

    Accepts a full blob URL:
      https://<account>.blob.core.windows.net/<container>/<blob_path>

    Returns the absolute path to the downloaded file.

    Requires the optional ``azure`` extra: ``pip install open-rating-engine[azure]``.
    """
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "Azure Blob support requires the optional 'azure' extra. "
            "Install it with: pip install open-rating-engine[azure]"
        ) from exc

    parsed = urlparse(blob_url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if len(path_parts) != 2:
        raise ValueError(f"Cannot parse blob URL: {blob_url!r}")
    container_name, blob_name = path_parts
    filename = os.path.basename(blob_name)
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, filename)

    if conn_str is None:
        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    blob_client = BlobServiceClient.from_connection_string(conn_str).get_blob_client(
        container=container_name, blob=blob_name
    )

    with open(local_path, "wb") as f:
        blob_client.download_blob().readinto(f)

    return local_path

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PyfuncPricingPipeline(PythonModel):
    """
    A dependency-aware pipeline that applies a sequence of
    :class:`BasePricingLayer` objects to a dataset of named DataFrames.

    Layers are executed in topological order. Independent layers within a
    wave are executed in parallel (for :class:`ColumnCalculation` layers);
    structural layers (:class:`TableAggregation`, :class:`TableJoin`) in a
    wave are executed sequentially.
    """

    def __init__(
        self,
        context: PythonModelContext,
        pydantic_model=None,
        **kwargs,
    ):
        self.layers: Dict[str, BasePricingLayer] = {}
        self.catalog: Dict = {}
        self.options: Dict = {"show_progress": True}
        self.pydantic_model = pydantic_model
        self._execution_order: Optional[List[str]] = None

        if "azure_blob" in kwargs:
            self.options["azure_blob"] = kwargs["azure_blob"]

        if not isinstance(context, PythonModelContext):
            raise TypeError("context must be an instance of PythonModelContext")
        if not hasattr(context, "artifacts") or not hasattr(context, "model_config"):
            raise ValueError(
                "context must have 'artifacts' and 'model_config' attributes"
            )

        self.context = context
        self.extra_pip_requirements = None

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def add_parameter(self, layer: BasePricingLayer) -> "PyfuncPricingPipeline":
        """Register a layer, invalidating any cached execution order."""
        if not isinstance(layer, BasePricingLayer):
            raise TypeError(f"Expected a BasePricingLayer, got {type(layer)}")
        if layer.name in self.layers:
            warnings.warn(
                f"Layer '{layer.name}' already exists and will be overwritten.",
                UserWarning,
                stacklevel=2,
            )
        self.layers[layer.name] = layer
        self._execution_order = None
        return self

    def remove_parameter(self, layer_name: str) -> "PyfuncPricingPipeline":
        """Remove a registered layer by name, invalidating any cached execution order."""
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' is not registered in the pipeline.")
        del self.layers[layer_name]
        self._execution_order = None
        return self

    def load_context(self, context: PythonModelContext) -> None:

        import jpype
        import jpype.imports
        import sys

        for name, path in context.artifacts.items():
            if isinstance(path, str):
                context.artifacts[name] = path.replace("\\", "/")
            print(f'Load Context: Adding Artifact - {name}', file=sys.stderr)
            if bool(re.match(r"^https://[^/]+\.blob\.core\.windows\.net/", path)):
                print(f"Load Context: Downloading artifact {name} from Azure Blob Storage...", file=sys.stderr)
                local_path = download_blob_to_temp(path)
                print(f"Load Context: Downloaded {name} to {local_path}", file=sys.stderr)
                context.artifacts[name] = local_path.replace("\\", "/")

        

        self.context = context

        # Only the H2O MOJO scorer needs a JVM. Pipelines without H2O layers
        # (e.g. pure rating-table / column-calculation algorithms) must score
        # without one — and without requiring an 'h2o_gen_model' artifact.
        h2o_layers = [l for l in self.layers.values() if isinstance(l, H2OScoringParameter)]

        if h2o_layers and not jpype.isJVMStarted():
            genmodel = context.artifacts.get('h2o_gen_model')
            if not genmodel:
                raise ValueError(
                    "H2O scoring layer(s) present but no 'h2o_gen_model' artifact "
                    "was provided to load_context."
                )
            print('Load Context: Starting JVM...', file=sys.stderr)
            jpype.startJVM(
                context.model_config.get('jvm_path', "") or jpype.getDefaultJVMPath(),
                context.model_config.get('java_options', ""),
                classpath=[genmodel],
            )
        elif jpype.isJVMStarted():
            print('Load Context: JVM already started.', file=sys.stderr)

        print(f"Load Context: {context.artifacts}, {context.model_config}", file=sys.stderr)

        for layer in h2o_layers:
            layer.init_from_context(context)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def set_catalog(self, key: str, value) -> "PyfuncPricingPipeline":
        """Store a shared object available to all layers via ``catalog``."""
        self.catalog[key] = value
        return self

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def validate_dependencies(self) -> bool:
        for layer_name, layer in self.layers.items():
            for dep in layer.dependencies:
                if dep not in self.layers:
                    raise DependencyError(
                        f"Layer '{layer_name}' depends on '{dep}' "
                        f"which is not registered in the pipeline."
                    )

        dep_graph = {n: l.dependencies for n, l in self.layers.items()}
        if self._has_cycle(dep_graph):
            raise DependencyError(
                "Circular dependency detected in the pipeline."
            )

        return True

    @staticmethod
    def _has_cycle(graph: Dict[str, List[str]]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}

        def dfs(node: str) -> bool:
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                return False
            color[node] = GRAY
            for neighbour in graph.get(node, []):
                if dfs(neighbour):
                    return True
            color[node] = BLACK
            return False

        return any(dfs(node) for node in graph if color[node] == WHITE)

    def _topological_sort(self) -> List[str]:
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = defaultdict(int)

        for layer_name in self.layers:
            in_degree[layer_name] = 0

        for layer_name, layer in self.layers.items():
            for dep in layer.dependencies:
                graph[dep].append(layer_name)
                in_degree[layer_name] += 1

        queue = deque(n for n in in_degree if in_degree[n] == 0)
        result: List[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbour in graph[current]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(result) != len(self.layers):
            raise DependencyError(
                "Unable to resolve layer dependencies — circular dependency detected."
            )

        return result

    # ------------------------------------------------------------------
    # Wave builder
    # ------------------------------------------------------------------

    def _build_waves(self) -> List[List[str]]:
        order = self.get_execution_order()
        layer_wave: Dict[str, int] = {}
        for name in order:
            deps = self.layers[name].dependencies
            layer_wave[name] = (max(layer_wave[d] for d in deps) + 1) if deps else 0

        if not layer_wave:
            return []

        waves: List[List[str]] = [[] for _ in range(max(layer_wave.values()) + 1)]
        for name, w in layer_wave.items():
            waves[w].append(name)
        return waves

    def _run_columnar_wave(
        self,
        wave: List[str],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Run ColumnCalculation layers in a wave in parallel."""
        snapshot = {k: v.copy() for k, v in data.items()}
        results: Dict[str, pd.Series] = {}
        errors: Dict[str, Exception] = {}

        def _compute(name: str) -> tuple:
            return name, self.layers[name]._compute_result(snapshot, self.catalog, self.options, self.context)

        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {executor.submit(_compute, name): name for name in wave}
            for future in as_completed(futures):
                name = futures[future]
                exc = future.exception()
                if exc:
                    errors[name] = exc
                else:
                    _, series = future.result()
                    results[name] = series

        if errors:
            raise next(iter(errors.values()))

        for name in wave:
            layer = self.layers[name]
            data[layer.target_table] = data[layer.target_table].copy()
            data[layer.target_table][layer.output_column] = results[name].values

        return data

    def _run_wave_parallel(
        self,
        wave: List[str],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a wave of layers. ColumnCalculation layers run in parallel;
        TableAggregation and TableJoin layers run sequentially (they mutate
        the data dict structurally).
        """
        structural_types = (TableTransform, TableJoin, TableDrop, RenameColumns, DropColumns, CustomLayer, SourceColumn)
        columnar = [n for n in wave if not isinstance(self.layers[n], structural_types)]
        structural = [n for n in wave if isinstance(self.layers[n], structural_types)]

        if columnar:
            data = self._run_columnar_wave(columnar, data)

        for name in structural:
            data = self.layers[name].apply(data, self.catalog, self.options, self.context)

        return data

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_execution_order(self) -> List[str]:
        """Return layer names in the order they will be executed."""
        if self._execution_order is None:
            self.validate_dependencies()
            self._execution_order = self._topological_sort()
        return self._execution_order.copy()

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Return ``{layer_name: [dependency, ...]}`` for every layer."""
        return {name: layer.dependencies.copy() for name, layer in self.layers.items()}

    def get_layer_names(self) -> List[str]:
        """Return all registered layer names in insertion order."""
        return list(self.layers.keys())

    def to_config(self) -> dict:
        """Return a JSON-serializable snapshot of the pipeline's layers and context."""
        order = self.get_execution_order()
        return {
            "layers": {name: self.layers[name].to_dict() for name in order},
            "execution_order": order,
            "catalog": {
                key: {
                    "type": "table",
                    "data": {
                        "columns": list(value.columns),
                        "rows": value.values.tolist(),
                    },
                }
                if isinstance(value, pd.DataFrame)
                else {"type": "key_value_pairs", "data": value}
                for key, value in self.catalog.items()
            },
            "data_structure": list(self.options.get("data_structure", [])),
            "context": {
                "artifacts": dict(self.context.artifacts),
                "model_config": dict(self.context.model_config),
            },
        }

    @classmethod
    def from_config(cls, config: dict, context: PythonModelContext = None) -> "PyfuncPricingPipeline":
        """Reconstruct a pipeline from a config dict."""
        if context is None:
            ctx_data = config.get("context", {})
            context = PythonModelContext(
                artifacts=ctx_data.get("artifacts", {}),
                model_config=ctx_data.get("model_config", {"environment": "local"}),
            )
        pipeline = cls(context)
        for layer_dict in config.get("layers", {}).values():
            pipeline.add_parameter(BasePricingLayer.from_dict(layer_dict))
        for key, value in config.get("catalog", {}).items():
            if value['type'] == 'table':
                pipeline.set_catalog(key=key, value=pd.DataFrame(value["data"]["rows"], columns=value["data"]["columns"]))
            elif value['type'] == 'key_value_pairs':
                pipeline.set_catalog(key=key, value=value['data'])
        pipeline.options["data_structure"] = list(config.get("data_structure") or [])
        return pipeline

    def print_execution_plan(self) -> None:
        """Print a human-readable execution plan grouped by parallel wave."""
        print("Execution Plan")
        print("=" * 60)
        n = 1
        for wave_idx, wave in enumerate(self._build_waves(), 1):
            parallel = len(wave) > 1
            label = f"Wave {wave_idx}  (parallel — {len(wave)} layers)" if parallel else f"Wave {wave_idx}"
            print(f"\n── {label} {'─' * max(0, 50 - len(label))}")
            for layer_name in wave:
                layer = self.layers[layer_name]
                deps = ", ".join(layer.dependencies) if layer.dependencies else "—"

                if isinstance(layer, RatingTablesLayer):
                    print(f"  {n:>3}. [RATING] [{layer.target_table}] → {layer.output_column}")
                    print(f"         Layer:        {layer_name}")
                    print(f"         Factors:      {list(layer.rating_tables.keys())}")
                elif isinstance(layer, ColumnCalculation):
                    calc_repr = (
                        f'"{layer.calculation}"'
                        if isinstance(layer.calculation, str)
                        else getattr(layer.calculation, "__name__", repr(layer.calculation))
                    )
                    print(f"  {n:>3}. [COL]  [{layer.target_table}] → {layer.output_column}")
                    print(f"         Layer:        {layer_name}")
                    print(f"         Calculation:  {calc_repr}")
                elif isinstance(layer, TableAggregation):
                    print(f"  {n:>3}. [AGG]  {layer.source_table} → {layer.output_table}  (by {layer.groupby_key})")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, TableTransform):
                    print(f"  {n:>3}. [TRANSFORM] {layer.source_table} → {layer.output_table}")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, TableJoin):
                    print(f"  {n:>3}. [JOIN] {layer.left_table} + {layer.right_table} → {layer.output_table}  ({layer.left_on} = {layer.right_on})")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, RenameColumns):
                    print(f"  {n:>3}. [RENAME] [{layer.target_table}] {layer.columns}")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, DropColumns):
                    print(f"  {n:>3}. [DROP COLS] [{layer.target_table}] {layer.columns}")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, TableDrop):
                    print(f"  {n:>3}. [DROP] {', '.join(layer.tables)}")
                    print(f"         Layer:        {layer_name}")
                elif isinstance(layer, CustomLayer):
                    func_name = getattr(layer.func, "__name__", repr(layer.func))
                    print(f"  {n:>3}. [CUSTOM] {layer_name}  (func={func_name})")
                else:
                    print(f"  {n:>3}. [???]  {layer_name}")

                print(f"         Dependencies: {deps}")
                if layer.description:
                    print(f"         Description:  {layer.description}")
                n += 1

    def plot_dag(self, height: int = 600) -> None:
        """
        Render an interactive DAG of the pipeline using Plotly.

        Hover over a node to see its name, type, dependencies, and description.
        Supports zoom, pan, and reset via the Plotly toolbar.

        Nodes are colour-coded by layer type:

        - Blue   : ColumnCalculation / H2OScoringParameter
        - Green  : TableTransform / TableAggregation
        - Orange : TableJoin
        - Red    : TableDrop
        - Purple : RenameColumns

        Parameters
        ----------
        height : int
            Figure height in pixels. Defaults to 600.
        """
        import networkx as nx
        import plotly.graph_objects as go

        _COLOURS = {
            TableAggregation:  "#4CAF50",
            TableTransform:    "#4CAF50",
            TableJoin:         "#FF9800",
            TableDrop:         "#F44336",
            RenameColumns:     "#9C27B0",
            DropColumns:       "#9C27B0",
            RatingTablesLayer: "#00BCD4",
            ColumnCalculation: "#2196F3",
            CustomLayer:       "#795548",
        }

        def _colour(layer):
            for cls in type(layer).__mro__:
                if cls in _COLOURS:
                    return _COLOURS[cls]
            return "#9E9E9E"

        def _hover(layer):
            lines = [
                f"<b>{layer.name}</b>",
                f"Type: {type(layer).__name__}",
                f"Deps: {', '.join(layer.dependencies) if layer.dependencies else '—'}",
            ]
            if layer.description:
                lines.append(f"Desc: {layer.description}")
            if isinstance(layer, RatingTablesLayer):
                lines.append(f"Factors: {', '.join(layer.rating_tables.keys())}")
            elif isinstance(layer, ColumnCalculation):
                calc = (
                    layer.calculation
                    if isinstance(layer.calculation, str)
                    else getattr(layer.calculation, "__name__", "callable")
                )
                lines.append(f"Calc: {calc}")
            if isinstance(layer, (TableTransform, TableAggregation)):
                lines.append(f"Source → Output: {layer.source_table} → {layer.output_table}")
            if isinstance(layer, TableJoin):
                lines.append(f"Join: {layer.left_table} + {layer.right_table} → {layer.output_table}")
                lines.append(f"On: {layer.left_on} = {layer.right_on}")
            if isinstance(layer, TableDrop):
                lines.append(f"Drops: {', '.join(layer.tables)}")
            if isinstance(layer, RenameColumns):
                lines.append(f"Renames: {layer.columns}")
            if isinstance(layer, DropColumns):
                lines.append(f"Drops: {layer.columns}")
            return "<br>".join(lines)

        G = nx.DiGraph()
        for name in self.layers:
            G.add_node(name)
        for name, layer in self.layers.items():
            for dep in layer.dependencies:
                G.add_edge(dep, name)

        waves = self._build_waves()
        pos = {}
        for wave_idx, wave in enumerate(waves):
            for row_idx, name in enumerate(wave):
                pos[name] = (wave_idx, -(row_idx - (len(wave) - 1) / 2))

        # Build edge traces (one trace per edge for clean arrows)
        edge_traces = []
        for src, dst in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.5, color="#888888"),
                hoverinfo="none",
                showlegend=False,
            ))

        # Arrowhead markers at the destination end of each edge
        arrow_traces = []
        for src, dst in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            arrow_traces.append(go.Scatter(
                x=[x1], y=[y1],
                mode="markers",
                marker=dict(symbol="arrow", size=12, color="#888888",
                            angleref="previous"),
                hoverinfo="none",
                showlegend=False,
            ))

        # Node trace
        node_x, node_y, node_colours, node_labels, node_hover = [], [], [], [], []
        for name in G.nodes():
            x, y = pos[name]
            layer = self.layers[name]
            node_x.append(x)
            node_y.append(y)
            node_colours.append(_colour(layer))
            node_labels.append(name)
            node_hover.append(_hover(layer))

        # Group nodes by type for the legend
        seen, legend_traces = set(), []
        for name in G.nodes():
            layer = self.layers[name]
            type_name = type(layer).__name__
            colour = _colour(layer)
            if type_name not in seen:
                seen.add(type_name)
                legend_traces.append(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=12, color=colour),
                    name=type_name, showlegend=True,
                ))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(size=40, color=node_colours, line=dict(width=1.5, color="white")),
            showlegend=False,
        )

        fig = go.Figure(
            data=edge_traces + arrow_traces + legend_traces + [node_trace],
            layout=go.Layout(
                title=dict(text="Pipeline DAG", font=dict(size=16)),
                height=height,
                showlegend=True,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=50, b=20),
            )
        )
        fig.show()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _snapshot_source_accessors(self, data: Dict[str, pd.DataFrame]) -> None:
        """Materialise the raw fed-in value of every source column under the
        ``<column>{}`` accessor, before any wave runs.

        Every declared source column enters the pipeline as a :class:`SourceColumn`
        layer. Snapshotting its incoming value here — once, pre-wave — keeps the raw
        reachable via ``df['<col>{}']`` (Earnix-style) even after an override
        ``ColumnCalculation`` reclaims the plain name, with **no** dependency
        subtlety (the accessor is materialised pre-pipeline). When the column is
        *not* overridden the accessor simply equals the plain column. Mutates
        ``data`` in place; ``data`` has already been copied from the caller's
        input. Internal-only — stripped from the output by
        :meth:`_strip_source_accessors`.
        """
        for layer in self.layers.values():
            if not isinstance(layer, SourceColumn):
                continue
            frame = data.get(layer.source_table)
            if frame is None or layer.source_column not in frame.columns:
                continue
            accessor = f"{layer.source_column}{{}}"
            if accessor not in frame.columns:
                frame[accessor] = frame[layer.source_column]

    def _strip_source_accessors(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Drop every ``<col>{}`` accessor column from every frame.

        Accessors are internal-only — usable inside layer calculations but never
        exposed in the scored output. Called after the wave loop, before the
        return branch, so it covers both the local and the deployed-serving
        (array) paths.
        """
        for table, frame in data.items():
            drop = [c for c in frame.columns if isinstance(c, str) and c.endswith("{}")]
            if drop:
                data[table] = frame.drop(columns=drop)
        return data

    def predict(self, model_input: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run the pipeline against ``model_input``.

        Parameters
        ----------
        model_input : dict[str, pd.DataFrame]
            A dict mapping table names to DataFrames.

        Returns
        -------
        dict[str, pd.DataFrame]
            The same structure as ``model_input`` with all computed columns
            appended (and any new tables created by aggregation/join layers).
        """
        if self.context.model_config.get("environment") != 'local':
            print("Formatting input data for non-local environment...", file=sys.stderr)
            model_input = self._data_to_df(model_input)

        if self.pydantic_model is not None:
            try:
                self.pydantic_model.model_validate(model_input)
            except Exception as exc:
                raise ValueError(f"Schema validation failed: {exc}") from exc

        data: Dict[str, pd.DataFrame] = {
            k: v.copy() for k, v in model_input.items()
        }

        self._snapshot_source_accessors(data)

        waves = self._build_waves()
        wave_iterator = (
            tqdm(waves, total=len(waves), desc="Calculating Prices")
            if self.options.get("show_progress")
            else waves
        )

        for wave in wave_iterator:
            try:
                if len(wave) == 1:
                    data = self.layers[wave[0]].apply(data, self.catalog, self.options, self.context)
                else:
                    data = self._run_wave_parallel(wave, data)
            except (LayerCalculationError, RuntimeError):
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected error in wave {wave}: {exc}"
                ) from exc

        # Accessors are internal-only — never exposed in the scored output. Strip
        # before the return branch so both local and deployed-serving paths are covered.
        data = self._strip_source_accessors(data)

        if self.context.model_config.get("environment") != 'local':
            return self._data_to_array(data)
        else:
            return data

    def profile_predict(
        self, context, model_input: Dict[str, pd.DataFrame]
    ) -> tuple:
        """
        Run the pipeline and return a per-layer timing report.

        Columnar layers in a parallel wave are still executed concurrently,
        but each thread records its own elapsed time so individual costs are
        visible even when layers overlap on the wall clock.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            Output data dict and a DataFrame with columns
            (wave, layer, type, parallel, elapsed_s) sorted slowest first.
        """
        structural_types = (TableTransform, TableJoin, TableDrop, RenameColumns, DropColumns, CustomLayer, SourceColumn)

        if self.pydantic_model is not None:
            try:
                self.pydantic_model.model_validate(model_input)
            except Exception as exc:
                raise ValueError(f"Schema validation failed: {exc}") from exc

        data: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in model_input.items()}
        self._snapshot_source_accessors(data)
        waves = self._build_waves()
        report = []

        for wave_idx, wave in enumerate(waves):
            columnar = [n for n in wave if not isinstance(self.layers[n], structural_types)]
            structural = [n for n in wave if isinstance(self.layers[n], structural_types)]

            if columnar:
                snapshot = {k: v.copy() for k, v in data.items()}
                results: Dict[str, pd.Series] = {}
                timings: Dict[str, float] = {}
                errors: Dict[str, Exception] = {}

                def _timed_compute(name: str) -> tuple:
                    t0 = time.perf_counter()
                    series = self.layers[name]._compute_result(snapshot, self.catalog, self.options, self.context)
                    return name, series, time.perf_counter() - t0

                with ThreadPoolExecutor(max_workers=len(columnar)) as executor:
                    futures = {executor.submit(_timed_compute, n): n for n in columnar}
                    for future in as_completed(futures):
                        name = futures[future]
                        exc = future.exception()
                        if exc:
                            errors[name] = exc
                        else:
                            _, series, elapsed = future.result()
                            results[name] = series
                            timings[name] = elapsed

                if errors:
                    raise next(iter(errors.values()))

                for name in columnar:
                    layer = self.layers[name]
                    data[layer.target_table] = data[layer.target_table].copy()
                    data[layer.target_table][layer.output_column] = results[name].values
                    report.append({
                        "wave": wave_idx + 1,
                        "layer": name,
                        "type": type(layer).__name__,
                        "parallel": len(columnar) > 1,
                        "elapsed_s": timings[name],
                    })

            for name in structural:
                layer = self.layers[name]
                t0 = time.perf_counter()
                data = layer.apply(data, self.catalog, self.options, self.context)
                report.append({
                    "wave": wave_idx + 1,
                    "layer": name,
                    "type": type(layer).__name__,
                    "parallel": False,
                    "elapsed_s": time.perf_counter() - t0,
                })

        # Accessors are internal-only — strip before returning (see predict).
        data = self._strip_source_accessors(data)

        timing_df = (
            pd.DataFrame(report)
            .sort_values("elapsed_s", ascending=False)
            .reset_index(drop=True)
        )

        return data, timing_df

    # ------------------------------------------------------------------
    # Extra pip requirements (MLflow logging helper)
    # ------------------------------------------------------------------

    def add_extra_pip_requirements(self, extra_pip_requirements: list) -> None:
        self.extra_pip_requirements = extra_pip_requirements

    @property
    def get_extra_pip_requirements(self) -> list:
        return self.extra_pip_requirements

    # # ------------------------------------------------------------------
    # # Serialization
    # # ------------------------------------------------------------------

    # def save(self, path: str) -> None:
    #     import dill
    #     with open(path, "wb") as f:
    #         dill.dump(self, f)

    # @classmethod
    # def load(cls, path: str) -> "PyfuncPricingPipeline":
    #     import dill
    #     with open(path, "rb") as f:
    #         pipeline = dill.load(f)
    #     pipeline.load_context(pipeline.context)
    #     return pipeline

    def _data_to_df(self, data):
        out = {}

        for key, arr in data.items():

            if isinstance(arr, pd.DataFrame):
                out[key] = arr
                continue

            if isinstance(arr, np.ndarray):
                # already structured objects (your case)
                if arr.dtype == object:
                    out[key] = pd.DataFrame(list(arr))
                else:
                    # fallback for numeric arrays
                    out[key] = pd.DataFrame(arr)

            else:
                out[key] = pd.DataFrame(arr)

        return out


    def _data_to_array(self, data):
        """
        Convert dict[str, pd.DataFrame] → dict[str, np.ndarray(dtype=object of dicts)]
        (MLflow-friendly format)
        """
        out = {}

        for key, df in data.items():

            if isinstance(df, np.ndarray):
                out[key] = df
                continue

            if isinstance(df, pd.DataFrame):
                # DataFrame → list[dict] → np.array(dtype=object)
                out[key] = np.array(
                    df.to_dict(orient="records"),
                    dtype=object
                )
            else:
                # fallback: treat as DataFrame-like
                out[key] = np.array(list(df), dtype=object)

        return out