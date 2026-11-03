# Standard library imports
import heapq
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local application/library-specific imports
from typing import List, Dict, Callable, Optional, Union
from mlflow.pyfunc import PythonModel, PythonModelContext

# NOTE: no `import h2o` here. It was unused in this module — the H2O work all happens in
# `pricing_layers.H2OScoringParameter`, and MOJO scoring itself goes through JPype, not the
# H2O Python client. It cost every consumer a heavyweight import for nothing, on the cold
# start of every scale-to-zero scoring container.

from .pricing_layers import (
    BasePricingLayer,
    ColumnCalculation,
    CustomLayer,
    DependencyError,
    DropColumns,
    H2OScoringParameter,
    LayerCalculationError,
    RatingTablesLayer,
    RenameColumns,
    SourceColumn,
    TableAggregation,
    TableDrop,
    TableJoin,
    TableTransform,
)

__all__ = [
    "PyfuncPricingPipeline",
]

# Layers that restructure ``data`` — replace, rename or drop whole tables. They run on
# the coordinating thread. Every other layer is columnar: a pure ``_compute_result``
# whose Series the pipeline writes onto ``target_table``.
_STRUCTURAL_TYPES = (TableTransform, TableJoin, TableDrop, RenameColumns, DropColumns, CustomLayer, SourceColumn)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PyfuncPricingPipeline(PythonModel):
    """
    A dependency-aware pipeline that applies a sequence of
    :class:`BasePricingLayer` objects to a dataset of named DataFrames.

    By default (``options["scheduler"] = "dag"``) each layer starts as soon as
    its own dependencies have finished: columnar layers (:class:`ColumnCalculation`
    and the model-scoring subclasses) run on a thread pool, structural layers
    (:class:`TableAggregation`, :class:`TableJoin`, ...) run on the calling
    thread. ``options["scheduler"] = "waves"`` restores the older executor,
    which runs whole dependency levels ("waves") one after another.

    Other options read by the executor: ``max_workers`` (thread-pool size,
    default: number of columnar layers capped at 32) and ``layer_costs``
    (``{layer_name: seconds}`` used to start the longest chain first).
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
        self._effective_deps: Optional[Dict[str, List[str]]] = None

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
        self._effective_deps = None
        return self

    def remove_parameter(self, layer_name: str) -> "PyfuncPricingPipeline":
        """Remove a registered layer by name, invalidating any cached execution order."""
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' is not registered in the pipeline.")
        del self.layers[layer_name]
        self._execution_order = None
        self._effective_deps = None
        return self

    def load_context(self, context: PythonModelContext) -> None:
        """Bind ``context`` to the pipeline and initialise layers that need artifacts.

        Every value in ``context.artifacts`` must be a local file path. The engine does
        not fetch remote storage: a caller holding artifacts elsewhere (blob storage, S3,
        a model registry) downloads them and passes the local paths in the context.
        """
        import sys

        for name, path in context.artifacts.items():
            if isinstance(path, str):
                context.artifacts[name] = path.replace("\\", "/")
            print(f'Load Context: Adding Artifact - {name}', file=sys.stderr)

        self.context = context

        # Only the H2O MOJO scorer needs a JVM. Pipelines without H2O layers
        # (e.g. pure rating-table / column-calculation algorithms) must score
        # without one — and without requiring an 'h2o_gen_model' artifact, or
        # jpype (the h2o extra) being installed.
        h2o_layers = [l for l in self.layers.values() if isinstance(l, H2OScoringParameter)]

        if h2o_layers:
            import jpype
            import jpype.imports

            if not jpype.isJVMStarted():
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
            else:
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
        on_layer_done: Optional[Callable[[str, float, float, bool], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run ColumnCalculation layers in a wave in parallel."""
        snapshot = {k: v.copy() for k, v in data.items()}
        results: Dict[str, tuple] = {}
        errors: Dict[str, Exception] = {}

        def _compute(name: str) -> tuple:
            start = time.perf_counter()
            series = self.layers[name]._compute_result(snapshot, self.catalog, self.options, self.context)
            return series, start, time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {executor.submit(_compute, name): name for name in wave}
            for future in as_completed(futures):
                name = futures[future]
                exc = future.exception()
                if exc:
                    errors[name] = exc
                else:
                    results[name] = future.result()

        if errors:
            raise next(iter(errors.values()))

        for name in wave:
            layer = self.layers[name]
            series, start, end = results[name]
            data[layer.target_table] = data[layer.target_table].copy()
            data[layer.target_table][layer.output_column] = series.values
            if on_layer_done is not None:
                on_layer_done(name, start, end, True)

        return data

    def _run_wave_parallel(
        self,
        wave: List[str],
        data: Dict[str, pd.DataFrame],
        on_layer_done: Optional[Callable[[str, float, float, bool], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a wave of layers. ColumnCalculation layers run in parallel;
        TableAggregation and TableJoin layers run sequentially (they mutate
        the data dict structurally).
        """
        columnar = [n for n in wave if not self._is_structural(self.layers[n])]
        structural = [n for n in wave if self._is_structural(self.layers[n])]

        if columnar:
            data = self._run_columnar_wave(columnar, data, on_layer_done)

        for name in structural:
            start = time.perf_counter()
            data = self.layers[name].apply(data, self.catalog, self.options, self.context)
            if on_layer_done is not None:
                on_layer_done(name, start, time.perf_counter(), False)

        return data

    def _run_waves(
        self,
        data: Dict[str, pd.DataFrame],
        on_layer_done: Optional[Callable[[str, float, float, bool], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run the pipeline wave by wave (``options["scheduler"] = "waves"``).

        Each wave starts only once the previous one has fully finished.
        """
        waves = self._build_waves()
        wave_iterator = (
            tqdm(waves, total=len(waves), desc="Calculating Prices")
            if self.options.get("show_progress")
            else waves
        )

        for wave in wave_iterator:
            try:
                if len(wave) == 1:
                    start = time.perf_counter()
                    data = self.layers[wave[0]].apply(data, self.catalog, self.options, self.context)
                    if on_layer_done is not None:
                        on_layer_done(wave[0], start, time.perf_counter(), False)
                else:
                    data = self._run_wave_parallel(wave, data, on_layer_done)
            except (LayerCalculationError, RuntimeError):
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected error in wave {wave}: {exc}"
                ) from exc

        return data

    # ------------------------------------------------------------------
    # DAG scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _is_structural(layer: BasePricingLayer) -> bool:
        return isinstance(layer, _STRUCTURAL_TYPES)

    def _canonical_order(self) -> List[str]:
        """Return the order the wave executor runs layers in: wave by wave, with
        columnar layers before structural ones inside a wave.

        The DAG scheduler orients its conflict edges by this order, so layers that
        touch the same table keep the relative order they have under ``"waves"``.
        """
        order: List[str] = []
        for wave in self._build_waves():
            order += [n for n in wave if not self._is_structural(self.layers[n])]
            order += [n for n in wave if self._is_structural(self.layers[n])]
        return order

    @staticmethod
    def _layer_footprint(layer: BasePricingLayer) -> Optional[set]:
        """Tables a layer reads or writes as a whole. ``None`` means any table."""
        if isinstance(layer, SourceColumn):
            return set()
        if isinstance(layer, TableTransform):
            return {layer.source_table, layer.output_table}
        if isinstance(layer, TableJoin):
            tables = {layer.left_table, layer.output_table}
            if layer.right_source == "data":
                tables.add(layer.right_table)
            return tables
        if isinstance(layer, (RenameColumns, DropColumns)):
            return {layer.target_table}
        if isinstance(layer, TableDrop):
            return set(layer.tables)
        if isinstance(layer, ColumnCalculation):
            return {layer.target_table}
        return None

    def _effective_dependencies(self) -> Dict[str, List[str]]:
        """Declared dependencies plus the conflict edges the DAG scheduler needs.

        Two layers that declared dependencies leave unordered can still clash on a
        table. For example, a join could rewrite ``policy`` while a model is still
        scoring onto it, and the model's positional values would then land on the
        wrong rows. Each of these pairs is ordered the way the wave executor orders
        it:

        * a structural layer and any layer whose footprint shares a table with it;
        * a layer with an unknown footprint (``CustomLayer``) and every other layer
          except the no-op ``SourceColumn`` layers;
        * two columnar layers that write the same ``(table, column)``.

        Columnar layers on the same table stay independent, because adding columns
        commutes. Every edge points forward in :meth:`_canonical_order`, so no cycle
        can appear.
        """
        # getattr: pipelines pickled before this cache existed lack the attribute.
        if getattr(self, "_effective_deps", None) is not None:
            return self._effective_deps

        order = self._canonical_order()
        rank = {name: i for i, name in enumerate(order)}
        footprints = {name: self._layer_footprint(self.layers[name]) for name in order}
        deps = {name: set(self.layers[name].dependencies) for name in order}

        def link(a: str, b: str) -> None:
            if a != b:
                first, second = sorted((a, b), key=rank.__getitem__)
                deps[second].add(first)

        by_table: Dict[str, List[str]] = defaultdict(list)
        for name in order:
            for table in footprints[name] or ():
                by_table[table].append(name)

        writers: Dict[tuple, List[str]] = defaultdict(list)
        for name in order:
            layer = self.layers[name]
            footprint = footprints[name]
            if footprint is None:
                for other in order:
                    if footprints[other] != set():
                        link(name, other)
            elif self._is_structural(layer):
                for table in footprint:
                    for other in by_table[table]:
                        link(name, other)
            else:
                writers[(layer.target_table, layer.output_column)].append(name)

        for names in writers.values():
            for a, b in zip(names, names[1:]):
                link(a, b)

        self._effective_deps = {name: sorted(deps[name], key=rank.__getitem__) for name in order}
        return self._effective_deps

    def _critical_path_priority(
        self, order: List[str], children: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Longest cost-weighted path from each layer to the end of the pipeline.

        Each layer costs 1 by default. ``options["layer_costs"]``, for example the
        ``elapsed_s`` column of :meth:`profile_predict`, overrides that.
        """
        costs = self.options.get("layer_costs") or {}
        priority: Dict[str, float] = {}
        for name in reversed(order):
            downstream = max((priority[c] for c in children[name]), default=0.0)
            priority[name] = float(costs.get(name, 1.0)) + downstream
        return priority

    @staticmethod
    def _commit_column(
        data: Dict[str, pd.DataFrame],
        layer: ColumnCalculation,
        series: pd.Series,
        rank: int,
        appended: Dict[str, Dict[str, int]],
    ) -> None:
        """Write a columnar result onto its table without mutating the current frame,
        which in-flight layers may still be reading.

        A new column goes among the columns this run has appended to the table, by
        canonical rank, so column order never depends on which layer finished first.
        """
        table, column = layer.target_table, layer.output_column
        frame = data[table].copy()
        if column in frame.columns:
            frame[column] = series.values
        else:
            later = [i for i, c in enumerate(frame.columns) if appended[table].get(c, -1) > rank]
            if later:
                frame.insert(later[0], column, series.values)
            else:
                frame[column] = series.values
            appended[table][column] = rank
        data[table] = frame

    @staticmethod
    def _raise_layer_error(name: str, exc: BaseException) -> None:
        if isinstance(exc, (LayerCalculationError, RuntimeError)):
            raise exc
        raise RuntimeError(f"Unexpected error in layer '{name}': {exc}") from exc

    def _run_dag(
        self,
        data: Dict[str, pd.DataFrame],
        on_layer_done: Optional[Callable[[str, float, float, bool], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run each layer as soon as its effective dependencies have finished.

        Columnar layers score on a thread pool, each against a shallow copy of
        ``data`` taken when it is submitted. That copy is safe to share because
        every write replaces a frame instead of mutating it. Structural layers and
        all writes to ``data`` run on the calling thread, so nothing needs a lock.
        When more layers are ready than there are workers, the layer at the head of
        the longest remaining chain starts first.

        ``on_layer_done(name, start, end, on_worker)`` is called with
        ``time.perf_counter`` timestamps as each layer finishes.
        """
        order = self._canonical_order()
        if not order:
            return data
        rank = {name: i for i, name in enumerate(order)}
        deps = self._effective_dependencies()
        children: Dict[str, List[str]] = defaultdict(list)
        for name in order:
            for dep in deps[name]:
                children[dep].append(name)
        priority = self._critical_path_priority(order, children)
        remaining = {name: len(deps[name]) for name in order}
        ready = [(-priority[n], rank[n], n) for n in order if remaining[n] == 0]
        heapq.heapify(ready)

        n_columnar = sum(not self._is_structural(self.layers[n]) for n in order)
        max_workers = self.options.get("max_workers") or min(32, max(1, n_columnar))
        progress = (
            tqdm(total=len(order), desc="Calculating Prices")
            if self.options.get("show_progress")
            else None
        )
        appended: Dict[str, Dict[str, int]] = defaultdict(dict)
        in_flight: Dict = {}

        def compute(name: str, snapshot: Dict[str, pd.DataFrame]) -> tuple:
            start = time.perf_counter()
            series = self.layers[name]._compute_result(snapshot, self.catalog, self.options, self.context)
            return series, start, time.perf_counter()

        def finish(name: str, start: float, end: float, on_worker: bool) -> None:
            if on_layer_done is not None:
                on_layer_done(name, start, end, on_worker)
            if progress is not None:
                progress.update(1)
            for child in children[name]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    heapq.heappush(ready, (-priority[child], rank[child], child))

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            while ready or in_flight:
                # Columnar layers that find the pool full wait here, so the
                # priority order holds instead of the executor's FIFO queue.
                waiting = []
                while ready:
                    item = heapq.heappop(ready)
                    name = item[2]
                    layer = self.layers[name]
                    if self._is_structural(layer):
                        start = time.perf_counter()
                        try:
                            data = layer.apply(data, self.catalog, self.options, self.context)
                        except Exception as exc:
                            self._raise_layer_error(name, exc)
                        finish(name, start, time.perf_counter(), False)
                    elif len(in_flight) < max_workers:
                        in_flight[executor.submit(compute, name, dict(data))] = name
                    else:
                        waiting.append(item)
                for item in waiting:
                    heapq.heappush(ready, item)

                if not in_flight:
                    continue
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: rank[in_flight[f]]):
                    name = in_flight.pop(future)
                    exc = future.exception()
                    if exc is not None:
                        self._raise_layer_error(name, exc)
                    series, start, end = future.result()
                    self._commit_column(data, self.layers[name], series, rank[name], appended)
                    finish(name, start, end, True)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if progress is not None:
                progress.close()

        return data

    def _run_layers(
        self,
        data: Dict[str, pd.DataFrame],
        on_layer_done: Optional[Callable[[str, float, float, bool], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        scheduler = self.options.get("scheduler", "dag")
        if scheduler == "dag":
            return self._run_dag(data, on_layer_done)
        if scheduler == "waves":
            return self._run_waves(data, on_layer_done)
        raise ValueError(f"options['scheduler'] must be 'dag' or 'waves', got {scheduler!r}")

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
        """Print a human-readable execution plan grouped by dependency level (wave).

        Under the default ``"dag"`` scheduler a wave is the earliest point at which
        its layers can start, not a barrier: each layer starts once its own
        dependencies finish.
        """
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

        data = self._run_layers(data)

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

        Layers still run concurrently under the configured scheduler. Each layer
        records its own start and end, so individual costs are visible even when
        layers overlap on the wall clock. ``start_s``/``end_s`` are seconds from
        the start of the run and give the overlap timeline.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            Output data dict and a DataFrame with columns (wave, layer, type,
            parallel, elapsed_s, start_s, end_s), sorted slowest first. ``wave`` is
            the layer's dependency level. ``parallel`` is True when the layer ran on
            a worker thread.
        """
        if self.pydantic_model is not None:
            try:
                self.pydantic_model.model_validate(model_input)
            except Exception as exc:
                raise ValueError(f"Schema validation failed: {exc}") from exc

        data: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in model_input.items()}
        self._snapshot_source_accessors(data)
        level = {name: i for i, wave in enumerate(self._build_waves(), 1) for name in wave}
        report = []
        t0 = time.perf_counter()

        def _record(name: str, start: float, end: float, on_worker: bool) -> None:
            report.append({
                "wave": level[name],
                "layer": name,
                "type": type(self.layers[name]).__name__,
                "parallel": on_worker,
                "elapsed_s": end - start,
                "start_s": start - t0,
                "end_s": end - t0,
            })

        data = self._run_layers(data, on_layer_done=_record)

        # Accessors are internal-only — strip before returning (see predict).
        data = self._strip_source_accessors(data)

        timing_df = (
            pd.DataFrame(report, columns=["wave", "layer", "type", "parallel", "elapsed_s", "start_s", "end_s"])
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