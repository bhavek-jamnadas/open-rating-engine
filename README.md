# open-rating-engine

An insurance/banking rating engine for Python. You describe a pricing algorithm as a set of
typed layers; the engine builds a dependency graph (DAG) from them and scores policies through
it. It also ships price optimisers that run on top of a pipeline.

A pipeline is an MLflow `PythonModel`, so you log, version, load and serve it with standard
MLflow tooling.

## Install

```bash
pip install open-rating-engine          # core: pipeline, layers, optimisers
pip install "open-rating-engine[all]"   # plus every optional extra
```

| Extra | Adds | Needed for |
|---|---|---|
| `h2o` | `h2o`, `jpype1` | `H2OScoringParameter` (H2O MOJO scoring) |
| `onnx` | `onnxruntime` | `OnnxScoring` |
| `viz` | `plotly`, `networkx` | `PyfuncPricingPipeline.plot_dag()` |
| `notebook` | `ipywidgets` | optimiser progress bars in Jupyter |

Optional dependencies are imported lazily, so `import open_rating_engine` works on a core-only
install.

## Imports

The headline names (the pipeline, every layer class, the layer exceptions and `apply_aging`) are
available from the top-level package. Everything else lives in its subpackage:

```python
from open_rating_engine import PyfuncPricingPipeline, RatingTablesLayer, apply_aging
from open_rating_engine.optimization import PriceOptimizerRatingFactor
from open_rating_engine.rating_engine.binning import thermometer_column_names
```

Package `__init__` modules load their names on first use, so `import open_rating_engine` is
near-instant. You pay for mlflow only when you touch the pipeline or a layer. Each module's
`__all__` defines its public API.

## Rating engine

`PyfuncPricingPipeline` (`open_rating_engine.rating_engine.pricing_pipeline`):

- `add_parameter()` / `remove_parameter()` register and remove layers.
- Layers are topologically sorted, and cycles are reported. Independent column calculations run
  in parallel waves. Structural layers (joins, drops, renames, aggregations) run in sequence so
  that tables stay consistent.
- `predict(context, model_input)` is the MLflow-compatible scoring entry point.
- `load_context(context)` expects every artifact in `context.artifacts` to be a local file
  path. The engine fetches nothing remote, so download artifacts from blob storage or a model
  registry before building the context.
- `profile_predict()` returns how long each layer took. `plot_dag()` draws the graph.

Pipeline state is a dict of named DataFrames. Each layer declares what it reads (`depends_on`)
and what it writes, and the pipeline builds the DAG from those declarations.

Layer types (`open_rating_engine.rating_engine.pricing_layers`), all extending `BasePricingLayer`:

| Layer | Purpose |
|---|---|
| `SourceColumn` | Declares an input column supplied in the scoring payload |
| `ColumnCalculation` | Computes a new column from an expression string (restricted builtins) or a callable |
| `TableTransform` | Creates or replaces a table with a callable |
| `TableAggregation` | Group-by with named aggregations, e.g. `{"total": ("cost", "sum")}` |
| `TableJoin` | Merges two tables; the key names can differ on each side |
| `RenameColumns` / `DropColumns` | Column-level changes |
| `TableDrop` | Removes an intermediate table from the pipeline state |
| `RatingTablesLayer` | Looks up rating factors from several tables and multiplies them together |
| `GlmScoring` | Scores a GLM or other additive model in closed form, with no JVM |
| `H2OScoringParameter` | Scores an H2O MOJO model (regression / binomial / multinomial) |
| `OnnxScoring` | Scores an ONNX model |
| `CustomLayer` | Runs any callable that doesn't fit a typed layer |

`open_rating_engine.rating_engine.binning` holds the thermometer (cumulative) encoding. Training
and scoring both use it, so they produce identical columns.

## Price optimisation

`open_rating_engine.optimization`:

| Optimiser | Decision |
|---|---|
| `PriceOptimizerIndividual` | A multiplier for each policy, for a single period |
| `PriceOptimizerRatingFactor` | A multiplier for each rating-factor level, for a single period |
| `PriceOptimizerIndividualMultiPeriod` | A lifetime multiplier for each policy, chosen against multi-period NPV |
| `PriceOptimizerRatingFactorMultiPeriod` | A rating-factor overlay, chosen by forward-simulating the horizon |

Helpers: `simulate_adjustment_grid`, `make_demand_emulation`, `make_log_multiplier_penalty` and
`make_log_multiplier_regularizer`.

Multi-period optimisers age the book each period with
`open_rating_engine.data_processing.data_aging.apply_aging`.

## Development

```bash
pip install -e ".[all,test]"
pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
