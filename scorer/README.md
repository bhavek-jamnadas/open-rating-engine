# ors-scorer

A Rust scorer for open-rating-engine pipelines. A pipeline is exported once, at
deploy time, as a **plan bundle**; one generic `ors-scorer` binary scores any
bundle with no Python, pandas or JVM.

```
bundle/
  plan.json          # written by open_rating_engine.scoring_plan.write_plan_bundle
  models/<key>.onnx  # the ONNX models the plan scores with, checked by sha256
```

## Export a pipeline

```python
from open_rating_engine.scoring_plan import export_plan, write_plan_bundle

result = export_plan(pipeline)          # never raises; lists every blocker
for blocker in result.blockers:
    print(blocker)                      # e.g. "risk.conversion_rate (H2OScoringParameter): ..."

write_plan_bundle(pipeline, "bundle")   # raises PlanExportError if anything blocks
```

Supported: `ColumnCalculation` expression strings, `RatingTablesLayer`,
`OnnxScoring`, `GlmScoring`, `TableJoin` (left joins, from data or the catalog),
`TableAggregation`, `RenameColumns`, `DropColumns`, `TableDrop` and `SourceColumn`
(including `<column>{}` accessors and `override_source`).

Expressions may use `df['col']`, arithmetic (`+ - * / **`), comparisons, `& | ~`,
`np.where/maximum/minimum/abs/log/exp/sqrt/floor/ceil/round/clip/isnan`,
`pd.cut` (with `labels=[...]` or `labels=False`), `pd.isna/notna`,
`.fillna/.isna/.notna/.astype(int|float)/.abs/.round/.clip`, and `aggregate()`.
Any part of an expression that reads no rows (catalog lookups, `options`,
`list(range(...)) + [np.inf]`) is evaluated at export and stored as a constant.

Blockers: H2O models (convert to ONNX or `GlmScoring`), `ColumnCalculation` with a
Python function, `CustomLayer`, `TableTransform`, non-left joins, a `pydantic_model`,
`options["emit_breakdown"]`, and anything outside the expression vocabulary.

## Build and run

```sh
docker build -t ors-scorer:0.1.0 .                     # base image, ~150MB
docker run -p 5001:5001 -v $PWD/bundle:/app/model ors-scorer:0.1.0
```

`GET /ping` and `POST /invocations` match the Python serving container: the body is
`{"inputs": {table: [records]}}` and the reply `{"predictions": {table: [records]}}`.
Each request logs one JSON line with `latency_ms`, `status`, `runtime`,
`scorer_version` and `plan_hash`.

`ors-scorer score --bundle DIR < request.json` scores one request without a server.

The scorer **refuses to start** if the bundle holds `input_validation.json`:
input validation is not implemented here, and serving unvalidated quotes silently
would be worse than not serving.

## Matching pandas

The scorer reproduces pandas and numpy 2 behaviour, not just the arithmetic:
dtype promotion (float32 model outputs stay float32 against Python constants),
NaN and `None` handling, `pd.cut` bin edges, pandas' Kahan summation in group
sums, merge key matching and the column order of every table.

ONNX Runtime runs on **one thread** by default (`ORS_ONNX_THREADS`). Tree ensembles
sum in an order that depends on the thread count, so on large batches a model's
float32 outputs differ between machines with different core counts. The Python
`OnnxScoring` layer uses ONNX Runtime's default threads and has this problem today.

Two checks keep the runtimes in step:

- `cargo test` scores the bundles in `tests/fixtures` and requires the pandas
  output in each `expected.json` exactly. Regenerate them with
  `python scorer/tests/make_fixtures.py` after changing the exporter.
- `open_rating_engine.scoring_plan.check_parity(pipeline, bundle, samples, command)`
  runs a pipeline and the scorer on the same inputs and compares every value. Run
  it on a snapshot's samples before deploying it with the Rust runtime. The Python
  tests run it when `ORS_SCORER` names a scorer command, for example
  `ORS_SCORER="docker run --rm -i -v {bundle}:/bundle ors-scorer:dev"`.

On the example motor pipeline without its H2O model (52 layers), the full
origination book (91,366 output rows, 2.4 million values) matches pandas exactly.
