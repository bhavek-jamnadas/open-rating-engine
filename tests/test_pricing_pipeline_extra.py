"""Extra pipeline coverage: load_context, non-local I/O, profiling, errors."""

import numpy as np
import pandas as pd
import pytest
from mlflow.pyfunc import PythonModelContext

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import (
    ColumnCalculation,
    CustomLayer,
    DropColumns,
    H2OScoringParameter,
    RatingTablesLayer,
    RenameColumns,
    TableAggregation,
    TableDrop,
    TableJoin,
)
from tests.mocks.h2o_mock import fake_load_model


def test_load_context_normalises_paths_and_inits_h2o(monkeypatch):
    import jpype

    monkeypatch.setattr(jpype, "isJVMStarted", lambda: True, raising=False)
    monkeypatch.setattr(H2OScoringParameter, "_load_model",
                         staticmethod(fake_load_model))

    ctx = PythonModelContext(
        artifacts={"h2o_gen_model": "C:\\models\\genmodel.jar",
                   "model_cost": "C:\\models\\m.zip"},
        model_config={"environment": "local"},
    )
    pipe = PyfuncPricingPipeline(ctx)
    pipe.options["show_progress"] = False
    layer = H2OScoringParameter(
        "s", "risk", "p", column_map={"a": "a"},
        mojo_artifact_key="model_cost", genmodel_artifact_key="h2o_gen_model",
        model_type="regression",
    )
    pipe.add_parameter(layer)
    pipe.load_context(ctx)

    assert ctx.artifacts["h2o_gen_model"] == "C:/models/genmodel.jar"
    assert layer.wrapper is not None  # init_from_context ran


def test_load_context_does_not_fetch_remote_artifacts(monkeypatch):
    """Artifacts are taken as given: a URL is the caller's to resolve, not the engine's."""
    import jpype

    monkeypatch.setattr(jpype, "isJVMStarted", lambda: False, raising=False)

    url = "https://acct.blob.core.windows.net/c/m.zip"
    ctx = PythonModelContext(
        artifacts={"model": url, "not_a_path": 42},
        model_config={"environment": "local"},
    )
    pipe = PyfuncPricingPipeline(ctx)
    pipe.load_context(ctx)

    assert ctx.artifacts == {"model": url, "not_a_path": 42}


def test_non_local_environment_roundtrips_arrays(sample_data):
    ctx = PythonModelContext(
        artifacts={}, model_config={"environment": "cloud"}
    )
    pipe = PyfuncPricingPipeline(ctx)
    pipe.options["show_progress"] = False
    pipe.add_parameter(
        ColumnCalculation("c", "policy", "x", "df['base_premium'] * 2")
    )
    out = pipe.predict(sample_data)
    # Non-local path returns dict[str, np.ndarray(dtype=object of dicts)]
    assert isinstance(out["policy"], np.ndarray)
    assert out["policy"][0]["x"] == 200.0


def test_data_to_df_accepts_array_input():
    ctx = PythonModelContext(artifacts={}, model_config={"environment": "cloud"})
    pipe = PyfuncPricingPipeline(ctx)
    arr = np.array([{"a": 1}, {"a": 2}], dtype=object)
    converted = pipe._data_to_df({"t": arr})
    pd.testing.assert_frame_equal(
        converted["t"].reset_index(drop=True), pd.DataFrame({"a": [1, 2]})
    )


def test_profile_predict_records_structural_layer(pipeline, sample_data,
                                                   mock_context):
    pipeline.add_parameter(
        TableAggregation("agg", "claims_history", "ca", "policy_number",
                         {"tot": ("claim_cost", "sum")})
    )
    pipeline.add_parameter(
        TableJoin("j", "policy", "ca", left_on="policy_number",
                  columns=["tot"], dependencies=["agg"])
    )
    _, timing = pipeline.profile_predict(mock_context, sample_data)
    join_row = timing[timing["layer"] == "j"].iloc[0]
    assert join_row["type"] == "TableJoin"
    assert bool(join_row["parallel"]) is False


def test_predict_wraps_unexpected_errors_as_runtime(pipeline, sample_data):
    # TableJoin on a missing table raises KeyError -> wrapped RuntimeError.
    pipeline.add_parameter(
        TableJoin("j", "policy", "nonexistent_table", left_on="policy_number")
    )
    with pytest.raises(RuntimeError, match="Unexpected error in wave"):
        pipeline.predict(sample_data)


def test_print_execution_plan_covers_all_layer_types(
    pipeline, capsys, sample_rating_tables
):
    pipeline.add_parameter(ColumnCalculation("col", "risk", "c1", "1"))
    pipeline.add_parameter(
        RatingTablesLayer("rt", "risk", "rf",
                          rating_tables={"risk_usage_mult":
                                         sample_rating_tables["risk_usage_mult"]})
    )
    pipeline.add_parameter(
        TableAggregation("agg", "claims_history", "ca", "policy_number",
                         {"t": ("claim_cost", "sum")})
    )
    pipeline.add_parameter(
        TableJoin("j", "policy", "ca", left_on="policy_number",
                  columns=["t"], dependencies=["agg"])
    )
    pipeline.add_parameter(RenameColumns("rn", "cover", {"cover_type": "ct"}))
    pipeline.add_parameter(DropColumns("dc", "risk", ["risk_usage"],
                                       dependencies=["col"]))
    pipeline.add_parameter(TableDrop("td", "ca", dependencies=["j"]))
    pipeline.add_parameter(
        CustomLayer("cu", lambda d, c, o, ctx=None: d)
    )
    pipeline.print_execution_plan()
    out = capsys.readouterr().out
    for tag in ["[COL]", "[RATING]", "[AGG]", "[JOIN]",
                "[RENAME]", "[DROP COLS]", "[DROP]", "[CUSTOM]"]:
        assert tag in out
