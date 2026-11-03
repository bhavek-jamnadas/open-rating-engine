"""The DAG scheduler: layers start as soon as their own dependencies finish.

Checks that it does not stall on the slowest layer of a wave, matches the wave
executor's output exactly (values *and* column order), orders layers that clash
on a table, and handles errors and priorities.
"""

import threading
import time

import pandas as pd
import pytest

from open_rating_engine.rating_engine.pricing_layers import (
    ColumnCalculation,
    CustomLayer,
    H2OScoringParameter,
    LayerCalculationError,
    RatingTablesLayer,
    RenameColumns,
    SourceColumn,
    TableAggregation,
    TableJoin,
)
from tests.mocks.h2o_mock import fake_load_model, make_fake_mojo_predict_pandas


def _sleepy(seconds, expr=lambda df: df.index * 0 + 1.0, log=None, name=None):
    """A ColumnCalculation callable that sleeps, optionally logging its start."""

    def _calc(df, data, catalog, options):
        if log is not None:
            log.append(name)
        time.sleep(seconds)
        return expr(df)

    return _calc


def _assert_same_output(a, b):
    assert a.keys() == b.keys()
    for table in a:
        pd.testing.assert_frame_equal(a[table], b[table])


def _run_both(pipeline, sample_data):
    pipeline.options["scheduler"] = "waves"
    waves = pipeline.predict(sample_data)
    pipeline.options["scheduler"] = "dag"
    dag = pipeline.predict(sample_data)
    return waves, dag


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #

def test_child_of_fast_layer_does_not_wait_for_slow_sibling(pipeline, sample_data, mock_context):
    pipeline.add_parameter(ColumnCalculation("slow", "policy", "s", _sleepy(0.5)))
    pipeline.add_parameter(ColumnCalculation("fast", "policy", "f", _sleepy(0.01)))
    pipeline.add_parameter(
        ColumnCalculation("child", "policy", "c", "df['f'] + 1", dependencies=["fast"])
    )

    out, timing = pipeline.profile_predict(mock_context, sample_data)

    rows = timing.set_index("layer")
    assert rows.loc["child", "start_s"] < rows.loc["slow", "end_s"]
    assert list(out["policy"]["c"]) == [2.0] * 4


def test_waves_scheduler_still_waits_for_the_whole_wave(pipeline, sample_data, mock_context):
    pipeline.options["scheduler"] = "waves"
    pipeline.add_parameter(ColumnCalculation("slow", "policy", "s", _sleepy(0.3)))
    pipeline.add_parameter(ColumnCalculation("fast", "policy", "f", _sleepy(0.01)))
    pipeline.add_parameter(
        ColumnCalculation("child", "policy", "c", "df['f'] + 1", dependencies=["fast"])
    )

    _, timing = pipeline.profile_predict(mock_context, sample_data)

    rows = timing.set_index("layer")
    assert rows.loc["child", "start_s"] >= rows.loc["slow", "end_s"]


def test_priority_starts_longest_chain_first(pipeline, sample_data):
    log = []
    pipeline.options["max_workers"] = 1
    # Registered first, so it would win a plain FIFO / canonical-order tie.
    pipeline.add_parameter(ColumnCalculation("lone", "policy", "l", _sleepy(0, log=log, name="lone")))
    pipeline.add_parameter(ColumnCalculation("head", "policy", "h", _sleepy(0, log=log, name="head")))
    pipeline.add_parameter(ColumnCalculation("mid", "policy", "m", "df['h']", dependencies=["head"]))
    pipeline.add_parameter(ColumnCalculation("tail", "policy", "t", "df['m']", dependencies=["mid"]))

    pipeline.predict(sample_data)

    assert log == ["head", "lone"]


def test_layer_costs_override_priority(pipeline, sample_data):
    log = []
    pipeline.options["max_workers"] = 1
    pipeline.options["layer_costs"] = {"lone": 100.0}
    pipeline.add_parameter(ColumnCalculation("head", "policy", "h", _sleepy(0, log=log, name="head")))
    pipeline.add_parameter(ColumnCalculation("mid", "policy", "m", "df['h']", dependencies=["head"]))
    pipeline.add_parameter(ColumnCalculation("lone", "policy", "l", _sleepy(0, log=log, name="lone")))

    pipeline.predict(sample_data)

    assert log == ["lone", "head"]


def test_max_workers_caps_concurrency(pipeline, sample_data):
    active, peak = [0], [0]
    lock = threading.Lock()

    def _calc(df, data, catalog, options):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return df.index * 0

    pipeline.options["max_workers"] = 2
    for i in range(6):
        pipeline.add_parameter(ColumnCalculation(f"c{i}", "policy", f"c{i}", _calc))

    pipeline.predict(sample_data)

    assert peak[0] == 2


# --------------------------------------------------------------------------- #
# Parity with the wave executor
# --------------------------------------------------------------------------- #

def test_parity_with_waves_on_mixed_pipeline(pipeline, sample_data, sample_rating_tables, monkeypatch):
    monkeypatch.setattr(H2OScoringParameter, "_load_model", staticmethod(fake_load_model))
    monkeypatch.setattr(
        H2OScoringParameter, "_mojo_predict_pandas",
        staticmethod(make_fake_mojo_predict_pandas("regression")),
    )
    pipeline.add_parameter(SourceColumn("policy.base_premium{}", "policy", "base_premium"))
    pipeline.add_parameter(
        ColumnCalculation("policy.base_premium", "policy", "base_premium",
                          "df['base_premium{}'] * 10", override_source=True)
    )
    # Out-of-order finishers on one table: column order must not depend on timing.
    for i, delay in enumerate([0.15, 0.0, 0.08, 0.02]):
        pipeline.add_parameter(
            ColumnCalculation(f"p{i}", "policy", f"p{i}",
                              _sleepy(delay, expr=lambda df, i=i: df["base_premium"] + i),
                              dependencies=["policy.base_premium"])
        )
    pipeline.add_parameter(
        H2OScoringParameter(
            "score", "risk", "prediction",
            column_map={"driver_youngest_age": "driver_youngest_age"},
            mojo_artifact_key="model_cost", genmodel_artifact_key="h2o_gen_model",
            model_type="regression",
        )
    )
    pipeline.add_parameter(
        RatingTablesLayer("rt", "risk", "usage_factor",
                          rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]})
    )
    pipeline.add_parameter(
        TableAggregation("agg", "risk", "risk_agg", "policy_number",
                         {"pred_sum": ("prediction", "sum")}, dependencies=["score"])
    )
    pipeline.add_parameter(
        TableJoin("join", "policy", "risk_agg", left_on="policy_number",
                  columns=["pred_sum"], dependencies=["agg"])
    )
    pipeline.add_parameter(RenameColumns("rn", "cover", {"cover_type": "ct"}))
    pipeline.add_parameter(
        ColumnCalculation("final", "policy", "final", "df['p0'] + df['pred_sum']",
                          dependencies=["p0", "join"])
    )

    for _ in range(3):
        waves, dag = _run_both(pipeline, sample_data)
        _assert_same_output(waves, dag)


def test_parity_with_waves_through_in_place_custom_layer(pipeline, sample_data):
    def _mutate(data, catalog, options, context=None):
        data["policy"]["doubled"] = data["policy"]["a"] * 2
        return data

    pipeline.add_parameter(ColumnCalculation("a", "policy", "a", _sleepy(0.05, lambda df: df["base_premium"])))
    pipeline.add_parameter(ColumnCalculation("b", "risk", "b", _sleepy(0.1)))
    pipeline.add_parameter(CustomLayer("custom", _mutate, dependencies=["a"]))
    pipeline.add_parameter(ColumnCalculation("c", "policy", "c", "df['doubled'] + 1", dependencies=["custom"]))

    waves, dag = _run_both(pipeline, sample_data)

    _assert_same_output(waves, dag)
    assert list(dag["policy"]["c"]) == [201.0, 401.0, 601.0, 801.0]


# --------------------------------------------------------------------------- #
# Table conflicts
# --------------------------------------------------------------------------- #

def test_row_changing_join_waits_for_columnar_layer_on_same_table(pipeline, sample_data, mock_context):
    # Two claim rows per policy: the join doubles the policy rows, so a slow
    # policy column committed *after* it would no longer line up.
    sample_data["claims_history"] = pd.concat(
        [sample_data["claims_history"]] * 2, ignore_index=True
    )
    pipeline.add_parameter(
        ColumnCalculation("slow", "policy", "s", _sleepy(0.3, lambda df: df["base_premium"]))
    )
    pipeline.add_parameter(
        TableJoin("join", "policy", "claims_history", left_on="policy_number", columns=["claim_cost"])
    )
    pipeline.add_parameter(ColumnCalculation("risk_layer", "risk", "r", _sleepy(0.01)))

    out, timing = pipeline.profile_predict(mock_context, sample_data)

    assert len(out["policy"]) == 8
    assert (out["policy"]["s"] == out["policy"]["base_premium"]).all()
    rows = timing.set_index("layer")
    assert rows.loc["join", "start_s"] >= rows.loc["slow", "end_s"]
    # A layer on another table is not held up by the policy join.
    assert rows.loc["risk_layer", "end_s"] < rows.loc["slow", "end_s"]


def test_effective_dependencies_only_link_clashing_tables(pipeline):
    pipeline.add_parameter(ColumnCalculation("pol", "policy", "x", "1"))
    pipeline.add_parameter(ColumnCalculation("rsk", "risk", "y", "1"))
    pipeline.add_parameter(SourceColumn("policy.base_premium", "policy", "base_premium"))
    pipeline.add_parameter(TableJoin("join", "policy", "claims_history", left_on="policy_number"))
    pipeline.add_parameter(ColumnCalculation("dup", "policy", "x", "2", dependencies=["rsk"]))

    deps = pipeline._effective_dependencies()

    assert deps["join"] == ["pol"]
    assert deps["rsk"] == []
    assert deps["policy.base_premium"] == []
    # Same (table, column) writers stay ordered, and the later one follows the join.
    assert set(deps["dup"]) == {"pol", "rsk", "join"}


def test_custom_layer_is_a_barrier(pipeline):
    pipeline.add_parameter(ColumnCalculation("before", "risk", "x", "1"))
    pipeline.add_parameter(SourceColumn("policy.base_premium", "policy", "base_premium"))
    pipeline.add_parameter(CustomLayer("custom", lambda d, c, o, ctx=None: d))
    pipeline.add_parameter(ColumnCalculation("after", "risk", "y", "1", dependencies=["before"]))

    deps = pipeline._effective_dependencies()

    assert deps["custom"] == ["before"]
    assert set(deps["after"]) == {"before", "custom"}
    assert "custom" not in deps["policy.base_premium"]


def test_effective_dependencies_reset_on_registration(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    pipeline._effective_dependencies()
    pipeline.add_parameter(ColumnCalculation("b", "policy", "y", "1"))
    assert pipeline._effective_deps is None


# --------------------------------------------------------------------------- #
# Errors and options
# --------------------------------------------------------------------------- #

def test_failure_stops_queued_layers(pipeline, sample_data):
    log = []
    pipeline.options["max_workers"] = 1
    pipeline.add_parameter(ColumnCalculation("bad", "policy", "x", "df['nope']"))
    pipeline.add_parameter(ColumnCalculation("child", "policy", "y", "1", dependencies=["bad"]))
    pipeline.add_parameter(ColumnCalculation("other", "policy", "z", _sleepy(0, log=log, name="other")))

    with pytest.raises(LayerCalculationError):
        pipeline.predict(sample_data)
    assert log == []


def test_unknown_scheduler_rejected(pipeline, sample_data):
    pipeline.options["scheduler"] = "nope"
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    with pytest.raises(ValueError, match="scheduler"):
        pipeline.predict(sample_data)


def test_pipeline_pickled_before_cache_existed_still_predicts(pipeline, sample_data):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    del pipeline._effective_deps
    out = pipeline.predict(sample_data)
    assert list(out["policy"]["x"]) == [1] * 4
