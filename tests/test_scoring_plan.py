"""Scoring-plan export, and parity with the Rust scorer when one is available.

The parity tests run only when ``ORS_SCORER`` names a scorer command, for example::

    ORS_SCORER="docker run --rm -i -v {bundle}:/bundle ors-scorer:dev" pytest tests/test_scoring_plan.py
    ORS_SCORER=scorer/target/release/ors-scorer pytest tests/test_scoring_plan.py
"""

import json
import os
import re
import shlex

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.rating_engine import (
    ColumnCalculation,
    CustomLayer,
    H2OScoringParameter,
    PyfuncPricingPipeline,
    TableJoin,
    TableTransform,
)
from open_rating_engine.scoring_plan import (
    PlanExportError,
    check_parity,
    export_plan,
    write_plan_bundle,
)
from open_rating_engine.scoring_plan.expressions import (
    ExpressionScope,
    UnsupportedExpression,
    compile_expression,
)
from open_rating_engine.scoring_plan.parity import compare_records
from tests.scoring_plan_cases import kitchen_sink_inputs, kitchen_sink_pipeline, write_tiny_onnx

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def tiny_model(tmp_path):
    return write_tiny_onnx(tmp_path / "tiny.onnx")


@pytest.fixture
def kitchen_sink(tiny_model):
    return kitchen_sink_pipeline(tiny_model)


def _column_scope(**kwargs):
    return ExpressionScope("column", target_table="risk", **kwargs)


# -- expressions -------------------------------------------------------------


def test_catalog_lookup_is_folded_to_a_constant():
    catalog = {"expenses": pd.DataFrame({"expense_type": ["fixed"], "value": [75.0]})}
    node = compile_expression(
        "catalog['expenses'].set_index('expense_type').loc['fixed', 'value'] + df['x']",
        _column_scope(catalog=catalog),
    )
    assert node["l"] == {"k": "const", "t": "float", "v": 75.0}
    assert node["r"] == {"k": "col", "name": "x"}


def test_cut_bins_are_evaluated_at_export():
    node = compile_expression(
        "pd.cut(df['age'], bins=list(range(15, 36, 10)) + [np.inf], labels=['a', 'b', 'c'])",
        _column_scope(),
    )
    assert node["edges"] == [15.0, 25.0, 35.0, "inf"]
    assert node["labels"] == ["a", "b", "c"]


def test_aggregate_keys_come_from_the_data_structure():
    structure = [
        {"name": "policy", "parent": None},
        {"name": "risk", "parent": "policy", "parent_key": "policy.id", "child_key": "risk.policy_id"},
    ]
    node = compile_expression(
        "aggregate(source='policy', source_column='policy.x', agg_func='first')",
        _column_scope(),
        structure,
    )
    assert (node["target_key"], node["source_key"]) == ("risk.policy_id", "policy.id")


@pytest.mark.parametrize(
    "source, reason",
    [
        ("df['x'] // 2", "FloorDiv"),
        ("df['name'].str.upper()", "method .upper()"),
        ("data['policy']['x']", "data[...]"),
        ("pd.cut(df['x'], bins=[0, 1, 2])", "labels"),
        ("df['x'] if True else df['y']", "IfExp"),
        ("np.random.rand() * df['x']", "np.random"),
        ("df['x'].map({'a': 1})", ".map()"),
        ("aggregate(source='risk', agg_func='mode', source_column='x')", "mode"),
    ],
)
def test_unsupported_expressions_are_named(source, reason):
    with pytest.raises(UnsupportedExpression, match=re.escape(reason)):
        compile_expression(source, _column_scope())


# -- export ------------------------------------------------------------------


def test_kitchen_sink_exports(kitchen_sink):
    result = export_plan(kitchen_sink)
    assert result.blockers == []
    ops = {step["op"] for step in result.plan["steps"]}
    assert ops == {"column", "rating_tables", "onnx", "glm", "join", "aggregate_table", "rename",
                   "drop_columns", "drop_tables"}
    assert ["risk", "value"] in result.plan["source_accessors"]
    # Steps follow the wave executor's order.
    exported = [step["layer"] for step in result.plan["steps"]]
    canonical = [n for n in kitchen_sink._canonical_order() if n in set(exported)]
    assert exported == canonical


def test_onnx_formulas_are_exported_only_where_used(kitchen_sink):
    steps = {step["layer"]: step for step in export_plan(kitchen_sink).plan["steps"]}
    plain, derived = steps["model_score"], steps["model_score_derived"]
    assert "derived" not in plain
    assert [d["name"] for d in derived["derived"]] == ["age", "risk_id"]
    assert derived["inputs"] == [["age_raw", "age"], ["value", "value"], ["usage", "usage"],
                                 ["rid", "risk_number"]]


def test_every_blocker_is_reported(pipeline):
    pipeline.add_parameter(ColumnCalculation("f", "risk", "f", lambda df, *_: df["a"]))
    pipeline.add_parameter(CustomLayer("c", func=lambda data, *_: data))
    pipeline.add_parameter(TableTransform("t", "risk", "risk2", calculation=lambda df, *_: df))
    pipeline.add_parameter(TableJoin("j", "risk", "policy", left_on="k", how="inner"))
    pipeline.add_parameter(ColumnCalculation("s", "risk", "s", "df['a'].str.len()"))
    pipeline.add_parameter(H2OScoringParameter(
        "h", "risk", "h", column_map={"a": "a"}, mojo_artifact_key="m", genmodel_artifact_key="g",
    ))
    result = export_plan(pipeline)
    assert not result.exportable
    reasons = {b.layer: b.reason for b in result.blockers}
    assert set(reasons) == {"f", "c", "t", "j", "s", "h"}
    assert "Python function" in reasons["f"]
    assert "JVM" in reasons["h"]


def test_write_plan_bundle_refuses_a_blocked_pipeline(pipeline, tmp_path):
    pipeline.add_parameter(ColumnCalculation("f", "risk", "f", lambda df, *_: df["a"]))
    with pytest.raises(PlanExportError, match="Python function"):
        write_plan_bundle(pipeline, tmp_path)
    assert not (tmp_path / "plan.json").exists()


def test_bundle_holds_the_plan_and_a_checksummed_model(kitchen_sink, tmp_path):
    plan_path = write_plan_bundle(kitchen_sink, tmp_path / "bundle")
    plan = json.loads(plan_path.read_text())
    model = plan["models"]["tiny_model"]
    assert (tmp_path / "bundle" / model["path"]).is_file()
    assert len(model["sha256"]) == 64
    # Non-finite floats are encoded as strings, so the plan is strict JSON.
    json.loads(plan_path.read_text(), parse_constant=lambda c: pytest.fail(f"bare {c} in plan.json"))


def test_missing_onnx_artifact_is_a_blocker(kitchen_sink):
    result = export_plan(kitchen_sink, artifacts={})
    assert [b.layer for b in result.blockers] == ["model_score", "model_score_derived"]


# -- parity ------------------------------------------------------------------


def test_compare_records_reports_each_difference():
    expected = {"t": [{"a": 1.0, "b": "x"}, {"a": None, "b": "y"}]}
    actual = {"t": [{"a": 1.0 + 1e-6, "b": "x"}, {"a": None, "b": "z"}]}
    report = compare_records(expected, actual)
    assert not report.ok
    assert [(m.column, m.row) for m in report.mismatches] == [("a", 0), ("b", 1)]
    assert compare_records(expected, expected).ok


SCORER = os.environ.get("ORS_SCORER")
needs_scorer = pytest.mark.skipif(not SCORER, reason="set ORS_SCORER to run the Rust scorer")


@needs_scorer
def test_kitchen_sink_matches_pandas_exactly(kitchen_sink, tmp_path):
    bundle = tmp_path / "bundle"
    write_plan_bundle(kitchen_sink, bundle)
    report = check_parity(kitchen_sink, bundle, kitchen_sink_inputs(), shlex.split(SCORER), rtol=0, atol=0)
    assert report.ok, report.summary()
    assert report.values_checked > 300


@needs_scorer
def test_layer_errors_surface_from_the_scorer(tmp_path, mock_context):
    from open_rating_engine.scoring_plan.parity import run_rust_scorer

    p = PyfuncPricingPipeline(mock_context)
    p.add_parameter(ColumnCalculation("bad", "risk", "bad", "df['missing'] * 2"))
    write_plan_bundle(p, tmp_path)
    with pytest.raises(RuntimeError, match="layer 'bad'.*missing"):
        run_rust_scorer(shlex.split(SCORER), tmp_path, {"risk": [{"a": 1}]})
