"""Tests for PyfuncPricingPipeline: registration, DAG resolution, predict."""

import warnings

import pandas as pd
import pytest
from mlflow.pyfunc import PythonModelContext

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import (
    ColumnCalculation,
    CustomLayer,
    DependencyError,
    SourceColumn,
    TableAggregation,
    TableJoin,
)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_requires_python_model_context():
    with pytest.raises(TypeError, match="PythonModelContext"):
        PyfuncPricingPipeline(context="not a context")


def test_default_options(pipeline):
    assert pipeline.options["show_progress"] is False
    assert pipeline.layers == {}


# --------------------------------------------------------------------------- #
# add / remove parameter
# --------------------------------------------------------------------------- #

def test_add_parameter_is_chainable_and_registers(pipeline):
    a = ColumnCalculation("a", "policy", "x", "1")
    b = ColumnCalculation("b", "policy", "y", "2")
    result = pipeline.add_parameter(a).add_parameter(b)
    assert result is pipeline
    assert pipeline.get_layer_names() == ["a", "b"]


def test_add_parameter_rejects_non_layer(pipeline):
    with pytest.raises(TypeError, match="BasePricingLayer"):
        pipeline.add_parameter("not a layer")


def test_add_parameter_duplicate_warns_and_overwrites(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    with pytest.warns(UserWarning, match="already exists"):
        pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "999"))
    assert pipeline.layers["a"].calculation == "999"


def test_remove_parameter(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    pipeline.remove_parameter("a")
    assert pipeline.get_layer_names() == []


def test_remove_unknown_parameter_raises(pipeline):
    with pytest.raises(ValueError, match="not registered"):
        pipeline.remove_parameter("ghost")


# --------------------------------------------------------------------------- #
# Dependency resolution
# --------------------------------------------------------------------------- #

def _chain_pipeline(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "ca", "1"))
    pipeline.add_parameter(ColumnCalculation("b", "policy", "cb", "df['ca'] + 1",
                                             dependencies=["a"]))
    pipeline.add_parameter(ColumnCalculation("c", "policy", "cc", "df['cb'] + 1",
                                             dependencies=["b"]))
    return pipeline


def test_topological_order_respects_chain(pipeline):
    _chain_pipeline(pipeline)
    assert pipeline.get_execution_order() == ["a", "b", "c"]


def test_execution_order_is_cached_until_invalidated(pipeline):
    _chain_pipeline(pipeline)
    pipeline.get_execution_order()
    assert pipeline._execution_order is not None
    pipeline.add_parameter(ColumnCalculation("d", "policy", "cd", "1"))
    assert pipeline._execution_order is None


def test_missing_dependency_raises(pipeline):
    pipeline.add_parameter(
        ColumnCalculation("b", "policy", "x", "1", dependencies=["missing"])
    )
    with pytest.raises(DependencyError, match="not registered"):
        pipeline.get_execution_order()


def test_circular_dependency_detected(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1",
                                             dependencies=["b"]))
    pipeline.add_parameter(ColumnCalculation("b", "policy", "y", "1",
                                             dependencies=["a"]))
    with pytest.raises(DependencyError, match="[Cc]ircular"):
        pipeline.get_execution_order()


def test_dependency_graph(pipeline):
    _chain_pipeline(pipeline)
    assert pipeline.get_dependency_graph() == {
        "a": [], "b": ["a"], "c": ["b"],
    }


def test_build_waves_groups_independent_layers(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "ca", "1"))
    pipeline.add_parameter(ColumnCalculation("b", "policy", "cb", "2"))
    pipeline.add_parameter(
        ColumnCalculation("c", "policy", "cc", "df['ca'] + df['cb']",
                          dependencies=["a", "b"])
    )
    waves = pipeline._build_waves()
    assert sorted(waves[0]) == ["a", "b"]
    assert waves[1] == ["c"]


def test_build_waves_empty_pipeline(pipeline):
    assert pipeline._build_waves() == []


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #

def test_predict_single_layer(pipeline, sample_data):
    pipeline.add_parameter(
        ColumnCalculation("p", "policy", "premium2", "df['base_premium'] * 2")
    )
    out = pipeline.predict(sample_data)
    assert list(out["policy"]["premium2"]) == [200.0, 400.0, 600.0, 800.0]


def test_predict_does_not_mutate_input(pipeline, sample_data):
    pipeline.add_parameter(ColumnCalculation("p", "policy", "x", "1"))
    pipeline.predict(sample_data)
    assert "x" not in sample_data["policy"].columns


def test_predict_multi_wave_parallel_columnar(pipeline, sample_data):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "ca",
                                             "df['base_premium'] * 2"))
    pipeline.add_parameter(ColumnCalculation("b", "policy", "cb",
                                             "df['customer_tenure'] + 1"))
    pipeline.add_parameter(
        ColumnCalculation("c", "policy", "cc", "df['ca'] + df['cb']",
                          dependencies=["a", "b"])
    )
    out = pipeline.predict(sample_data)
    # row 0: 200 + 1 = 201
    assert out["policy"]["cc"].iloc[0] == 201.0


def test_predict_with_aggregation_and_join(pipeline, sample_data):
    pipeline.add_parameter(
        TableAggregation("agg", "claims_history", "claims_agg",
                         "policy_number", {"total_cost": ("claim_cost", "sum")})
    )
    pipeline.add_parameter(
        TableJoin("join", "policy", "claims_agg", left_on="policy_number",
                  columns=["total_cost"], dependencies=["agg"])
    )
    out = pipeline.predict(sample_data)
    assert "total_cost" in out["policy"].columns
    assert out["policy"]["total_cost"].sum() == 10500.0


def test_predict_propagates_layer_errors(pipeline, sample_data):
    from open_rating_engine.rating_engine.pricing_layers import LayerCalculationError

    pipeline.add_parameter(ColumnCalculation("bad", "policy", "x", "df['nope']"))
    with pytest.raises(LayerCalculationError):
        pipeline.predict(sample_data)


def test_set_catalog_accessible_in_calculation(pipeline, sample_data):
    pipeline.set_catalog("loading", 1.25)
    pipeline.add_parameter(
        ColumnCalculation("c", "policy", "loaded",
                          "df['base_premium'] * catalog['loading']")
    )
    out = pipeline.predict(sample_data)
    assert out["policy"]["loaded"].iloc[0] == 125.0


def test_profile_predict_returns_timing_frame(pipeline, sample_data, mock_context):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    pipeline.add_parameter(ColumnCalculation("b", "policy", "y", "2"))
    data, timing = pipeline.profile_predict(mock_context, sample_data)
    assert "x" in data["policy"].columns
    assert set(timing.columns) == {"wave", "layer", "type", "parallel", "elapsed_s"}
    assert set(timing["layer"]) == {"a", "b"}


def test_print_execution_plan_runs(pipeline, sample_data, capsys):
    _chain_pipeline(pipeline)
    pipeline.print_execution_plan()
    assert "Execution Plan" in capsys.readouterr().out


def test_plot_dag_builds_figure(pipeline, monkeypatch):
    _chain_pipeline(pipeline)
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    pipeline.plot_dag()  # must not raise


def test_custom_layer_in_pipeline(pipeline, sample_data):
    def add_total(data, catalog, options, context=None):
        data["policy"]["total"] = data["policy"]["base_premium"] * 1.1
        return data

    pipeline.add_parameter(CustomLayer("ct", add_total))
    out = pipeline.predict(sample_data)
    assert out["policy"]["total"].iloc[0] == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# SourceColumn integration
# --------------------------------------------------------------------------- #

def test_source_column_pipeline_identity_and_dependent(pipeline, sample_data):
    """A SourceColumn is a no-op root; a dependent ColumnCalculation that
    depends on it still computes correctly, and the source column is untouched."""
    pipeline.add_parameter(
        SourceColumn("policy.base_premium", "policy", "base_premium", dtype="numeric")
    )
    pipeline.add_parameter(
        ColumnCalculation(
            "doubled", "policy", "premium2", "df['base_premium'] * 2",
            dependencies=["policy.base_premium"],
        )
    )
    # topo-sort resolves (source layer is a registered wave-0 root)
    assert pipeline.get_execution_order()[0] == "policy.base_premium"
    out = pipeline.predict(sample_data)
    assert list(out["policy"]["base_premium"]) == [100.0, 200.0, 300.0, 400.0]
    assert list(out["policy"]["premium2"]) == [200.0, 400.0, 600.0, 800.0]


def test_source_column_routed_as_structural_not_columnar(pipeline):
    """Regression: a SourceColumn must NOT be sent through the columnar
    _compute_result path (it has no target_table/output_column)."""
    layer = SourceColumn("risk.driver_youngest_age", "risk", "driver_youngest_age")
    pipeline.add_parameter(layer)
    from open_rating_engine.rating_engine.pricing_pipeline import (
        TableTransform, TableJoin, TableDrop, RenameColumns, DropColumns,
        CustomLayer as _CL, SourceColumn as _SC,
    )
    structural_types = (
        TableTransform, TableJoin, TableDrop, RenameColumns, DropColumns, _CL, _SC,
    )
    assert isinstance(layer, structural_types)
    assert not hasattr(layer, "_compute_result")
