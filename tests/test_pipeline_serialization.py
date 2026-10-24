"""to_config / from_config round-trips and catalog serialization."""

import pandas as pd
import pytest

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import (
    ColumnCalculation,
    RatingTablesLayer,
    SourceColumn,
    TableAggregation,
    TableJoin,
)


def test_to_config_structure(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    cfg = pipeline.to_config()
    assert set(cfg) == {"layers", "execution_order", "catalog", "data_structure", "context"}
    assert cfg["execution_order"] == ["a"]
    assert cfg["layers"]["a"]["type"] == "ColumnCalculation"
    assert cfg["context"]["model_config"]["environment"] == "local"
    assert cfg["data_structure"] == []


def test_data_structure_roundtrips_through_config(pipeline):
    """The data_structure snapshot persists across to_config / from_config so
    that re-hydrated pipelines can still auto-derive aggregate() join keys."""
    snapshot = [
        {"name": "policy", "parent": None, "parent_key": None, "child_key": None, "cardinality": None},
        {
            "name": "risk",
            "parent": "policy",
            "parent_key": "policy.policy_number",
            "child_key": "risk.policy_number",
            "cardinality": "one_to_many",
        },
    ]
    pipeline.options["data_structure"] = snapshot
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))
    cfg = pipeline.to_config()
    assert cfg["data_structure"] == snapshot

    rebuilt = PyfuncPricingPipeline.from_config(cfg)
    assert rebuilt.options["data_structure"] == snapshot


def test_from_config_rebuilds_pipeline(pipeline):
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "df['base_premium']"))
    pipeline.add_parameter(
        TableAggregation("agg", "claims_history", "ca", "policy_number",
                         {"tot": ("claim_cost", "sum")})
    )
    pipeline.add_parameter(
        TableJoin("j", "policy", "ca", left_on="policy_number",
                  columns=["tot"], dependencies=["agg"])
    )
    cfg = pipeline.to_config()

    rebuilt = PyfuncPricingPipeline.from_config(cfg)
    assert rebuilt.get_layer_names() == pipeline.get_layer_names()
    assert rebuilt.get_dependency_graph() == pipeline.get_dependency_graph()


def test_from_config_roundtrip_preserves_predictions(pipeline, sample_data):
    pipeline.add_parameter(
        ColumnCalculation("p", "policy", "double", "df['base_premium'] * 2")
    )
    rebuilt = PyfuncPricingPipeline.from_config(pipeline.to_config())
    rebuilt.options["show_progress"] = False
    out = rebuilt.predict(sample_data)
    assert list(out["policy"]["double"]) == [200.0, 400.0, 600.0, 800.0]


def test_catalog_dataframe_roundtrip(pipeline):
    pipeline.set_catalog("rates", pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}))
    pipeline.set_catalog("scalar", {"loading": 1.1})
    pipeline.add_parameter(ColumnCalculation("a", "policy", "x", "1"))

    rebuilt = PyfuncPricingPipeline.from_config(pipeline.to_config())
    pd.testing.assert_frame_equal(
        rebuilt.catalog["rates"].reset_index(drop=True),
        pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}),
    )
    assert rebuilt.catalog["scalar"] == {"loading": 1.1}


def test_rating_tables_layer_config_roundtrip(pipeline, sample_rating_tables):
    pipeline.add_parameter(
        RatingTablesLayer(
            "rt", "risk", "factor",
            rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
            formula="risk_usage_mult['value']",
        )
    )
    rebuilt = PyfuncPricingPipeline.from_config(pipeline.to_config())
    layer = rebuilt.layers["rt"]
    assert isinstance(layer, RatingTablesLayer)
    assert layer.formula == "risk_usage_mult['value']"


def test_source_column_config_roundtrip(pipeline):
    pipeline.add_parameter(
        SourceColumn(
            "risk.driver_youngest_age", "risk", "driver_youngest_age",
            dtype="numeric",
        )
    )
    rebuilt = PyfuncPricingPipeline.from_config(pipeline.to_config())
    layer = rebuilt.layers["risk.driver_youngest_age"]
    assert isinstance(layer, SourceColumn)
    assert layer.source_table == "risk"
    assert layer.source_column == "driver_youngest_age"
    assert layer.dtype == "numeric"
    assert layer.transformation is None
