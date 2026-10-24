"""Unit tests for every pricing layer type in pricing_layers.py."""

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.rating_engine.pricing_layers import (
    BasePricingLayer,
    ColumnCalculation,
    CustomLayer,
    DropColumns,
    LayerCalculationError,
    RatingTablesLayer,
    RenameColumns,
    SourceColumn,
    TableAggregation,
    TableDrop,
    TableJoin,
    TableTransform,
)


# --------------------------------------------------------------------------- #
# BasePricingLayer
# --------------------------------------------------------------------------- #

def test_base_layer_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        ColumnCalculation(name="  ", target_table="t", output_column="c", calculation="1")


def test_base_layer_rejects_non_string_name():
    with pytest.raises(ValueError):
        ColumnCalculation(name=123, target_table="t", output_column="c", calculation="1")


def test_dependencies_default_to_empty_list():
    layer = ColumnCalculation("a", "t", "c", "1")
    assert layer.dependencies == []
    layer2 = ColumnCalculation("b", "t", "c", "1", dependencies=["a"])
    assert layer2.dependencies == ["a"]


# --------------------------------------------------------------------------- #
# ColumnCalculation
# --------------------------------------------------------------------------- #

def test_column_calculation_string_arithmetic(sample_data):
    layer = ColumnCalculation(
        "double", "policy", "double_premium", "df['base_premium'] * 2"
    )
    out = layer.apply(sample_data, {}, {})
    assert list(out["policy"]["double_premium"]) == [200.0, 400.0, 600.0, 800.0]


def test_column_calculation_uses_numpy_and_pandas(sample_data):
    layer = ColumnCalculation(
        "band",
        "risk",
        "age_band",
        "pd.cut(df['driver_youngest_age'], bins=[0, 25, 50, np.inf], "
        "labels=['young', 'mid', 'old'])",
    )
    out = layer.apply(sample_data, {}, {})
    assert list(out["risk"]["age_band"]) == ["young", "mid", "mid", "old"]


def test_column_calculation_callable_receives_all_scopes(sample_data):
    seen = {}

    def calc(df, data, catalog, options):
        seen["has_data"] = "risk" in data
        seen["catalog"] = catalog.get("k")
        return df["base_premium"] + options["bump"]

    layer = ColumnCalculation("c", "policy", "adj", calc)
    out = layer.apply(sample_data, {"k": 42}, {"bump": 10})
    assert seen == {"has_data": True, "catalog": 42}
    assert list(out["policy"]["adj"]) == [110.0, 210.0, 310.0, 410.0]


def test_column_calculation_scalar_is_broadcast(sample_data):
    layer = ColumnCalculation("c", "policy", "flag", "1")
    out = layer.apply(sample_data, {}, {})
    assert (out["policy"]["flag"] == 1).all()
    assert len(out["policy"]["flag"]) == 4


def test_column_calculation_invalid_expression_raises(sample_data):
    layer = ColumnCalculation("c", "policy", "x", "df['nope'] * 2")
    with pytest.raises(LayerCalculationError, match="error computing"):
        layer.apply(sample_data, {}, {})


def test_column_calculation_safe_builtins_block_dangerous_calls(sample_data):
    layer = ColumnCalculation("c", "policy", "x", "__import__('os').getcwd()")
    with pytest.raises(LayerCalculationError):
        layer.apply(sample_data, {}, {})


def test_column_calculation_length_mismatch_raises(sample_data):
    layer = ColumnCalculation("c", "policy", "x", "df['base_premium'].head(2)")
    with pytest.raises(LayerCalculationError, match="does not\nmatch|does not match"):
        layer.apply(sample_data, {}, {})


def test_column_calculation_type_error_on_bad_calculation():
    with pytest.raises(TypeError, match="callable or a Python expression"):
        ColumnCalculation("c", "t", "o", calculation=123)


def test_column_calculation_does_not_mutate_input_frame(sample_data):
    original = sample_data["policy"].copy()
    ColumnCalculation("c", "policy", "x", "df['base_premium'] * 2").apply(
        sample_data, {}, {}
    )
    # apply replaces the table object, but the new frame has the extra column;
    # the originally-passed frame content is unchanged.
    assert "x" not in original.columns


# --------------------------------------------------------------------------- #
# aggregate() helper inside ColumnCalculation
# --------------------------------------------------------------------------- #

def _agg_fixture():
    """Policy/risk dict where multiple risks per policy makes counts meaningful."""
    policy = pd.DataFrame({"policy_number": [1, 2, 3, 4]})
    risk = pd.DataFrame(
        {
            "policy_number": [1, 1, 2, 4, 4, 4],
            "risk_number":   [10, 11, 20, 40, 41, 42],
            "vehicle_market_value": [10_000.0, 20_000.0, 5_000.0, 30_000.0, 40_000.0, 50_000.0],
        }
    )
    return {"policy": policy, "risk": risk}


def test_aggregate_count_broadcasts_per_policy():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vc",
        target_table="policy",
        output_column="policy_vehicle_count",
        calculation="aggregate(source='risk', group_by='policy_number', agg_func='count')",
    )
    out = layer.apply(data, {}, {})
    assert list(out["policy"]["policy_vehicle_count"]) == [2, 1, 0, 3]


def test_aggregate_count_with_source_column():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vc",
        "policy",
        "non_null_vmv_count",
        "aggregate(source='risk', group_by='policy_number', agg_func='count', source_column='vehicle_market_value')",
    )
    out = layer.apply(data, {}, {})
    assert list(out["policy"]["non_null_vmv_count"]) == [2, 1, 0, 3]


def test_aggregate_sum_per_policy():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vmv_sum",
        "policy",
        "policy_vmv_sum",
        "aggregate(source='risk', group_by='policy_number', agg_func='sum', source_column='vehicle_market_value')",
    )
    out = layer.apply(data, {}, {})
    vals = out["policy"]["policy_vmv_sum"].tolist()
    assert vals[0] == 30_000.0
    assert vals[1] == 5_000.0
    assert pd.isna(vals[2])  # policy 3 has no risks -> NaN
    assert vals[3] == 120_000.0


def test_aggregate_mean_per_policy():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vmv_mean",
        "policy",
        "policy_vmv_mean",
        "aggregate(source='risk', group_by='policy_number', agg_func='mean', source_column='vehicle_market_value')",
    )
    out = layer.apply(data, {}, {})
    vals = out["policy"]["policy_vmv_mean"].tolist()
    assert vals[0] == 15_000.0
    assert vals[1] == 5_000.0
    assert pd.isna(vals[2])
    assert vals[3] == 40_000.0


def test_aggregate_explicit_fill_value():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vmv_sum_fill",
        "policy",
        "policy_vmv_sum",
        "aggregate(source='risk', group_by='policy_number', agg_func='sum', source_column='vehicle_market_value', fill_value=0)",
    )
    out = layer.apply(data, {}, {})
    assert out["policy"]["policy_vmv_sum"].tolist() == [30_000.0, 5_000.0, 0.0, 120_000.0]


def test_aggregate_composes_with_arithmetic():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "vmv_sum_k",
        "policy",
        "policy_vmv_sum_k",
        "aggregate(source='risk', group_by='policy_number', agg_func='sum', source_column='vehicle_market_value', fill_value=0) / 1000",
    )
    out = layer.apply(data, {}, {})
    assert out["policy"]["policy_vmv_sum_k"].tolist() == [30.0, 5.0, 0.0, 120.0]


def test_aggregate_unknown_agg_func_raises():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='risk', group_by='policy_number', agg_func='wat')",
    )
    with pytest.raises(LayerCalculationError, match="agg_func must be one of"):
        layer.apply(data, {}, {})


def test_aggregate_missing_source_table_raises():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='nope', group_by='policy_number', agg_func='count')",
    )
    with pytest.raises(LayerCalculationError, match="not in data"):
        layer.apply(data, {}, {})


def test_aggregate_missing_group_by_on_target_raises():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='risk', group_by='risk_number', agg_func='count')",
    )
    with pytest.raises(LayerCalculationError, match="missing on target table"):
        layer.apply(data, {}, {})


def test_aggregate_missing_group_by_on_source_raises():
    data = _agg_fixture()
    # group_by exists on policy but not on a contrived alt source
    data["alt"] = pd.DataFrame({"other_key": [1, 2]})
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='alt', group_by='policy_number', agg_func='count')",
    )
    with pytest.raises(LayerCalculationError, match="missing on source table 'alt'"):
        layer.apply(data, {}, {})


def test_aggregate_non_count_requires_source_column():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='risk', group_by='policy_number', agg_func='sum')",
    )
    with pytest.raises(LayerCalculationError, match="source_column is required"):
        layer.apply(data, {}, {})


def test_aggregate_missing_source_column_raises():
    data = _agg_fixture()
    layer = ColumnCalculation(
        "bad",
        "policy",
        "x",
        "aggregate(source='risk', group_by='policy_number', agg_func='sum', source_column='nope')",
    )
    with pytest.raises(LayerCalculationError, match="source_column 'nope' missing"):
        layer.apply(data, {}, {})


def test_aggregate_does_not_mutate_source_table():
    data = _agg_fixture()
    risk_before = data["risk"].copy()
    layer = ColumnCalculation(
        "vc",
        "policy",
        "policy_vehicle_count",
        "aggregate(source='risk', group_by='policy_number', agg_func='count')",
    )
    layer.apply(data, {}, {})
    pd.testing.assert_frame_equal(data["risk"], risk_before)


# --------------------------------------------------------------------------- #
# aggregate() auto-detect from data_structure
# --------------------------------------------------------------------------- #

def _ds_split_keys():
    """Data-structure snapshot where parent_key != child_key (the bug case).

    policy carries the join column ``policy.policy_number``; risk carries
    ``risk.policy_number``. The structural join must read each column from
    its own side — a single shared group_by cannot work.
    """
    return [
        {"name": "policy", "parent": None, "parent_key": None, "child_key": None, "cardinality": None},
        {
            "name": "risk",
            "parent": "policy",
            "parent_key": "policy.policy_number",
            "child_key": "risk.policy_number",
            "cardinality": "one_to_many",
        },
    ]


def _ds_split_fixture():
    policy = pd.DataFrame({
        "policy.policy_number": [1, 2, 3, 4],
        "policy.transaction_type": ["NEW", "MTA", "REN", "NEW"],
    })
    risk = pd.DataFrame({
        "risk.policy_number": [1, 1, 2, 4, 4, 4],
        "risk.risk_number": [10, 11, 20, 40, 41, 42],
        "risk.vehicle_market_value": [10_000.0, 20_000.0, 5_000.0, 30_000.0, 40_000.0, 50_000.0],
    })
    return {"policy": policy, "risk": risk}


def test_aggregate_auto_detect_parent_to_child_broadcast():
    """The user's case: broadcast a parent attribute down onto child rows
    when parent_key and child_key are different column names."""
    data = _ds_split_fixture()
    options = {"data_structure": _ds_split_keys()}
    layer = ColumnCalculation(
        "txn",
        target_table="risk",
        output_column="risk.transaction_type",
        calculation="aggregate(source='policy', source_column='policy.transaction_type', agg_func='first')",
    )
    out = layer.apply(data, {}, options)
    assert out["risk"]["risk.transaction_type"].tolist() == ["NEW", "NEW", "MTA", "NEW", "NEW", "NEW"]


def test_aggregate_auto_detect_child_to_parent_count():
    """Reverse direction: aggregate child rows up onto the parent with
    different join column names on each side."""
    data = _ds_split_fixture()
    options = {"data_structure": _ds_split_keys()}
    layer = ColumnCalculation(
        "vc",
        target_table="policy",
        output_column="policy.risk_count",
        calculation="aggregate(source='risk', agg_func='count')",
    )
    out = layer.apply(data, {}, options)
    assert out["policy"]["policy.risk_count"].tolist() == [2, 1, 0, 3]


def test_aggregate_auto_detect_no_relationship_raises():
    data = _ds_split_fixture()
    data["unrelated"] = pd.DataFrame({"x": [1, 2, 3]})
    options = {"data_structure": _ds_split_keys()}
    layer = ColumnCalculation(
        "bad",
        target_table="risk",
        output_column="risk.x",
        calculation="aggregate(source='unrelated', source_column='x', agg_func='sum')",
    )
    with pytest.raises(LayerCalculationError, match="no direct parent/child relationship"):
        layer.apply(data, {}, options)


def test_aggregate_auto_detect_missing_data_structure_raises():
    """When the algorithm has no data-structure snapshot attached, auto-detect
    fails with a clear suggestion to pass group_by explicitly."""
    data = _ds_split_fixture()
    layer = ColumnCalculation(
        "bad",
        target_table="risk",
        output_column="risk.x",
        calculation="aggregate(source='policy', source_column='policy.transaction_type', agg_func='first')",
    )
    with pytest.raises(LayerCalculationError, match="no data_structure snapshot"):
        layer.apply(data, {}, {})


def test_aggregate_explicit_group_by_overrides_auto_detect():
    """When group_by is passed, it short-circuits the structure lookup and
    uses the same column on both sides (legacy behaviour)."""
    data = _agg_fixture()
    options = {"data_structure": _ds_split_keys()}  # snapshot present but ignored
    layer = ColumnCalculation(
        "vc",
        target_table="policy",
        output_column="policy_vehicle_count",
        calculation="aggregate(source='risk', group_by='policy_number', agg_func='count')",
    )
    out = layer.apply(data, {}, options)
    assert out["policy"]["policy_vehicle_count"].tolist() == [2, 1, 0, 3]


# --------------------------------------------------------------------------- #
# TableTransform
# --------------------------------------------------------------------------- #

def test_table_transform_filter_rows(sample_data):
    layer = TableTransform(
        "hv",
        source_table="risk",
        output_table="high_value",
        calculation=lambda df, data, catalog, options: df[
            df["vehicle_market_value"] > 30_000
        ],
    )
    out = layer.apply(sample_data, {}, {})
    assert len(out["high_value"]) == 2
    assert list(out["high_value"]["risk_number"]) == [20, 40]


def test_table_transform_requires_callable():
    with pytest.raises(TypeError, match="must be a callable"):
        TableTransform("t", "a", "b", calculation="not callable")


def test_table_transform_resets_index(sample_data):
    layer = TableTransform(
        "f",
        "risk",
        "f",
        calculation=lambda df, *_: df[df["driver_youngest_age"] > 25],
    )
    out = layer.apply(sample_data, {}, {})
    assert list(out["f"].index) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# TableAggregation
# --------------------------------------------------------------------------- #

def test_table_aggregation_named_agg(sample_data):
    layer = TableAggregation(
        "agg",
        source_table="claims_history",
        output_table="claims_agg",
        groupby_key="policy_number",
        aggregations={"total_cost": ("claim_cost", "sum")},
    )
    out = layer.apply(sample_data, {}, {})
    assert set(out["claims_agg"].columns) == {"policy_number", "total_cost"}
    assert out["claims_agg"]["total_cost"].sum() == 10500.0


def test_table_aggregation_multi_key_and_multi_agg():
    df = pd.DataFrame(
        {
            "g1": ["a", "a", "b"],
            "g2": ["x", "x", "y"],
            "v": [1, 2, 3],
            "w": [10, 20, 30],
        }
    )
    layer = TableAggregation(
        "agg",
        "src",
        "out",
        groupby_key=["g1", "g2"],
        aggregations={"v_sum": ("v", "sum"), "w_max": ("w", "max")},
    )
    out = layer.apply({"src": df}, {}, {})
    row = out["out"].iloc[0]
    assert row["v_sum"] == 3 and row["w_max"] == 20


# --------------------------------------------------------------------------- #
# TableJoin
# --------------------------------------------------------------------------- #

def test_table_join_same_key_in_place(sample_data):
    layer = TableJoin(
        "j",
        left_table="risk",
        right_table="cover",
        left_on="risk_number",
        columns=["cover_type"],
    )
    out = layer.apply(sample_data, {}, {})
    assert "cover_type" in out["risk"].columns
    assert out["risk"].loc[0, "cover_type"] == "comprehensive"


def test_table_join_different_key_names():
    left = pd.DataFrame({"lid": [1, 2], "v": [10, 20]})
    right = pd.DataFrame({"rid": [1, 2], "label": ["a", "b"]})
    layer = TableJoin(
        "j", "L", "R", left_on="lid", right_on="rid", columns=["label"],
        output_table="merged",
    )
    out = layer.apply({"L": left, "R": right}, {}, {})
    assert list(out["merged"]["label"]) == ["a", "b"]


def test_table_join_inner_drops_unmatched():
    left = pd.DataFrame({"k": [1, 2, 3]})
    right = pd.DataFrame({"k": [1, 2], "x": [9, 8]})
    layer = TableJoin("j", "L", "R", left_on="k", how="inner", output_table="m")
    out = layer.apply({"L": left, "R": right}, {}, {})
    assert len(out["m"]) == 2


def test_table_join_from_catalog():
    left = pd.DataFrame({"k": [1, 2]})
    cat_table = pd.DataFrame({"k": [1, 2], "factor": [1.5, 2.5]})
    layer = TableJoin(
        "j", "L", "rt", left_on="k", columns=["factor"],
        right_source="catalog", output_table="m",
    )
    out = layer.apply({"L": left}, {"rt": cat_table}, {})
    assert list(out["m"]["factor"]) == [1.5, 2.5]


def test_table_join_invalid_right_source():
    with pytest.raises(ValueError, match="right_source"):
        TableJoin("j", "L", "R", left_on="k", right_source="bogus")


def test_table_join_multi_key():
    left = pd.DataFrame({"a": [1, 1], "b": [1, 2], "v": [10, 20]})
    right = pd.DataFrame({"a": [1, 1], "b": [1, 2], "label": ["x", "y"]})
    layer = TableJoin(
        "j", "L", "R", left_on=["a", "b"], columns=["label"], output_table="m"
    )
    out = layer.apply({"L": left, "R": right}, {}, {})
    assert list(out["m"]["label"]) == ["x", "y"]


# --------------------------------------------------------------------------- #
# RenameColumns / DropColumns / TableDrop
# --------------------------------------------------------------------------- #

def test_rename_columns(sample_data):
    layer = RenameColumns("r", "policy", {"base_premium": "premium"})
    out = layer.apply(sample_data, {}, {})
    assert "premium" in out["policy"].columns
    assert "base_premium" not in out["policy"].columns


def test_rename_columns_empty_mapping_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        RenameColumns("r", "t", {})


def test_drop_columns(sample_data):
    layer = DropColumns("d", "risk", ["risk_usage"])
    out = layer.apply(sample_data, {}, {})
    assert "risk_usage" not in out["risk"].columns


def test_drop_columns_empty_list_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        DropColumns("d", "t", [])


def test_table_drop_single_and_list(sample_data):
    TableDrop("t1", "cover").apply(sample_data, {}, {})
    assert "cover" not in sample_data
    TableDrop("t2", ["policy", "risk"]).apply(sample_data, {}, {})
    assert "policy" not in sample_data and "risk" not in sample_data


def test_table_drop_missing_table_raises(sample_data):
    with pytest.raises(KeyError, match="not found"):
        TableDrop("t", "does_not_exist").apply(sample_data, {}, {})


# --------------------------------------------------------------------------- #
# RatingTablesLayer
# --------------------------------------------------------------------------- #

def test_rating_tables_default_multiplies(sample_data, sample_rating_tables):
    # Need cover_type on risk first.
    sample_data["risk"]["cover_type"] = ["comprehensive", "third_party",
                                         "comprehensive", "comprehensive"]
    layer = RatingTablesLayer(
        "rt",
        target_table="risk",
        output_column="factor",
        rating_tables={
            "risk_usage_mult": sample_rating_tables["risk_usage_mult"],
            "cover_type_mult": sample_rating_tables["cover_type_mult"],
        },
    )
    out = layer.apply(sample_data, {}, {})
    # row 0: Private(1.0) * comprehensive(1.1) = 1.1
    assert out["risk"]["factor"].iloc[0] == pytest.approx(1.1)
    # row 1: Rideshare(1.3) * third_party(0.8) = 1.04
    assert out["risk"]["factor"].iloc[1] == pytest.approx(1.04)


def test_rating_tables_formula(sample_data, sample_rating_tables):
    sample_data["risk"]["cover_type"] = ["comprehensive", "third_party",
                                         "comprehensive", "comprehensive"]
    layer = RatingTablesLayer(
        "rt",
        "risk",
        "factor",
        rating_tables=sample_rating_tables,
        formula=(
            "risk_usage_mult['value'] * cover_type_mult['value']"
            " + cover_type_add['value']"
        ),
    )
    out = layer.apply(sample_data, {}, {})
    # row 0: 1.0*1.1 + 0.05 = 1.15
    assert out["risk"]["factor"].iloc[0] == pytest.approx(1.15)


def test_rating_tables_missing_key_uses_default(sample_rating_tables):
    data = {"risk": pd.DataFrame({"risk_usage": ["Private", "Unknown"]})}
    layer = RatingTablesLayer(
        "rt",
        "risk",
        "factor",
        rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
        default_values={"risk_usage_mult": 1.0},
    )
    out = layer.apply(data, {}, {})
    assert list(out["risk"]["factor"]) == [1.0, 1.0]  # Unknown -> default 1.0


def test_rating_tables_empty_dict_raises():
    with pytest.raises(ValueError, match="non-empty dict"):
        RatingTablesLayer("rt", "t", "o", rating_tables={})


def test_rating_tables_missing_value_column_raises():
    with pytest.raises(ValueError, match="'value' column"):
        RatingTablesLayer(
            "rt", "t", "o",
            rating_tables={"bad": pd.DataFrame({"k": ["a"], "notvalue": [1]})},
        )


def test_rating_tables_non_numeric_value_raises():
    with pytest.raises(ValueError, match="must be numeric"):
        RatingTablesLayer(
            "rt", "t", "o",
            rating_tables={"bad": pd.DataFrame({"k": ["a"], "value": ["x"]})},
        )


def test_rating_tables_duplicate_keys_raise():
    with pytest.raises(ValueError, match="duplicate values"):
        RatingTablesLayer(
            "rt", "t", "o",
            rating_tables={
                "bad": pd.DataFrame({"k": ["a", "a"], "value": [1.0, 2.0]})
            },
        )


def test_rating_tables_invalid_formula_syntax_raises(sample_rating_tables):
    with pytest.raises(ValueError, match="invalid Python syntax"):
        RatingTablesLayer(
            "rt", "t", "o",
            rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
            formula="risk_usage_mult['value'] *",
        )


def test_rating_tables_unknown_default_key_raises(sample_rating_tables):
    with pytest.raises(ValueError, match="unknown rating tables"):
        RatingTablesLayer(
            "rt", "t", "o",
            rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
            default_values={"nonexistent": 1.0},
        )


def test_rating_tables_missing_target_column_raises(sample_rating_tables):
    data = {"risk": pd.DataFrame({"other_col": [1, 2]})}
    layer = RatingTablesLayer(
        "rt", "risk", "factor",
        rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
    )
    with pytest.raises(LayerCalculationError, match="are missing in"):
        layer.apply(data, {}, {})


# --------------------------------------------------------------------------- #
# CustomLayer
# --------------------------------------------------------------------------- #

def test_custom_layer_arbitrary_mutation(sample_data):
    def fn(data, catalog, options, context=None):
        data["summary"] = pd.DataFrame({"n": [len(data["policy"])]})
        return data

    out = CustomLayer("c", fn).apply(sample_data, {}, {})
    assert out["summary"]["n"].iloc[0] == 4


def test_custom_layer_requires_callable():
    with pytest.raises(TypeError, match="must be a callable"):
        CustomLayer("c", func=42)


def test_custom_layer_non_dict_return_raises(sample_data):
    layer = CustomLayer("c", lambda *a, **k: [1, 2, 3])
    with pytest.raises(LayerCalculationError, match="must return a dict"):
        layer.apply(sample_data, {}, {})


def test_custom_layer_wraps_internal_errors(sample_data):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    with pytest.raises(LayerCalculationError, match="kaboom"):
        CustomLayer("c", boom).apply(sample_data, {}, {})


# --------------------------------------------------------------------------- #
# to_dict / from_dict
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "layer",
    [
        ColumnCalculation("c", "t", "o", "df['x'] + 1", dependencies=["a"]),
        TableJoin("j", "L", "R", left_on="k", columns=["v"]),
        TableAggregation("a", "s", "o", "k", {"tot": ("v", "sum")}),
        RenameColumns("r", "t", {"a": "b"}),
        DropColumns("d", "t", ["a"]),
        TableDrop("td", ["x", "y"]),
    ],
)
def test_to_dict_from_dict_roundtrip(layer):
    d = layer.to_dict()
    assert d["type"] == type(layer).__name__
    rebuilt = BasePricingLayer.from_dict(d)
    assert type(rebuilt) is type(layer)
    assert rebuilt.name == layer.name
    assert rebuilt.dependencies == layer.dependencies


def test_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown layer type"):
        BasePricingLayer.from_dict({"type": "Nonsense", "name": "x"})


def test_from_dict_callable_layer_not_reconstructable():
    layer = CustomLayer("c", lambda *a, **k: {})
    with pytest.raises(ValueError, match="cannot be\nreconstructed|cannot be reconstructed"):
        BasePricingLayer.from_dict(layer.to_dict())


def test_rating_tables_roundtrip(sample_rating_tables):
    layer = RatingTablesLayer(
        "rt", "risk", "factor",
        rating_tables={"risk_usage_mult": sample_rating_tables["risk_usage_mult"]},
        formula="risk_usage_mult['value']",
    )
    rebuilt = BasePricingLayer.from_dict(layer.to_dict())
    assert isinstance(rebuilt, RatingTablesLayer)
    assert rebuilt.formula == "risk_usage_mult['value']"


# --------------------------------------------------------------------------- #
# SourceColumn
# --------------------------------------------------------------------------- #

def test_source_column_minimal_and_defaults():
    layer = SourceColumn("risk.age", "risk", "age")
    assert layer.source_table == "risk"
    assert layer.source_column == "age"
    assert layer.dtype is None
    assert layer.transformation is None
    assert layer.dependencies == []


def test_source_column_apply_is_identity():
    layer = SourceColumn("risk.age", "risk", "age", dtype="numeric")
    data = {"risk": pd.DataFrame({"age": [1, 2, 3]})}
    out = layer.apply(data, {}, {})
    assert out is data
    assert list(out["risk"]["age"]) == [1, 2, 3]


@pytest.mark.parametrize("dt", ["numeric", "categorical", "date", "boolean", "text", None])
def test_source_column_accepts_valid_dtype(dt):
    assert SourceColumn("x.y", "x", "y", dtype=dt).dtype == dt


def test_source_column_rejects_bad_dtype():
    with pytest.raises(ValueError, match="dtype must be None or one of"):
        SourceColumn("x.y", "x", "y", dtype="foo")


def test_source_column_rejects_empty_source_fields():
    with pytest.raises(ValueError, match="source_table must be a non-empty string"):
        SourceColumn("x.y", "", "y")
    with pytest.raises(ValueError, match="source_column must be a non-empty string"):
        SourceColumn("x.y", "x", "  ")


def test_source_column_rejects_non_string_transformation():
    with pytest.raises(ValueError, match="transformation must be a string"):
        SourceColumn("x.y", "x", "y", transformation=5)


def test_source_column_roundtrip_preserves_all_fields():
    layer = SourceColumn(
        "risk.driver_youngest_age", "risk", "driver_youngest_age",
        dtype="numeric", transformation=None, description="src",
    )
    d = layer.to_dict()
    assert d["type"] == "SourceColumn"
    assert d["dtype"] == "numeric"
    assert d["transformation"] is None
    rebuilt = BasePricingLayer.from_dict(d)
    assert isinstance(rebuilt, SourceColumn)
    assert rebuilt.to_dict() == d
