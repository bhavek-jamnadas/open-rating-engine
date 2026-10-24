"""Unit tests for the aging preprocessor.

Pure tests against ``open_rating_engine.data_processing.data_aging.apply_aging``
— no DB, no FastAPI. The function operates on resolved ``layer_configs``
and an ``input_frames`` dict the way a scoring pipeline driver calls it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.data_processing.data_aging import AgingRuleError, apply_aging


def _source_column_layer(
    table: str,
    column: str,
    *,
    expression: str | None = None,
    enabled: bool = True,
    dtype: str = "numeric",
) -> dict:
    cfg = {
        "type": "SourceColumn",
        "name": f"{table}.{column}",
        "source_table": table,
        "source_column": column,
        "dtype": dtype,
        "dependencies": [],
    }
    if expression is not None:
        cfg["aging_rule"] = {"expression": expression, "enabled": enabled}
    return cfg


# ── No-op paths ───────────────────────────────────────────────────────────────


def test_apply_aging_no_layers_is_noop():
    frames = {"risk": pd.DataFrame({"x": [1, 2, 3]})}
    result = apply_aging({}, frames)
    assert result is frames
    pd.testing.assert_series_equal(result["risk"]["x"], pd.Series([1, 2, 3], name="x"))


def test_apply_aging_no_rule_on_source_column_is_noop():
    """SourceColumn without an ``aging_rule`` key (the default state) is skipped."""
    layers = {"risk.x": _source_column_layer("risk", "x")}
    frames = {"risk": pd.DataFrame({"x": [1, 2, 3]})}
    apply_aging(layers, frames)
    pd.testing.assert_series_equal(frames["risk"]["x"], pd.Series([1, 2, 3], name="x"))


def test_apply_aging_disabled_rule_is_noop():
    layers = {
        "risk.x": _source_column_layer(
            "risk", "x", expression="df['x'] + 999", enabled=False
        ),
    }
    frames = {"risk": pd.DataFrame({"x": [1, 2, 3]})}
    apply_aging(layers, frames)
    pd.testing.assert_series_equal(frames["risk"]["x"], pd.Series([1, 2, 3], name="x"))


def test_apply_aging_skips_non_source_column_layers():
    layers = {
        "risk.derived": {
            "type": "ColumnCalculation",
            "name": "risk.derived",
            "target_table": "risk",
            "output_column": "derived",
            "calculation": "df['x'] * 2",
            "aging_rule": {"expression": "df['x'] + 999", "enabled": True},
        },
    }
    frames = {"risk": pd.DataFrame({"x": [1, 2, 3]})}
    apply_aging(layers, frames)
    pd.testing.assert_series_equal(frames["risk"]["x"], pd.Series([1, 2, 3], name="x"))


def test_apply_aging_skips_when_source_table_absent():
    """The pipeline may be configured for tables the caller didn't supply."""
    layers = {
        "vehicle.value": _source_column_layer(
            "vehicle", "value", expression="df['value'] * 0.9"
        ),
    }
    frames = {"risk": pd.DataFrame({"x": [1, 2, 3]})}
    apply_aging(layers, frames)
    assert "vehicle" not in frames
    pd.testing.assert_series_equal(frames["risk"]["x"], pd.Series([1, 2, 3], name="x"))


# ── Happy paths ───────────────────────────────────────────────────────────────


def test_apply_aging_numeric_increment():
    layers = {
        "driver.age": _source_column_layer(
            "driver", "age", expression="df['age'] + 1"
        ),
    }
    frames = {"driver": pd.DataFrame({"age": [25, 40, 55]})}
    apply_aging(layers, frames)
    pd.testing.assert_series_equal(
        frames["driver"]["age"], pd.Series([26, 41, 56], name="age"),
    )


def test_apply_aging_numeric_decay():
    layers = {
        "vehicle.value": _source_column_layer(
            "vehicle", "value", expression="df['value'] * 0.9"
        ),
    }
    frames = {"vehicle": pd.DataFrame({"value": [100_000.0, 50_000.0]})}
    apply_aging(layers, frames)
    pd.testing.assert_series_equal(
        frames["vehicle"]["value"],
        pd.Series([90_000.0, 45_000.0], name="value"),
    )


def test_apply_aging_date_shift():
    layers = {
        "policy.eff_date": _source_column_layer(
            "policy",
            "eff_date",
            expression="df['eff_date'] + pd.DateOffset(years=1)",
            dtype="date",
        ),
    }
    frames = {"policy": pd.DataFrame({
        "eff_date": pd.to_datetime(["2025-06-01", "2024-12-31"])
    })}
    apply_aging(layers, frames)
    expected = pd.to_datetime(["2026-06-01", "2025-12-31"])
    pd.testing.assert_series_equal(
        frames["policy"]["eff_date"],
        pd.Series(expected, name="eff_date"),
    )


def test_apply_aging_multiple_rules_independent():
    layers = {
        "driver.age": _source_column_layer(
            "driver", "age", expression="df['age'] + 1"
        ),
        "vehicle.value": _source_column_layer(
            "vehicle", "value", expression="df['value'] * 0.9"
        ),
    }
    frames = {
        "driver": pd.DataFrame({"age": [25]}),
        "vehicle": pd.DataFrame({"value": [100.0]}),
    }
    apply_aging(layers, frames)
    assert frames["driver"]["age"].tolist() == [26]
    assert frames["vehicle"]["value"].tolist() == [90.0]


def test_apply_aging_other_columns_untouched():
    layers = {
        "risk.x": _source_column_layer(
            "risk", "x", expression="df['x'] + 1"
        ),
    }
    frames = {"risk": pd.DataFrame({"x": [1, 2], "y": [10, 20]})}
    apply_aging(layers, frames)
    assert frames["risk"]["x"].tolist() == [2, 3]
    assert frames["risk"]["y"].tolist() == [10, 20]


def test_apply_aging_returns_same_dict_for_chaining():
    """Multi-year drivers rely on chaining apply_aging → predict → apply_aging."""
    layers = {}
    frames = {"risk": pd.DataFrame({"x": [1]})}
    assert apply_aging(layers, frames) is frames


# ── Null + scalar broadcasting ────────────────────────────────────────────────


def test_apply_aging_nulls_propagate():
    layers = {
        "risk.age": _source_column_layer(
            "risk", "age", expression="df['age'] + 1"
        ),
    }
    frames = {"risk": pd.DataFrame({"age": [25.0, np.nan, 40.0]})}
    apply_aging(layers, frames)
    assert frames["risk"]["age"].iloc[0] == 26.0
    assert pd.isna(frames["risk"]["age"].iloc[1])
    assert frames["risk"]["age"].iloc[2] == 41.0


def test_apply_aging_scalar_result_broadcasts():
    """An expression that yields a scalar (e.g. ``0``) is broadcast to df.index."""
    layers = {
        "risk.flag": _source_column_layer(
            "risk", "flag", expression="0"
        ),
    }
    frames = {"risk": pd.DataFrame({"flag": [1, 1, 1]})}
    apply_aging(layers, frames)
    assert frames["risk"]["flag"].tolist() == [0, 0, 0]


# ── Error surfaces ────────────────────────────────────────────────────────────


def test_apply_aging_expression_raises_surfaced_with_context():
    layers = {
        "risk.age": _source_column_layer(
            "risk", "age", expression="df['nope'] + 1"
        ),
    }
    frames = {"risk": pd.DataFrame({"age": [25]})}
    with pytest.raises(AgingRuleError) as exc_info:
        apply_aging(layers, frames)
    msg = str(exc_info.value)
    assert "risk.age" in msg
    assert "nope" in msg


def test_apply_aging_wrong_length_result_surfaced():
    """Series of mismatched length is rejected with a clear error."""
    layers = {
        "risk.age": _source_column_layer(
            "risk", "age", expression="pd.Series([1, 2])"
        ),
    }
    frames = {"risk": pd.DataFrame({"age": [25, 30, 35]})}
    with pytest.raises(AgingRuleError) as exc_info:
        apply_aging(layers, frames)
    msg = str(exc_info.value)
    assert "risk.age" in msg
    assert "length" in msg


# ── Column-isolation guarantee (no-override invariant proxy) ──────────────────


def test_apply_aging_only_writes_to_named_source_column():
    """Aging never writes to any column other than ``source_column``. This is
    what makes the no-override invariant sufficient: every downstream layer's
    column producer is untouched after aging."""
    layers = {
        "risk.age": _source_column_layer(
            "risk", "age", expression="df['age'] + 1"
        ),
    }
    frames = {"risk": pd.DataFrame({
        "age": [25],
        "vehicle_id": ["V1"],   # produced by a join in the real DAG; aging must leave it alone
        "premium": [500.0],     # produced by an upstream score; aging must leave it alone
    })}
    apply_aging(layers, frames)
    assert frames["risk"]["age"].tolist() == [26]
    assert frames["risk"]["vehicle_id"].tolist() == ["V1"]
    assert frames["risk"]["premium"].tolist() == [500.0]
