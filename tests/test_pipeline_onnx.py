"""OnnxScoring — exercised with the onnxruntime seam mocked.

We patch ``_load_model`` so no real ``onnxruntime.InferenceSession`` is built;
:class:`FakeSession` returns the row-wise sum of the fed feature columns, which
is enough to assert the select/rename/feed/output plumbing.
"""

from copy import deepcopy

import numpy as np
import pytest

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import (
    OnnxScoring,
    LayerCalculationError,
)
from tests.mocks.onnx_mock import make_fake_load_model

INPUTS = ["driver_youngest_age", "vehicle_market_value"]


@pytest.fixture
def patch_onnx(monkeypatch):
    monkeypatch.setattr(
        OnnxScoring, "_load_model", staticmethod(make_fake_load_model(INPUTS))
    )


def _layer(column_map=None):
    return OnnxScoring(
        name="score",
        target_table="risk",
        output_column="prediction",
        column_map=column_map or {c: c for c in INPUTS},
        model_artifact_key="model_cost",
    )


def _expected_sum(df) -> np.ndarray:
    cols = [df[c].to_numpy().astype(np.float32) for c in INPUTS]
    return np.sum(cols, axis=0)


def test_constructor_rejects_empty_column_map():
    with pytest.raises(ValueError, match="non-empty"):
        OnnxScoring("s", "t", "o", column_map={}, model_artifact_key="m")


def test_scoring_writes_output_column(patch_onnx, sample_data, mock_context):
    layer = _layer()
    out = layer.apply(sample_data, {}, {}, mock_context)
    assert "prediction" in out["risk"].columns
    assert len(out["risk"]["prediction"]) == 4
    assert np.allclose(
        out["risk"]["prediction"].to_numpy(), _expected_sum(out["risk"])
    )


def test_column_map_renames_source_to_required(patch_onnx, sample_data, mock_context):
    """When source data columns differ from the model's input names, column_map
    selects + renames before scoring."""
    sample_data["risk"] = sample_data["risk"].rename(columns={
        "driver_youngest_age": "src_age",
        "vehicle_market_value": "src_value",
    })
    layer = OnnxScoring(
        name="score",
        target_table="risk",
        output_column="prediction",
        column_map={
            "driver_youngest_age": "src_age",
            "vehicle_market_value": "src_value",
        },
        model_artifact_key="model_cost",
    )
    out = layer.apply(sample_data, {}, {}, mock_context)
    expected = (
        out["risk"]["src_age"].to_numpy().astype(np.float32)
        + out["risk"]["src_value"].to_numpy().astype(np.float32)
    )
    assert np.allclose(out["risk"]["prediction"].to_numpy(), expected)


def test_missing_source_columns_raises(patch_onnx, sample_data, mock_context):
    layer = OnnxScoring(
        "s", "risk", "o",
        column_map={"driver_youngest_age": "does_not_exist"},
        model_artifact_key="model_cost",
    )
    with pytest.raises(LayerCalculationError, match="source columns missing"):
        layer.apply(sample_data, {}, {}, mock_context)


def test_getstate_drops_unpicklable_session(patch_onnx, mock_context):
    layer = _layer()
    layer.init_from_context(mock_context)
    assert layer.session is not None
    assert layer.__getstate__()["session"] is None


def test_deepcopy_shares_loaded_session(patch_onnx, mock_context):
    layer = _layer()
    layer.init_from_context(mock_context)
    clone = deepcopy(layer)
    assert clone.session is layer.session
    assert clone.column_map == layer.column_map
    assert clone.column_map is not layer.column_map


def test_from_dict_roundtrip():
    cfg = {
        "type": "OnnxScoring",
        "name": "score",
        "target_table": "risk",
        "output_column": "prediction",
        "column_map": {"a": "src_a", "b": "src_b"},
        "model_artifact_key": "m",
        "dependencies": ["dep1"],
        "description": "d",
    }
    layer = OnnxScoring._from_dict(cfg)
    assert layer.column_map == {"a": "src_a", "b": "src_b"}
    assert layer.model_artifact_key == "m"
    assert layer.model_columns == ["a", "b"]
    out = layer.to_dict()
    assert out["type"] == "OnnxScoring"
    assert out["model_artifact_key"] == "m"
    assert out["column_map"] == {"a": "src_a", "b": "src_b"}


def test_onnx_layer_inside_pipeline(patch_onnx, pipeline, sample_data, mock_context):
    pipeline.add_parameter(_layer())
    out = pipeline.predict(sample_data)
    assert "prediction" in out["risk"].columns
    assert len(out["risk"]) == 4


# -- derived (formula) columns -------------------------------------------------


def _derived_layer(**kw):
    return OnnxScoring(
        name="score",
        target_table="risk",
        output_column="prediction",
        # The model was trained on a CLIPPED age; the data holds the raw one.
        column_map={"age_raw": "driver_youngest_age", "vehicle_market_value": "vehicle_market_value"},
        derived=[{"name": "driver_youngest_age", "expression": "np.clip(age_raw, 25, 60)"}],
        model_artifact_key="model_cost",
        **kw,
    )


def test_derived_formula_feeds_a_model_input(patch_onnx, sample_data, mock_context):
    out = _derived_layer().apply(sample_data, {}, {}, mock_context)["risk"]
    expected = (
        np.clip(out["driver_youngest_age"].to_numpy(), 25, 60).astype(np.float32)
        + out["vehicle_market_value"].to_numpy().astype(np.float32)
    )
    assert np.allclose(out["prediction"].to_numpy(), expected)


def test_the_formula_does_not_write_back_to_the_table(patch_onnx, sample_data, mock_context):
    before = sample_data["risk"]["driver_youngest_age"].copy()
    out = _derived_layer().apply(sample_data, {}, {}, mock_context)["risk"]
    assert out["driver_youngest_age"].equals(before)
    assert "age_raw" not in out.columns


def test_a_formula_can_read_an_earlier_formula(patch_onnx, sample_data, mock_context):
    layer = OnnxScoring(
        name="score", target_table="risk", output_column="prediction",
        column_map={"age_raw": "driver_youngest_age", "v": "vehicle_market_value"},
        derived=[{"name": "driver_youngest_age", "expression": "age_raw * 2"},
                 {"name": "vehicle_market_value", "expression": "v + driver_youngest_age"}],
        model_artifact_key="model_cost",
    )
    out = layer.apply(sample_data, {}, {}, mock_context)["risk"]
    age2 = out["driver_youngest_age"].to_numpy() * 2
    expected = age2.astype(np.float32) + (out["vehicle_market_value"].to_numpy() + age2).astype(np.float32)
    assert np.allclose(out["prediction"].to_numpy(), expected)


def test_a_model_input_neither_mapped_nor_derived_is_named(patch_onnx, sample_data, mock_context):
    layer = OnnxScoring(
        name="score", target_table="risk", output_column="prediction",
        column_map={"age_raw": "driver_youngest_age"},
        derived=[{"name": "driver_youngest_age", "expression": "age_raw"}],
        model_artifact_key="model_cost",
    )
    with pytest.raises(LayerCalculationError, match="vehicle_market_value"):
        layer.apply(sample_data, {}, {}, mock_context)


def test_a_failing_formula_names_itself(patch_onnx, sample_data, mock_context):
    layer = OnnxScoring(
        name="score", target_table="risk", output_column="prediction",
        column_map={"age_raw": "driver_youngest_age", "vehicle_market_value": "vehicle_market_value"},
        derived=[{"name": "driver_youngest_age", "expression": "nope * 2"}],
        model_artifact_key="model_cost",
    )
    with pytest.raises(LayerCalculationError, match="formula column 'driver_youngest_age'"):
        layer.apply(sample_data, {}, {}, mock_context)


@pytest.mark.parametrize("derived", [[{"name": "x"}], [{"expression": "1"}], ["np.log(a)"]])
def test_malformed_derived_entries_are_refused(derived):
    with pytest.raises(ValueError, match="derived entry"):
        OnnxScoring("s", "t", "o", column_map={"a": "a"}, model_artifact_key="m", derived=derived)


def test_a_derived_name_cannot_also_be_mapped():
    with pytest.raises(ValueError, match="also mapped"):
        OnnxScoring("s", "t", "o", column_map={"a": "a", "b": "b"}, model_artifact_key="m",
                    derived=[{"name": "b", "expression": "a * 2"}])


def test_derived_round_trips_and_is_omitted_when_unused():
    plain = _layer()
    assert "derived" not in plain.parameters and "derived" not in plain._layer_params()
    layer = _derived_layer()
    back = OnnxScoring._from_dict(deepcopy(layer.parameters))
    assert back.derived == layer.derived
    assert back.column_map == layer.column_map
    assert back._layer_params() == layer._layer_params()


def test_a_layer_pickled_before_derived_existed_still_scores(patch_onnx, sample_data, mock_context):
    """Pipelines are cloudpickled; one written by an older engine has no `derived`."""
    layer = _layer()
    del layer.__dict__["derived"]
    out = layer.apply(sample_data, {}, {}, mock_context)
    assert np.allclose(out["risk"]["prediction"].to_numpy(), _expected_sum(out["risk"]))
