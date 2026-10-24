"""OnnxScoring — exercised with the onnxruntime seam mocked.

We patch ``_load_model`` so no real ``onnxruntime.InferenceSession`` is built;
:class:`FakeSession` returns the row-wise sum of the fed feature columns, which
is enough to assert the select/rename/feed/output plumbing.
"""

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
