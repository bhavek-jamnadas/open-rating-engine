"""H2OScoringParameter — exercised with the JVM/MOJO seams mocked.

We patch ``_load_model`` (JPype + hex genmodel) and ``_mojo_predict_pandas``
(per-row wrapper scoring) so no JVM is ever started.
"""

from copy import deepcopy

import pandas as pd
import pytest

import numpy as np

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import (
    H2OScoringParameter,
    LayerCalculationError,
    _row_to_h2o_payload,
)
from tests.mocks.h2o_mock import fake_load_model, make_fake_mojo_predict_pandas


@pytest.fixture
def patch_h2o(monkeypatch):
    monkeypatch.setattr(H2OScoringParameter, "_load_model",
                         staticmethod(fake_load_model))

    def _apply(model_type):
        monkeypatch.setattr(
            H2OScoringParameter, "_mojo_predict_pandas",
            staticmethod(make_fake_mojo_predict_pandas(model_type)),
        )

    return _apply


def _layer(model_type):
    return H2OScoringParameter(
        name="score",
        target_table="risk",
        output_column="prediction",
        column_map={
            "driver_youngest_age": "driver_youngest_age",
            "vehicle_market_value": "vehicle_market_value",
        },
        mojo_artifact_key="model_cost",
        genmodel_artifact_key="h2o_gen_model",
        model_type=model_type,
    )


def test_constructor_rejects_empty_column_map():
    with pytest.raises(ValueError, match="non-empty"):
        H2OScoringParameter("s", "t", "o", column_map={},
                            mojo_artifact_key="m", genmodel_artifact_key="g")


def test_regression_scoring(patch_h2o, sample_data, mock_context):
    patch_h2o("regression")
    layer = _layer("regression")
    out = layer.apply(sample_data, {}, {}, mock_context)
    assert out["risk"]["prediction"].iloc[0] == pytest.approx(100.0)
    assert len(out["risk"]["prediction"]) == 4


def test_binomial_scoring_returns_p1(patch_h2o, sample_data, mock_context):
    patch_h2o("binomial")
    out = _layer("binomial").apply(sample_data, {}, {}, mock_context)
    assert out["risk"]["prediction"].iloc[0] == pytest.approx(0.1)
    assert out["risk"]["prediction"].iloc[-1] == pytest.approx(0.9)


def test_multinomial_scoring_returns_last_column(patch_h2o, sample_data, mock_context):
    patch_h2o("multinomial")
    out = _layer("multinomial").apply(sample_data, {}, {}, mock_context)
    assert (out["risk"]["prediction"] == 0).all()


def test_missing_source_columns_raises(patch_h2o, sample_data, mock_context):
    patch_h2o("regression")
    layer = H2OScoringParameter(
        "s", "risk", "o",
        column_map={"does_not_exist": "does_not_exist"},
        mojo_artifact_key="model_cost", genmodel_artifact_key="h2o_gen_model",
        model_type="regression",
    )
    with pytest.raises(LayerCalculationError, match="source columns missing"):
        layer.apply(sample_data, {}, {}, mock_context)


def test_column_map_renames_source_to_required(patch_h2o, sample_data, mock_context):
    """When source data columns differ from the MOJO's required names,
    column_map should select+rename before scoring."""
    patch_h2o("regression")
    sample_data["risk"] = sample_data["risk"].rename(columns={
        "driver_youngest_age": "src_age",
        "vehicle_market_value": "src_value",
    })
    layer = H2OScoringParameter(
        name="score",
        target_table="risk",
        output_column="prediction",
        column_map={
            "driver_youngest_age": "src_age",
            "vehicle_market_value": "src_value",
        },
        mojo_artifact_key="model_cost",
        genmodel_artifact_key="h2o_gen_model",
        model_type="regression",
    )
    out = layer.apply(sample_data, {}, {}, mock_context)
    assert out["risk"]["prediction"].iloc[0] == pytest.approx(100.0)
    assert len(out["risk"]["prediction"]) == 4


def test_legacy_dict_with_model_columns_is_accepted():
    """_from_dict should accept legacy configs that only have model_columns
    by synthesising an identity column_map."""
    layer = H2OScoringParameter._from_dict({
        "name": "score",
        "target_table": "risk",
        "output_column": "prediction",
        "model_columns": ["a", "b"],
        "mojo_artifact_key": "m",
        "genmodel_artifact_key": "g",
    })
    assert layer.column_map == {"a": "a", "b": "b"}
    assert layer.model_columns == ["a", "b"]


def test_getstate_drops_unpicklable_wrapper(patch_h2o, mock_context):
    layer = _layer("regression")
    layer.init_from_context(mock_context)
    assert layer.wrapper is not None
    assert layer.__getstate__()["wrapper"] is None


def test_deepcopy_shares_loaded_wrapper(patch_h2o, mock_context):
    layer = _layer("regression")
    layer.init_from_context(mock_context)
    clone = deepcopy(layer)
    assert clone.wrapper is layer.wrapper
    assert clone.column_map == layer.column_map
    assert clone.column_map is not layer.column_map


def test_h2o_layer_inside_pipeline(patch_h2o, pipeline, sample_data, mock_context):
    patch_h2o("regression")
    pipeline.add_parameter(_layer("regression"))
    out = pipeline.predict(sample_data)
    assert "prediction" in out["risk"].columns
    assert len(out["risk"]) == 4


def test_row_to_h2o_payload_omits_missing():
    """Missing values must be dropped, not forwarded as Java null — a null in
    RowData triggers a NullPointerException inside genmodel."""
    row = pd.Series({"age": 58, "prev_gwp": np.nan, "make": "BMW", "tenure": None})
    payload = _row_to_h2o_payload(row)
    assert payload == {"age": "58", "make": "BMW"}
    assert "prev_gwp" not in payload and "tenure" not in payload


def _group_levels_layer(mapping):
    return H2OScoringParameter(
        name="score", target_table="risk", output_column="prediction",
        column_map={"x": "x"},
        mojo_artifact_key="m", genmodel_artifact_key="g",
        preprocessing=[{"type": "group_levels", "column": "x", "mapping": mapping}],
    )


def test_apply_preprocessing_group_levels_remaps_numeric_with_missing():
    """A count column with a null loads as float64; values must canonicalise
    ('2.0' → '2'), bare '0' must survive, grouped levels collapse, missing → NA —
    matching fit-time apply_level_combining so the MOJO sees the trained labels."""
    layer = _group_levels_layer({"1": "1-2", "2": "1-2"})
    out = layer._apply_preprocessing(pd.DataFrame({"x": [0, 1, 2, None]}))
    vals = out["x"].tolist()
    assert vals[:3] == ["0", "1-2", "1-2"]   # passthrough '0', grouped 1/2
    assert vals[3] is None or pd.isna(vals[3])  # genuine missing → NA


def test_apply_preprocessing_group_levels_string_input_passes_unmapped_through():
    layer = _group_levels_layer({"250": "grp", "500": "grp"})
    out = layer._apply_preprocessing(pd.DataFrame({"x": ["250", "500", "750", None]}))
    vals = out["x"].tolist()
    assert vals[:3] == ["grp", "grp", "750"]  # 750 unmapped → canonical passthrough
    assert vals[3] is None or pd.isna(vals[3])
