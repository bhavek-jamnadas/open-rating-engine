"""simulate_adjustment_grid / make_demand_emulation — on a tiny in-memory pipeline.

The pipeline is a single-table ``RatingTablesLayer`` feeding ``gwp = base_premium *
demand_multiplier``, so the expected grid is known in closed form.
"""

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.rating_engine import PyfuncPricingPipeline
from open_rating_engine.rating_engine.pricing_layers import ColumnCalculation, RatingTablesLayer
from open_rating_engine.optimization.simulate_adjustment_grid import (
    make_demand_emulation,
    simulate_adjustment_grid,
)

FACTORS = np.array([0.9, 1.0, 1.1])


@pytest.fixture
def demand_pipeline(pipeline):
    pipeline.add_parameter(RatingTablesLayer(
        name="demand_multiplier",
        target_table="risk",
        output_column="demand_multiplier",
        rating_tables={"band": pd.DataFrame({"band": ["a", "b"], "value": [1.0, 1.0]})},
        default_values={"band": 1.0},
    ))
    pipeline.add_parameter(ColumnCalculation(
        name="gwp",
        target_table="risk",
        output_column="gwp",
        calculation="df['base_premium'] * df['demand_multiplier']",
        dependencies=["demand_multiplier"],
    ))
    return pipeline


@pytest.fixture
def book():
    return {"risk": pd.DataFrame({"band": ["a", "b", "c"], "base_premium": [100.0, 200.0, 300.0]})}


def test_demand_emulation_returns_input_and_mutates_layer(demand_pipeline, book):
    emulation = make_demand_emulation("demand_multiplier", "band", mode="override")

    returned = emulation(book, demand_pipeline, 1.2)

    assert returned is book
    assert "gwp" not in returned["risk"].columns
    layer = demand_pipeline.layers["demand_multiplier"]
    assert (layer.rating_tables["band"]["value"] == 1.2).all()
    assert layer.default_values["band"] == 1.2


def test_grid_scores_each_factor_once(demand_pipeline, book, monkeypatch):
    calls = []
    original_predict = PyfuncPricingPipeline.predict

    def counting_predict(self, model_input):
        calls.append(1)
        return original_predict(self, model_input)

    monkeypatch.setattr(PyfuncPricingPipeline, "predict", counting_predict)

    simulate_adjustment_grid(
        data=book,
        pipeline=demand_pipeline,
        premium_emulation=make_demand_emulation("demand_multiplier", "band", mode="override"),
        output_table="risk",
        output_columns={"gwp": "gwp"},
        adjustment_factor=FACTORS,
    )

    assert len(calls) == len(FACTORS)


def test_grid_values(demand_pipeline, book):
    result = simulate_adjustment_grid(
        data=book,
        pipeline=demand_pipeline,
        premium_emulation=make_demand_emulation("demand_multiplier", "band", mode="override"),
        output_table="risk",
        output_columns={"gwp": "gwp"},
        adjustment_factor=FACTORS,
        parallel_n_jobs=2,
    )

    expected = np.outer(book["risk"]["base_premium"], FACTORS)
    np.testing.assert_allclose(result["gwp"], expected)
    # The baseline pipeline is untouched: every factor ran on its own copy.
    assert (demand_pipeline.layers["demand_multiplier"].rating_tables["band"]["value"] == 1.0).all()
