"""Tests for optimization pure helpers.

The pre-split unified ``PriceOptimizer`` (``method="Individual"/"RatingFactor"``)
and its ``_calculate_multiplier_for_target`` helper lived in the now-deleted
``open_rating_engine.optimization._old.single_period_price_optimizer`` and have been
replaced by the dedicated ``PriceOptimizerIndividual`` /
``PriceOptimizerRatingFactor`` classes (covered by ``test_individual_*`` /
``test_rating_factor_optimizer``). The only survivor exercised here is the
``_fast_linear_interpolator`` helper, which now lives in
``rating_factor_price_optimizer``.
"""

import numpy as np
import pytest

from open_rating_engine.optimization.rating_factor_price_optimizer import (
    _fast_linear_interpolator,
)


# --------------------------------------------------------------------------- #
# Pure numeric helpers
# --------------------------------------------------------------------------- #

def test_fast_linear_interpolator_matches_numpy():
    price_points = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    conversion = np.array([[1.0, 0.5, 0.0], [0.8, 0.4, 0.2]])
    targets = np.array([0.5, 1.5])
    out = _fast_linear_interpolator(price_points, conversion, targets)
    assert out[0] == pytest.approx(0.75)   # midway 1.0 -> 0.5
    assert out[1] == pytest.approx(0.30)   # midway 0.4 -> 0.2
