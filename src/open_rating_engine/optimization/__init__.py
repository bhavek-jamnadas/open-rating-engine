"""Price optimisers that run on top of a pricing pipeline. Loaded lazily; see ``.._lazy``."""

from typing import TYPE_CHECKING

from .._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    exports={
        ".individual_price_optimizer": ["PriceOptimizerIndividual"],
        ".rating_factor_price_optimizer": ["PriceOptimizerRatingFactor"],
        ".multi_period_individual_price_optimizer": [
            "PriceOptimizerIndividualMultiPeriod",
            "make_log_multiplier_penalty",
        ],
        ".multi_period_rating_factor_price_optimizer": [
            "PriceOptimizerRatingFactorMultiPeriod",
            "make_log_multiplier_regularizer",
        ],
        ".simulate_adjustment_grid": ["simulate_adjustment_grid", "make_demand_emulation"],
    },
)

if TYPE_CHECKING:
    from .individual_price_optimizer import PriceOptimizerIndividual
    from .multi_period_individual_price_optimizer import (
        PriceOptimizerIndividualMultiPeriod,
        make_log_multiplier_penalty,
    )
    from .multi_period_rating_factor_price_optimizer import (
        PriceOptimizerRatingFactorMultiPeriod,
        make_log_multiplier_regularizer,
    )
    from .rating_factor_price_optimizer import PriceOptimizerRatingFactor
    from .simulate_adjustment_grid import make_demand_emulation, simulate_adjustment_grid
