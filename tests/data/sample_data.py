"""Small, deterministic synthetic datasets for the test suite.

These intentionally mirror the shape of the data used in the rating engine's
``_example_usage.py`` (``policy`` / ``risk`` / ``cover`` / ``claims_history``)
but are tiny so tests stay fast and assertions are exact.
"""

import numpy as np
import pandas as pd


def make_policy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_number": [1, 2, 3, 4],
            "transaction_type": ["NB", "RN", "NB", "RN"],
            "customer_tenure": [0, 7, 12, 3],
            "base_premium": [100.0, 200.0, 300.0, 400.0],
        }
    )


def make_risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_number": [1, 2, 3, 4],
            "risk_number": [10, 20, 30, 40],
            "driver_youngest_age": [19, 28, 41, 67],
            "vehicle_market_value": [15_000.0, 32_000.0, 8_000.0, 55_000.0],
            "risk_usage": ["Private", "Rideshare", "Private", "Private"],
        }
    )


def make_cover() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_number": [10, 20, 30, 40],
            "cover_type": ["comprehensive", "third_party", "comprehensive", "comprehensive"],
        }
    )


def make_claims_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_number": [1, 2, 3, 4],
            "claim_count": [0, 1, 0, 2],
            "claim_cost": [0.0, 1500.0, 0.0, 9000.0],
        }
    )


def make_data() -> dict:
    """The full ``data`` dict consumed by ``PyfuncPricingPipeline.predict``."""
    return {
        "policy": make_policy(),
        "risk": make_risk(),
        "cover": make_cover(),
        "claims_history": make_claims_history(),
    }


def make_rating_tables() -> dict:
    """Rating tables keyed by columns present on the ``risk`` table.

    ``RatingTablesLayer`` requires object/categorical key columns and a
    numeric ``value`` column.
    """
    return {
        "risk_usage_mult": pd.DataFrame(
            {
                "risk_usage": ["Private", "Rideshare"],
                "value": [1.0, 1.3],
            }
        ),
        "cover_type_mult": pd.DataFrame(
            {
                "cover_type": ["comprehensive", "third_party"],
                "value": [1.1, 0.8],
            }
        ),
        "cover_type_add": pd.DataFrame(
            {
                "cover_type": ["comprehensive", "third_party"],
                "value": [0.05, 0.0],
            }
        ),
    }
