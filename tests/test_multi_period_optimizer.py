"""Phase 2 tests for ``PriceOptimizerMultiPeriod`` — the aging subsystem.

These exercise the two-basket aging loop in isolation (no pipeline, no H2O, no
scoring): construction/validation, the one-step ``apply_aging`` delegate, and a
**hand-computed multi-year trajectory** for both new-business inflow modes. They
lock in that the carried book ages one step per period while the new-business
cohort is re-injected un-aged (design §9.1).

Imported via the submodule path (not the package ``__init__``) so the file is
agnostic to the package-level export wiring.
"""

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.optimization.multi_period_rating_factor_price_optimizer import (
    PriceOptimizerRatingFactorMultiPeriod,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #

def _src(table, column, expression, *, enabled=True):
    """A SourceColumn-style layer config carrying a one-year aging rule, in the
    shape ``apply_aging`` consumes."""
    return {
        "type": "SourceColumn",
        "name": f"{table}.{column}",
        "source_table": table,
        "source_column": column,
        "dependencies": [],
        "aging_rule": {"expression": expression, "enabled": enabled},
    }


@pytest.fixture
def rn_rules():
    """Full one-step aging rules for the carried renewal book (all on ``risk``):
    demographics age, the vehicle depreciates 10%/yr, NB graduates to RN, the
    policy term rolls forward a year, and ``previous_gwp`` picks up last period's
    ``gwp`` (the feedback hand-off)."""
    return {
        "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
        "risk.risk_customer_tenure": _src("risk", "risk_customer_tenure", "df['risk_customer_tenure'] + 1"),
        "risk.vehicle_market_value": _src("risk", "vehicle_market_value", "df['vehicle_market_value'] * 0.9"),
        "risk.transaction_type": _src("risk", "transaction_type", "'RN'"),
        "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        "risk.policy_term_start_date": _src(
            "risk", "policy_term_start_date", "df['policy_term_start_date'] + pd.DateOffset(years=1)"
        ),
    }


def _risk_row(driver_age, tenure, vmv, tt, gwp, prev_gwp, start="2025-01-01"):
    return pd.DataFrame({
        "driver_youngest_age": [driver_age],
        "risk_customer_tenure": [tenure],
        "vehicle_market_value": [float(vmv)],
        "transaction_type": pd.Series([tt], dtype=object),
        "gwp": [float(gwp)],
        "previous_gwp": [float(prev_gwp)],
        "policy_term_start_date": pd.to_datetime([start]),
    })


@pytest.fixture
def data_nb():
    # A 25-year-old, tenure 0, £20k vehicle, brand new (transaction_type NB).
    return {"risk": _risk_row(25, 0, 20_000, "NB", 600.0, 0.0)}


@pytest.fixture
def data_rn():
    # A 40-year-old renewal, tenure 2, £10k vehicle, last year's premium £480.
    return {"risk": _risk_row(40, 2, 10_000, "RN", 500.0, 480.0)}


def _make(data_nb, data_rn, rn_rules, **kw):
    kw.setdefault("n_periods", 3)
    return PriceOptimizerRatingFactorMultiPeriod(
        data_nb=data_nb, data_rn=data_rn, aging_rules_rn=rn_rules, **kw
    )


# --------------------------------------------------------------------------- #
# Construction & validation
# --------------------------------------------------------------------------- #

def test_init_stores_baskets_and_config(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules, n_periods=4, discount_rate=0.07, timing_offset=0.5)
    assert set(opt.input.data_nb) == {"risk"}
    assert opt.parameters.n_periods == 4
    assert opt.parameters.new_business_mode == "runoff"
    assert opt.parameters.discount_rate == pytest.approx(0.07)
    assert opt.parameters.aging_rules_nb == {}  # None -> empty (NB un-aged)


def test_init_copies_baskets_defensively(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules)
    # Mutating the caller's frame after construction must not touch the stored basket.
    data_rn["risk"].loc[0, "driver_youngest_age"] = 999
    assert opt.input.data_rn["risk"].loc[0, "driver_youngest_age"] == 40


@pytest.mark.parametrize("bad", [[1, 2, 3], None, {}, "frame"])
def test_init_rejects_non_dict_or_empty_basket(bad, data_rn, rn_rules):
    with pytest.raises(TypeError):
        _make(bad, data_rn, rn_rules)


def test_init_rejects_non_dataframe_basket_value(data_rn, rn_rules):
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        _make({"risk": [1, 2, 3]}, data_rn, rn_rules)


def test_init_rejects_all_empty_frames(data_rn, rn_rules):
    with pytest.raises(ValueError, match="no rows"):
        _make({"risk": pd.DataFrame({"a": []})}, data_rn, rn_rules)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_init_rejects_bad_n_periods(bad, data_nb, data_rn, rn_rules):
    with pytest.raises(ValueError, match="n_periods"):
        _make(data_nb, data_rn, rn_rules, n_periods=bad)


def test_init_requires_rn_rules(data_nb, data_rn):
    with pytest.raises(TypeError, match="aging_rules_rn is required"):
        PriceOptimizerRatingFactorMultiPeriod(data_nb=data_nb, data_rn=data_rn, n_periods=3, aging_rules_rn=None)


def test_init_rejects_non_dict_rules(data_nb, data_rn):
    with pytest.raises(TypeError, match="SourceColumn"):
        PriceOptimizerRatingFactorMultiPeriod(
            data_nb=data_nb, data_rn=data_rn, n_periods=3, aging_rules_rn="nope"
        )


def test_init_rejects_unknown_mode(data_nb, data_rn, rn_rules):
    with pytest.raises(ValueError, match="new_business_mode"):
        _make(data_nb, data_rn, rn_rules, new_business_mode="steady")


@pytest.mark.parametrize("bad", ["x", None, True])
def test_init_rejects_non_numeric_discount(bad, data_nb, data_rn, rn_rules):
    with pytest.raises(TypeError, match="discount_rate"):
        _make(data_nb, data_rn, rn_rules, discount_rate=bad)


def test_init_rejects_discount_at_or_below_minus_one(data_nb, data_rn, rn_rules):
    with pytest.raises(ValueError, match="discount_rate"):
        _make(data_nb, data_rn, rn_rules, discount_rate=-1)


# --------------------------------------------------------------------------- #
# _age_one_step — the apply_aging delegate
# --------------------------------------------------------------------------- #

def test_age_one_step_applies_all_rules(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules)
    aged = opt._age_one_step(opt.input.data_rn, rn_rules)["risk"]
    assert aged.loc[0, "driver_youngest_age"] == 41
    assert aged.loc[0, "risk_customer_tenure"] == 3
    assert aged.loc[0, "vehicle_market_value"] == pytest.approx(9_000.0)
    assert aged.loc[0, "transaction_type"] == "RN"
    assert aged.loc[0, "previous_gwp"] == pytest.approx(500.0)  # <- last gwp
    assert aged.loc[0, "policy_term_start_date"] == pd.Timestamp("2026-01-01")


def test_age_one_step_does_not_mutate_input(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules)
    before = opt.input.data_rn["risk"].copy(deep=True)
    opt._age_one_step(opt.input.data_rn, rn_rules)
    pd.testing.assert_frame_equal(opt.input.data_rn["risk"], before)


def test_age_one_step_multi_table():
    opt = PriceOptimizerRatingFactorMultiPeriod(
        data_nb={"driver": pd.DataFrame({"age": [30]})},
        data_rn={"driver": pd.DataFrame({"age": [50]}), "vehicle": pd.DataFrame({"value": [8_000.0]})},
        n_periods=2,
        aging_rules_rn={
            "driver.age": _src("driver", "age", "df['age'] + 1"),
            "vehicle.value": _src("vehicle", "value", "df['value'] * 0.9"),
        },
    )
    aged = opt._age_one_step(opt.input.data_rn, opt.parameters.aging_rules_rn)
    assert aged["driver"].loc[0, "age"] == 51
    assert aged["vehicle"].loc[0, "value"] == pytest.approx(7_200.0)


# --------------------------------------------------------------------------- #
# Two-basket trajectory — runoff (hand-computed)
# --------------------------------------------------------------------------- #

def test_runoff_trajectory_hand_computed(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules, n_periods=3, new_business_mode="runoff")
    baskets = opt._age_book_trajectory()
    assert len(baskets) == 3

    # Runoff: no new business after period 0, so the book stays at 2 rows
    # (the period-0 NB + the seed renewal), each aged one further step per period.
    for b in baskets:
        assert len(b["risk"]) == 2
    risk = [b["risk"] for b in baskets]

    # Row 0 is the period-0 NB cohort; row 1 is the seed renewal. Order is
    # preserved by the loop (NB concatenated first, then the carry ages in place).
    # driver_youngest_age: NB 25->26->27, RN 40->41->42.
    assert risk[0]["driver_youngest_age"].tolist() == [25, 40]
    assert risk[1]["driver_youngest_age"].tolist() == [26, 41]
    assert risk[2]["driver_youngest_age"].tolist() == [27, 42]

    # Tenure: NB 0->1->2, RN 2->3->4.
    assert risk[2]["risk_customer_tenure"].tolist() == [2, 4]

    # Vehicle depreciates 10%/yr by *applying the rule once per period* (no
    # ``** periods`` term): NB 20000->18000->16200, RN 10000->9000->8100.
    assert risk[1]["vehicle_market_value"].tolist() == pytest.approx([18_000.0, 9_000.0])
    assert risk[2]["vehicle_market_value"].tolist() == pytest.approx([16_200.0, 8_100.0])

    # The period-0 NB graduates to RN after the first aging step; everything is RN thereafter.
    assert risk[0]["transaction_type"].tolist() == ["NB", "RN"]
    assert risk[1]["transaction_type"].tolist() == ["RN", "RN"]
    assert risk[2]["transaction_type"].tolist() == ["RN", "RN"]

    # previous_gwp picks up gwp once aged (feedback hand-off); period 0 is untouched.
    assert risk[0]["previous_gwp"].tolist() == [0.0, 480.0]
    assert (risk[1]["previous_gwp"] == risk[1]["gwp"]).all()

    # Policy term rolls forward one calendar year per period.
    assert (risk[2]["policy_term_start_date"] == pd.Timestamp("2027-01-01")).all()


def test_runoff_single_period_is_seed_book_unaged(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules, n_periods=1)
    baskets = opt._age_book_trajectory()
    assert len(baskets) == 1
    # No aging applied: the seed NB + RN, untouched.
    assert baskets[0]["risk"]["driver_youngest_age"].tolist() == [25, 40]
    assert baskets[0]["risk"]["transaction_type"].tolist() == ["NB", "RN"]


# --------------------------------------------------------------------------- #
# Two-basket trajectory — going_concern (NB re-injected un-aged)
# --------------------------------------------------------------------------- #

def test_going_concern_reinjects_unaged_new_business(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules, n_periods=3, new_business_mode="going_concern")
    baskets = opt._age_book_trajectory()

    # The book grows by one fresh NB policy each period: 2, 3, 4 rows.
    assert [len(b["risk"]) for b in baskets] == [2, 3, 4]

    for t, b in enumerate(baskets):
        risk = b["risk"]
        nb_rows = risk[risk["transaction_type"] == "NB"]
        # Exactly one un-aged NB cohort present every period...
        assert len(nb_rows) == 1
        # ...and it is identical to the original draw — never aged.
        assert nb_rows["driver_youngest_age"].tolist() == [25]
        assert nb_rows["risk_customer_tenure"].tolist() == [0]
        assert nb_rows["vehicle_market_value"].tolist() == pytest.approx([20_000.0])
        assert (nb_rows["policy_term_start_date"] == pd.Timestamp("2025-01-01")).all()

    # The oldest renewal (the original seed) keeps aging one step per period.
    assert baskets[2]["risk"]["driver_youngest_age"].max() == 42


# --------------------------------------------------------------------------- #
# _fresh_nb — un-aged by default, optional calendar shift
# --------------------------------------------------------------------------- #

def test_fresh_nb_is_unaged_every_period_without_nb_rules(data_nb, data_rn, rn_rules):
    opt = _make(data_nb, data_rn, rn_rules)
    for period in (0, 1, 5):
        nb = opt._fresh_nb(period)["risk"]
        assert nb["driver_youngest_age"].tolist() == [25]
        assert nb["policy_term_start_date"].tolist() == [pd.Timestamp("2025-01-01")]


def test_fresh_nb_date_shift_only_leaves_demographics_unaged(data_nb, data_rn, rn_rules):
    # NB rules carry a calendar shift but NO demographic aging (design §9.1).
    nb_rules = {
        "risk.policy_term_start_date": _src(
            "risk", "policy_term_start_date", "df['policy_term_start_date'] + pd.DateOffset(years=1)"
        ),
    }
    opt = _make(data_nb, data_rn, rn_rules, aging_rules_nb=nb_rules)
    nb2 = opt._fresh_nb(2)["risk"]
    # Term shifted forward 2 years (applied once per period)...
    assert nb2["policy_term_start_date"].tolist() == [pd.Timestamp("2027-01-01")]
    # ...but the driver is still 25 — new business is a fresh draw, not aged.
    assert nb2["driver_youngest_age"].tolist() == [25]
