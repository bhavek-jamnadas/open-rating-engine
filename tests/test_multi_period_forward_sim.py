"""Phase 3 tests for ``PriceOptimizerMultiPeriod`` — the forward simulation.

These build a small **no-H2O** pipeline (pure ``ColumnCalculation`` layers + the
demand ``RatingTablesLayer``) so the whole forward sim runs offline. The
centrepiece is the **single-period reduction** (design §16.1, R1): with
``n_periods == 1`` and ``discount_rate == 0`` the multi-period optimizer must
reproduce :class:`PriceOptimizerRatingFactor`'s optimum on the same pipeline.
The rest lock the forward-sim invariants of §16.2 (survivorship monotonicity,
NPV = undiscounted sum at ``r == 0``, ``previous_gwp_t == gwp_{t-1}``).

Imported via submodule paths so the file is agnostic to package export wiring.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")  # PriceOptimizerRatingFactor JIT-compiles its interpolator

from mlflow.pyfunc import PythonModelContext

from open_rating_engine import (
    ColumnCalculation,
    PyfuncPricingPipeline,
    RatingTablesLayer,
    TableAggregation,
    TableJoin,
)
from open_rating_engine.optimization.multi_period_rating_factor_price_optimizer import (
    PriceOptimizerRatingFactorMultiPeriod,
    make_log_multiplier_regularizer,
)
from open_rating_engine.optimization.rating_factor_price_optimizer import (
    PriceOptimizerRatingFactor,
)
from open_rating_engine.optimization.simulate_adjustment_grid import (
    make_demand_emulation,
    simulate_adjustment_grid,
)


# --------------------------------------------------------------------------- #
# A small no-H2O pricing pipeline
# --------------------------------------------------------------------------- #
#
# risk.demand_multiplier  = age-band rating table (the overlay being optimized)
# risk.gwp                = base_premium * demand_multiplier
# risk.price_change       = (gwp - previous_gwp) / previous_gwp
# risk.conversion_rate    = logistic, DECREASING in price_change (real trade-off)
# risk.offered_profit     = gwp - technical_cost
# risk.expected_profit    = offered_profit * conversion_rate
#
# A single demand table with identity (1.0) initial values makes the
# single-period ``override`` grid exact: each row's metric depends only on its
# own multiplier, so interpolating the uniform-factor grid at the per-category
# multiplier == scoring that row at that multiplier.

_ALL_BANDS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def _context():
    return PythonModelContext(artifacts={}, model_config={"environment": "local"})


def _build_pipeline(table_bands, *, derive_bands):
    pipe = PyfuncPricingPipeline(_context())
    pipe.options["show_progress"] = False

    demand = RatingTablesLayer(
        name="demand_multiplier",
        target_table="risk",
        output_column="demand_multiplier",
        rating_tables={
            "age_mult": pd.DataFrame({
                "risk_age_band": pd.Series(table_bands, dtype=object),
                "value": [1.0] * len(table_bands),
            })
        },
        default_values={"age_mult": 1.0},
        dependencies=(["risk_age_band"] if derive_bands else []),
    )
    if derive_bands:
        pipe.add_parameter(ColumnCalculation(
            name="risk_age_band", target_table="risk", output_column="risk_age_band",
            calculation=(
                "pd.cut(df['driver_youngest_age'], bins=list(range(15, 66, 10)) + [np.inf], "
                "labels=['15-24','25-34','35-44','45-54','55-64','65+']).astype('object')"
            ),
            dependencies=[],
        ))
    pipe.add_parameter(demand)
    return _add_economics(pipe)


def _add_economics(pipe):
    """Add the shared gwp → price_change → (logistic) conversion → profit chain.
    Conversion decreases in price, so there's a real conversion/profit trade-off."""
    pipe.add_parameter(ColumnCalculation(
        name="gwp", target_table="risk", output_column="gwp",
        calculation="df['base_premium'] * df['demand_multiplier']",
        dependencies=["demand_multiplier"],
    ))
    pipe.add_parameter(ColumnCalculation(
        name="price_change", target_table="risk", output_column="price_change",
        calculation="np.where(df['previous_gwp'] != 0, (df['gwp'] - df['previous_gwp']) / df['previous_gwp'], 0.0)",
        dependencies=["gwp"],
    ))
    pipe.add_parameter(ColumnCalculation(
        name="conversion_rate", target_table="risk", output_column="conversion_rate",
        calculation="1.0 / (1.0 + np.exp(3.0 * df['price_change']))",
        dependencies=["price_change"],
    ))
    pipe.add_parameter(ColumnCalculation(
        name="offered_profit", target_table="risk", output_column="offered_profit",
        calculation="df['gwp'] - df['technical_cost']",
        dependencies=["gwp"],
    ))
    pipe.add_parameter(ColumnCalculation(
        name="expected_profit", target_table="risk", output_column="expected_profit",
        calculation="df['offered_profit'] * df['conversion_rate']",
        dependencies=["offered_profit", "conversion_rate"],
    ))
    return pipe


def _build_vintaged_pipeline():
    """Pipeline whose demand overlay is *vintaged* — two rating tables, one on
    ``risk_age_band`` and one on ``transaction_type`` (NB vs RN) — so the
    optimizer can price new business and renewals with distinct multipliers
    (design §8.3). The product of the two tables is the demand multiplier."""
    pipe = PyfuncPricingPipeline(_context())
    pipe.options["show_progress"] = False
    pipe.add_parameter(ColumnCalculation(
        name="risk_age_band", target_table="risk", output_column="risk_age_band",
        calculation=(
            "pd.cut(df['driver_youngest_age'], bins=list(range(15, 66, 10)) + [np.inf], "
            "labels=['15-24','25-34','35-44','45-54','55-64','65+']).astype('object')"
        ),
        dependencies=[],
    ))
    pipe.add_parameter(RatingTablesLayer(
        name="demand_multiplier", target_table="risk", output_column="demand_multiplier",
        rating_tables={
            "age_mult": pd.DataFrame({
                "risk_age_band": pd.Series(_ALL_BANDS, dtype=object), "value": [1.0] * len(_ALL_BANDS),
            }),
            "tt_mult": pd.DataFrame({
                "transaction_type": pd.Series(["NB", "RN"], dtype=object), "value": [1.0, 1.0],
            }),
        },
        default_values={"age_mult": 1.0, "tt_mult": 1.0},
        dependencies=["risk_age_band"],
    ))
    return _add_economics(pipe)


def _src(table, column, expression):
    return {
        "type": "SourceColumn",
        "name": f"{table}.{column}",
        "source_table": table,
        "source_column": column,
        "dependencies": [],
        "aging_rule": {"expression": expression, "enabled": True},
    }


_MP_COLS = dict(
    conversion_col="conversion_rate",
    profit_col="expected_profit",
    gwp_col="gwp",
    cost_col="technical_cost",
    sample_weight_col="sample_weights",
)


# --------------------------------------------------------------------------- #
# Single-period reduction (R1) — the keystone correctness test
# --------------------------------------------------------------------------- #

def test_single_period_reduces_to_rating_factor_optimum():
    bands = ["15-24", "25-34", "35-44", "45-54"]
    pipe = _build_pipeline(bands, derive_bands=False)

    nb = {"risk": pd.DataFrame({
        "risk_age_band": pd.Series(["15-24", "25-34"], dtype=object),
        "base_premium": [450.0, 520.0],
        "technical_cost": [280.0, 310.0],
        "previous_gwp": [440.0, 500.0],
    })}
    rn = {"risk": pd.DataFrame({
        "risk_age_band": pd.Series(["25-34", "35-44", "45-54", "45-54"], dtype=object),
        "base_premium": [510.0, 480.0, 600.0, 590.0],
        "technical_cost": [300.0, 350.0, 300.0, 320.0],
        "previous_gwp": [495.0, 470.0, 580.0, 575.0],
    })}
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}

    # --- Single-period: precompute the grid on the same pipeline, then optimize.
    grid = simulate_adjustment_grid(
        data=combined,
        pipeline=pipe,
        premium_emulation=make_demand_emulation("demand_multiplier", "age_mult", mode="override"),
        output_table="risk",
        output_columns={"objective_array": "expected_profit", "constraint_array": "conversion_rate"},
        adjustment_factor=np.arange(0.85, 1.1501, 0.01),  # fine grid, spans the (0.9,1.1) bounds
    )
    sp = PriceOptimizerRatingFactor.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier", scored_data=pipe.predict(combined),
        adjustment_factors=grid["adjustment_factor_array"],
        objective_array=grid["objective_array"], constraint_array=grid["constraint_array"],
    )

    # --- Multi-period: n_periods=1, r=0, total profit (sum), same pipeline + book.
    mp = PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=1,
        aging_rules_rn={"risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']")},
        discount_rate=0.0, objective_type="total_expected_profit", **_MP_COLS,
    )

    # x0 = ones is feasible for both at this target (the grid has an exact 1.0 node).
    target = float(mp._constraint_value(mp.input.initial_flat))

    sp_res = sp.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP")
    mp_res = mp.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP")

    assert sp_res["result"].success and mp_res["result"].success
    # Both hit the conversion target...
    assert abs(sp_res["summary"]["constraint_gap"].iloc[0]) < 1e-4
    assert abs(mp_res["summary"]["constraint_gap"].iloc[0]) < 1e-4

    # ...and recover the SAME per-category overlay (within interpolation error).
    sp_m = sp_res["optimal_multipliers"]["age_mult"]
    mp_m = mp_res["optimal_multipliers"]["age_mult"]
    for cat in bands:
        assert mp_m[cat] == pytest.approx(sp_m[cat], abs=0.03), cat

    # Guard against a vacuous match: the constrained optimum genuinely
    # redistributes the overlay across bands (it is not the trivial all-ones),
    # so agreement above is a real test of the forward sim vs. the grid.
    assert max(sp_m.values()) - min(sp_m.values()) > 0.05


# --------------------------------------------------------------------------- #
# Forward-sim invariants (§16.2)
# --------------------------------------------------------------------------- #

def _invariant_optimizer(n_periods, *, discount_rate=0.0):
    pipe = _build_pipeline(_ALL_BANDS, derive_bands=True)
    nb = {"risk": pd.DataFrame({
        "driver_youngest_age": [24],
        "base_premium": [450.0],
        "technical_cost": [280.0],
        "previous_gwp": [440.0],
    })}
    rn = {"risk": pd.DataFrame({
        "driver_youngest_age": [34, 44],
        "base_premium": [520.0, 600.0],
        "technical_cost": [310.0, 330.0],
        "previous_gwp": [500.0, 580.0],
    })}
    return PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        aging_rules_rn={
            "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        },
        discount_rate=discount_rate, **_MP_COLS,
    )


def test_npv_equals_undiscounted_sum_at_zero_rate():
    opt = _invariant_optimizer(n_periods=3, discount_rate=0.0)
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    w = agg.df["sample_weights"].to_numpy()
    profit = agg.df["expected_profit"].to_numpy()
    assert agg.npv == pytest.approx(float((w * profit).sum()))


def test_discounting_matches_explicit_factor_and_reduces_npv():
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.10)
    x = opt.input.initial_flat
    agg = opt._simulate_and_aggregate(x)
    df = agg.df
    manual = float((
        df["sample_weights"] * df["expected_profit"]
        / (1.10 ** (df["__period"] + 0.5))
    ).sum())
    assert agg.npv == pytest.approx(manual)
    # Discounting strictly shrinks a positive-profit horizon vs r=0.
    undiscounted = _invariant_optimizer(n_periods=2, discount_rate=0.0)._simulate_and_aggregate(x).npv
    assert agg.npv < undiscounted


def test_survivorship_weight_is_non_increasing_per_period():
    opt = _invariant_optimizer(n_periods=3, discount_rate=0.0)
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    pp = opt._per_period_summary(agg)
    # Runoff: the book's total weight is the surviving mass, which only shrinks
    # (each period scaled by conversion <= 1, no new business injected).
    assert (pp["weight"].diff().dropna() <= 1e-9).all()
    # Conversions are genuine probabilities, so the mechanism is sound.
    assert agg.df["conversion_rate"].between(0.0, 1.0).all()


def test_previous_gwp_picks_up_last_period_gwp():
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.0)
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    df = agg.df
    p0 = df[df["__period"] == 0].reset_index(drop=True)
    p1 = df[df["__period"] == 1].reset_index(drop=True)
    # Runoff preserves row order across periods, so the feedback hand-off is
    # row-aligned: this period's previous_gwp is exactly last period's gwp.
    assert np.allclose(p1["previous_gwp"].to_numpy(), p0["gwp"].to_numpy())


# --------------------------------------------------------------------------- #
# Overlay injection + optimize surface
# --------------------------------------------------------------------------- #

def test_apply_overlay_writes_multipliers_into_rating_table():
    opt = _invariant_optimizer(n_periods=1)
    # Set a recognisable multiplier on the first category, ones elsewhere.
    cats = opt.input.category_orders["age_mult"]
    x = np.ones(opt.input.param_total)
    x[opt.input.param_offsets["age_mult"]] = 1.07
    piped = opt._apply_overlay(x, deepcopy(opt.pipeline))
    values = piped.layers["demand_multiplier"].rating_tables["age_mult"]["value"].to_numpy()
    expected = np.array([1.07 if i == 0 else 1.0 for i in range(len(cats))])
    assert np.allclose(values, expected)
    # The shared pipeline on the optimizer is untouched (overlay hits the copy).
    assert np.allclose(
        opt.pipeline.layers["demand_multiplier"].rating_tables["age_mult"]["value"].to_numpy(), 1.0
    )


def test_optimize_multi_period_meets_constraint_and_reports_per_period():
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    target = float(opt._constraint_value(opt.input.initial_flat))
    res = opt.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP")

    assert res["result"].success
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3
    for v in res["optimal_multipliers"]["age_mult"].values():
        assert 0.9 - 1e-9 <= v <= 1.1 + 1e-9
    # Per-period breakdown is saved (one row per simulated period).
    assert list(opt.output.per_period_summary["period"]) == [0, 1]
    assert res["summary"]["objective_type"].iloc[0] == "total_expected_profit"


def test_optimize_requires_from_pipeline():
    bare = PriceOptimizerRatingFactorMultiPeriod(
        data_nb={"risk": pd.DataFrame({"a": [1.0]})},
        data_rn={"risk": pd.DataFrame({"a": [1.0]})},
        n_periods=1,
        aging_rules_rn={"risk.a": _src("risk", "a", "df['a']")},
    )
    with pytest.raises(RuntimeError, match="from_pipeline"):
        bare.optimize(target_constraint=0.5)
    with pytest.raises(RuntimeError, match="from_pipeline"):
        bare._simulate_horizon(np.array([1.0]))


# --------------------------------------------------------------------------- #
# Phase 4 — constraints & penalties
# --------------------------------------------------------------------------- #
#
# New business sits in band "15-24"; the carried renewal book in "35-44"/"45-54"
# (disjoint), so the shared overlay can move renewal categories independently of
# the new-business category — which is what lets the ENBP ceiling bite while the
# lifetime conversion target is still met.

def _constraint_optimizer(n_periods=1, **kw):
    bands = ["15-24", "35-44", "45-54"]
    pipe = _build_pipeline(bands, derive_bands=False)
    nb = {"risk": pd.DataFrame({
        "risk_age_band": pd.Series(["15-24"], dtype=object),
        "base_premium": [500.0], "technical_cost": [300.0], "previous_gwp": [490.0],
    })}
    rn = {"risk": pd.DataFrame({
        "risk_age_band": pd.Series(["35-44", "45-54"], dtype=object),
        "base_premium": [520.0, 600.0], "technical_cost": [300.0, 320.0],
        "previous_gwp": [500.0, 580.0],
    })}
    return PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        aging_rules_rn={
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
            "risk.base_premium": _src("risk", "base_premium", "df['base_premium']"),
        },
        **_MP_COLS, **kw,
    )


def test_enbp_ceiling_caps_renewal_overlay():
    opt = _constraint_optimizer(n_periods=1, discount_rate=0.0)
    target = float(opt._constraint_value(opt.input.initial_flat))
    # Ceiling below the natural 1.0 renewal overlay, so it must bind.
    res = opt.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP", enbp_ceiling=0.98)

    assert res["result"].success
    pp = opt.output.per_period_summary
    # Renewal pricing is held at/under the ENBP ceiling every period (§12)...
    assert (pp["renewal_price_multiplier"] <= 0.98 + 1e-3).all()
    # ...and the ceiling genuinely pulled renewals below the un-constrained 1.0.
    assert pp["renewal_price_multiplier"].iloc[0] < 0.99
    # The lifetime conversion target is still met (the NB band absorbs the slack).
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3


def _period_optimizer(n_periods=2, **kw):
    """Two-period optimizer whose aging moves each policy into a *disjoint* band
    set per period (period 0: {15-24, 35-44}; period 1: {25-34, 45-54}). With the
    shared overlay's categories thus split across periods, per-period conversion
    is driven by its own multipliers — so a per-period constraint is independent
    of the lifetime constraint and SLSQP stays well-conditioned.

    Ages are chosen to *cross* a band edge on aging (``pd.cut`` is right-closed:
    bins ``(15,25],(25,35],…``), so 25→26 moves 15-24→25-34 and 45→46 moves
    35-44→45-54. Every one of the four overlay categories is then active in
    exactly one period — no empty (flat) decision variables to degenerate on."""
    pipe = _build_pipeline(["15-24", "25-34", "35-44", "45-54"], derive_bands=True)
    nb = {"risk": pd.DataFrame({
        "driver_youngest_age": [25], "base_premium": [500.0],
        "technical_cost": [300.0], "previous_gwp": [490.0],
    })}
    rn = {"risk": pd.DataFrame({
        "driver_youngest_age": [45], "base_premium": [600.0],
        "technical_cost": [330.0], "previous_gwp": [580.0],
    })}
    return PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        aging_rules_rn={
            "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        },
        **_MP_COLS, **kw,
    )


def test_per_period_conversion_floor_is_respected():
    opt = _period_optimizer(n_periods=2, discount_rate=0.0)
    natural_p0 = opt._period_conversion_value(opt.input.initial_flat, 0)
    target = float(opt._constraint_value(opt.input.initial_flat))
    floor = natural_p0 + 0.01  # protect (lift) year-0 retention above its natural level

    res = opt.optimize(
        target_constraint=target, bounds=(0.9, 1.1), method="SLSQP",
        per_period_targets={0: ("floor", floor)},
    )
    assert res["result"].success
    p0_conv = opt.output.per_period_summary.set_index("period").loc[0, "conversion"]
    assert p0_conv >= floor - 1e-3


def test_per_period_target_equality_is_hit():
    opt = _period_optimizer(n_periods=2, discount_rate=0.0)
    natural_p0 = opt._period_conversion_value(opt.input.initial_flat, 0)
    target = float(opt._constraint_value(opt.input.initial_flat))
    want = natural_p0 - 0.01  # pin year-0 conversion to an explicit (off-natural) level

    res = opt.optimize(
        target_constraint=target, bounds=(0.9, 1.1), method="SLSQP",
        per_period_targets={0: ("target", want)},
    )
    assert res["result"].success
    p0_conv = opt.output.per_period_summary.set_index("period").loc[0, "conversion"]
    assert p0_conv == pytest.approx(want, abs=3e-3)


def test_l2_regularizer_pulls_overlay_toward_one():
    t = float(_constraint_optimizer()._constraint_value(_constraint_optimizer().input.initial_flat))

    free = _constraint_optimizer()
    free_m = list(free.optimize(target_constraint=t, bounds=(0.9, 1.1))["optimal_multipliers"]["age_mult"].values())

    reg = _constraint_optimizer(regularizer=make_log_multiplier_regularizer(l2_penalty=1.0e4))
    reg_m = list(reg.optimize(target_constraint=t, bounds=(0.9, 1.1))["optimal_multipliers"]["age_mult"].values())

    free_spread = max(free_m) - min(free_m)
    reg_spread = max(reg_m) - min(reg_m)
    assert free_spread > 0.02         # non-vacuous: unregularized redistributes the overlay
    assert reg_spread < free_spread   # L2 on log(mu) keeps the overlay closer to 1.0


def test_average_expected_profit_objective_optimizes():
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.05, objective_type="average_expected_profit")
    target = float(opt._constraint_value(opt.input.initial_flat))
    res = opt.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP")

    assert res["result"].success
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3
    assert res["summary"]["objective_type"].iloc[0] == "average_expected_profit"


def test_per_period_targets_validation():
    opt = _constraint_optimizer(n_periods=2)
    target = float(opt._constraint_value(opt.input.initial_flat))
    with pytest.raises(ValueError, match="period"):
        opt.optimize(target_constraint=target, per_period_targets={5: ("floor", 0.5)})  # out of range
    with pytest.raises(ValueError, match="kind"):
        opt.optimize(target_constraint=target, per_period_targets={0: ("bogus", 0.5)})
    with pytest.raises(TypeError, match="numeric"):
        opt.optimize(target_constraint=target, per_period_targets={0: ("floor", "high")})
    with pytest.raises(TypeError, match="tuple"):
        opt.optimize(target_constraint=target, per_period_targets={0: 0.5})
    with pytest.raises(TypeError, match="enbp_ceiling"):
        opt.optimize(target_constraint=target, enbp_ceiling="low")


def test_log_multiplier_regularizer_penalizes_deviation():
    reg = make_log_multiplier_regularizer(l1_penalty=2.0, l2_penalty=3.0)
    assert reg({"f": {"a": 1.0}}) == pytest.approx(0.0)  # log(1) = 0 -> no penalty
    lm = np.log(1.2)
    assert reg({"f": {"a": 1.2}}) == pytest.approx(2.0 * abs(lm) + 3.0 * lm * lm)
    # Symmetric in log-space: a markup and its reciprocal discount cost the same.
    assert reg({"f": {"a": 1.25}}) == pytest.approx(reg({"f": {"a": 1 / 1.25}}))
    # Sums across every factor and category.
    assert reg({"f1": {"a": 1.2}, "f2": {"b": 1.2}}) == pytest.approx(2 * reg({"f": {"a": 1.2}}))
    with pytest.raises(ValueError):
        make_log_multiplier_regularizer(l1_penalty=-1.0)


# --------------------------------------------------------------------------- #
# Phase 5 — performance (precompute / overlay-once / warm-start)
# --------------------------------------------------------------------------- #

def test_demographic_aging_precompute_paid_once(monkeypatch):
    """The x-independent demographic trajectory (design §13.1) is built once and
    cached — aging is NOT repaid on every objective evaluation."""
    # apply_aging is invoked from the extracted horizon engine, so patch it there
    # (the aging methods live on _MultiPeriodHorizonEngine in _horizon_engine).
    import open_rating_engine.optimization._horizon_engine as eng

    opt = _invariant_optimizer(n_periods=3)
    calls = {"n": 0}
    real = eng.apply_aging

    def counting(rules, frames):
        calls["n"] += 1
        return real(rules, frames)

    monkeypatch.setattr(eng, "apply_aging", counting)

    base = opt.input.initial_flat
    for k in range(4):  # four distinct candidate overlays -> four full simulations
        opt._objective_fn(base + 0.01 * (k + 1))

    # Only the trajectory's demographic aging steps run (n_periods - 1), once —
    # the previous_gwp feedback is threaded per period without apply_aging.
    assert calls["n"] == opt.parameters.n_periods - 1


def test_overlay_injected_once_per_simulation():
    """The single shared overlay is deep-copied / injected once per evaluation and
    reused across all periods, not re-injected per period."""
    opt = _invariant_optimizer(n_periods=3)
    calls = {"n": 0}
    real = opt._apply_overlay

    def counting(x, pipeline):
        calls["n"] += 1
        return real(x, pipeline)

    opt._apply_overlay = counting
    opt._simulate_horizon(opt.input.initial_flat)
    assert calls["n"] == 1


def test_warm_start_converges_and_validates_x0():
    opt = _constraint_optimizer(n_periods=1)
    target = float(opt._constraint_value(opt.input.initial_flat))

    # Warm-start from a perturbed point: still reaches the constrained optimum.
    res = opt.optimize(
        target_constraint=target, bounds=(0.9, 1.1), x0=opt.input.initial_flat * 1.03
    )
    assert res["result"].success
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3

    with pytest.raises(ValueError, match="x0"):
        opt.optimize(target_constraint=target, x0=np.ones(opt.input.param_total + 1))


def test_precompute_preserves_single_period_reduction():
    """Behaviour preservation: the precompute path still satisfies the n_periods=1
    forward sim invariants — NPV equals the plain weighted profit at r=0, scored
    exactly once through the (cached-trajectory) horizon."""
    opt = _invariant_optimizer(n_periods=1, discount_rate=0.0)
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    w = agg.df["sample_weights"].to_numpy()
    profit = agg.df["expected_profit"].to_numpy()
    assert agg.npv == pytest.approx(float((w * profit).sum()))
    # Trajectory cache is populated and reused (one entry per period).
    assert len(opt._demographic_trajectory()) == 1


# --------------------------------------------------------------------------- #
# Phase 6 — going-concern new-business inflow + vintaged overlays
# --------------------------------------------------------------------------- #

def _going_concern_optimizer(n_periods=3, nb_growth_rate=0.0, **kw):
    """Going-concern optimizer: a fresh new-business cohort is written every
    period. Bands span the ages reached over the horizon so no decision variable
    is flat (1 NB row + 2 renewal rows seed the book)."""
    pipe = _build_pipeline(["15-24", "25-34", "35-44", "45-54"], derive_bands=True)
    nb = {"risk": pd.DataFrame({
        "driver_youngest_age": [25], "base_premium": [500.0],
        "technical_cost": [300.0], "previous_gwp": [490.0],
    })}
    rn = {"risk": pd.DataFrame({
        "driver_youngest_age": [40, 50], "base_premium": [550.0, 620.0],
        "technical_cost": [320.0, 350.0], "previous_gwp": [530.0, 600.0],
    })}
    return PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        new_business_mode="going_concern", nb_growth_rate=nb_growth_rate,
        aging_rules_rn={
            "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        },
        **_MP_COLS, **kw,
    )


def test_going_concern_injects_and_graduates_new_business():
    opt = _going_concern_optimizer(n_periods=3)
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    pp = opt._per_period_summary(agg)

    # Fresh new business is written every period (vs runoff: only at period 0).
    assert (pp["new_business_weight"] > 0).all()
    # The book grows as each period's converted NB graduates into the renewal
    # stock: rows = seed (1 NB + 2 RN) then +1 graduating cohort per period.
    counts = agg.df.groupby("__period").size().tolist()
    assert counts == [3, 4, 5]
    # Renewal stock persists every period, seeded by the two in-force renewals.
    # Its *weight* need not grow — survivorship shrinks carried weights by the
    # conversion rate (< 1), which can outpace the graduating-cohort inflow even
    # as the row *count* (above) rises.
    assert (pp["renewal_weight"] > 0).all()
    assert pp.loc[pp["period"] == 0, "renewal_weight"].iloc[0] == pytest.approx(2.0)


def test_nb_growth_scales_new_business_volume():
    opt = _going_concern_optimizer(n_periods=3, nb_growth_rate=0.2)
    pp = opt._per_period_summary(
        opt._simulate_and_aggregate(opt.input.initial_flat)
    ).set_index("period")

    base = pp.loc[0, "new_business_weight"]
    assert base == pytest.approx(1.0)                                   # one NB policy, weight 1 at t=0
    assert pp.loc[1, "new_business_weight"] == pytest.approx(base * 1.2)
    assert pp.loc[2, "new_business_weight"] == pytest.approx(base * 1.2 ** 2)


def test_nb_growth_is_noop_under_runoff():
    # Growth only applies to fresh NB inflow; runoff writes none after period 0.
    opt = _invariant_optimizer(n_periods=3, discount_rate=0.0)
    opt.parameters.nb_growth_rate = 0.5  # would scale NB if any were injected
    pp = opt._per_period_summary(opt._simulate_and_aggregate(opt.input.initial_flat))
    assert (pp.loc[pp["period"] > 0, "new_business_weight"] == 0.0).all()


def test_going_concern_optimize_converges():
    opt = _going_concern_optimizer(n_periods=2)
    target = float(opt._constraint_value(opt.input.initial_flat))
    res = opt.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP")

    assert res["result"].success
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3
    assert (opt.output.per_period_summary["new_business_weight"] > 0).all()


def test_vintaged_overlay_applies_per_transaction_type():
    pipe = _build_vintaged_pipeline()
    # NB and a renewal in the SAME age band, so the only pricing difference is
    # the transaction_type (vintage) multiplier.
    nb = {"risk": pd.DataFrame({
        "driver_youngest_age": [45], "transaction_type": pd.Series(["NB"], dtype=object),
        "base_premium": [500.0], "technical_cost": [300.0], "previous_gwp": [490.0],
    })}
    rn = {"risk": pd.DataFrame({
        "driver_youngest_age": [45], "transaction_type": pd.Series(["RN"], dtype=object),
        "base_premium": [500.0], "technical_cost": [300.0], "previous_gwp": [490.0],
    })}
    opt = PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=1, new_business_mode="going_concern",
        aging_rules_rn={
            "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
            "risk.transaction_type": _src("risk", "transaction_type", "'RN'"),
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        },
        **_MP_COLS,
    )

    # The vintage is a first-class decision factor — no rewrite needed (§8.3).
    assert set(opt.input.factor_names) == {"age_mult", "tt_mult"}
    assert set(opt.input.initial_multipliers["tt_mult"]) == {"NB", "RN"}

    # Inject distinct vintage multipliers and score period 0 (both rows age 45).
    x = np.array(opt.input.initial_flat, dtype=float)
    off, cats = opt.input.param_offsets["tt_mult"], opt.input.category_orders["tt_mult"]
    x[off + cats.index("NB")] = 1.06
    x[off + cats.index("RN")] = 0.94
    df = opt._simulate_horizon(x)

    p0 = df[df["__period"] == 0]
    nb_mult = p0.loc[p0["transaction_type"] == "NB", "demand_multiplier"].iloc[0]
    rn_mult = p0.loc[p0["transaction_type"] == "RN", "demand_multiplier"].iloc[0]
    # Identical age band ⇒ identical age multiplier ⇒ the ratio is purely vintage.
    assert nb_mult / rn_mult == pytest.approx(1.06 / 0.94, rel=1e-6)


# --------------------------------------------------------------------------- #
# Progress readout (bar / text) — a side-effect-only heartbeat for the solve
# --------------------------------------------------------------------------- #

def test_progress_false_is_silent(capsys):
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    target = float(opt._constraint_value(opt.input.initial_flat))
    opt.optimize(target_constraint=target, bounds=(0.9, 1.1), method="SLSQP", progress=False)
    # The silent path prints nothing to stdout (tqdm bars, if any, go to stderr).
    assert capsys.readouterr().out == ""


def test_progress_text_prints_iteration_lines(capsys):
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    # Shift the target off the initial constraint so SLSQP must iterate (and so the
    # per-iteration callback fires at least once).
    target = float(opt._constraint_value(opt.input.initial_flat)) - 0.02
    opt.optimize(target_constraint=target, bounds=(0.8, 1.2), method="SLSQP", progress="text")
    out = capsys.readouterr().out
    assert "iter 1" in out
    assert "npv=" in out and "conv=" in out
    # SLSQP enforces constraints exactly (no penalty term), so `pen` is omitted.
    assert "pen=" not in out


def test_progress_text_includes_penalty(capsys):
    # Penalty methods minimise -npv + w*penalty; the readout surfaces that penalty.
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    target = float(opt._constraint_value(opt.input.initial_flat)) - 0.02  # force a gap
    opt.optimize(
        target_constraint=target, bounds=(0.8, 1.2), method="Nelder-Mead",
        optimizer_options={"maxiter": 3}, progress="text",
    )
    out = capsys.readouterr().out
    assert "pen=" in out


def test_progress_bar_matches_silent_result():
    # The readout is side-effect-only: bar mode must not change the optimum.
    results = {}
    for mode in (False, "bar"):
        opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
        t = float(opt._constraint_value(opt.input.initial_flat))
        results[mode] = opt.optimize(
            target_constraint=t, bounds=(0.9, 1.1), method="SLSQP", progress=mode
        )
    assert np.allclose(results[False]["result"].x, results["bar"]["result"].x)
    assert results["bar"]["summary"]["npv"].iloc[0] == pytest.approx(
        results[False]["summary"]["npv"].iloc[0]
    )


def test_progress_invalid_value_raises():
    opt = _invariant_optimizer(n_periods=1)
    with pytest.raises(ValueError, match="progress"):
        opt.optimize(target_constraint=0.5, progress="loud")


def test_efficient_frontier_progress_text(capsys):
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    c = float(opt._constraint_value(opt.input.initial_flat))
    opt.calculate_efficient_frontier([c, c - 0.01], bounds=(0.8, 1.2), method="SLSQP", progress="text")
    out = capsys.readouterr().out
    assert "target" in out                 # the outer sweep prints one line per target
    assert "iter" not in out               # inner per-target solves stay silent


# --------------------------------------------------------------------------- #
# baseline() — a realistic, horizon-consistent target_constraint
# --------------------------------------------------------------------------- #

def test_baseline_matches_simulate_aggregate():
    # baseline()["conversion"] is exactly the constraint optimize() measures at
    # the do-nothing point — same _simulate_and_aggregate path, so they agree.
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    base = opt.baseline()
    agg = opt._simulate_and_aggregate(opt.input.initial_flat)
    assert base["conversion"] == pytest.approx(agg.constraint)
    assert base["npv"] == pytest.approx(agg.npv)
    assert base["total_volume"] == pytest.approx(agg.volume)


def test_baseline_target_is_feasible():
    # The headline guarantee: the baseline conversion is an achievable equality
    # target. optimize() starts feasible at x0 = initial_flat and hits it.
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.05)
    t = opt.baseline()["conversion"]
    res = opt.optimize(target_constraint=t, bounds=(0.9, 1.1), method="SLSQP", progress=False)
    assert res["result"].success
    assert res["summary"]["constraint"].iloc[0] == pytest.approx(t, abs=1e-3)
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-3


def test_baseline_at_explicit_x():
    # Passing x scores an arbitrary overlay (general evaluator), matching that
    # x's own aggregate constraint rather than the do-nothing point.
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.0)
    x = opt.input.initial_flat * 1.05  # uniform +5% price → lower conversion
    base = opt.baseline(x=x)
    assert base["conversion"] == pytest.approx(opt._simulate_and_aggregate(x).constraint)
    # A non-trivial overlay moves conversion off the do-nothing baseline.
    assert base["conversion"] != pytest.approx(opt.baseline()["conversion"])


def test_baseline_is_pure():
    # baseline() must not mutate the optimization log or store an optimized book.
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.0)
    before = len(opt.output.optimization_scoring_history)
    opt.baseline()
    assert len(opt.output.optimization_scoring_history) == before
    assert not hasattr(opt.output, "output_optimized")


def test_baseline_shape():
    opt = _invariant_optimizer(n_periods=2, discount_rate=0.0)
    base = opt.baseline()
    assert set(base) == {
        "conversion", "npv", "total_volume",
        "summary", "per_period_summary", "output_dataframe",
    }
    assert list(base["per_period_summary"]["period"]) == [0, 1]
    summary = base["summary"]
    assert len(summary) == 1
    assert summary["method"].iloc[0] == "baseline"
    assert summary["constraint_gap"].iloc[0] == pytest.approx(0.0)


def test_baseline_invalid_x_raises():
    opt = _invariant_optimizer(n_periods=1)
    with pytest.raises(ValueError, match="1-D vector of length"):
        opt.baseline(x=np.ones(opt.input.param_total + 1))


# --------------------------------------------------------------------------- #
# going-concern identifier collision (id_columns re-keying)
# --------------------------------------------------------------------------- #
#
# Fresh new business is re-injected from data_nb verbatim each period, so it
# reuses the carried policies' policy_number. A policy→risk derivation keyed on
# policy_number then attributes the fresh 'NB' to the carried renewal. id_columns
# makes each period's fresh cohort's identifiers collision-free.

def _collision_pipeline():
    """Pipeline that derives a risk-level ``transaction_type`` from the POLICY
    table via ``groupby(policy_number).first`` + left-join — the collision-prone,
    policy_number-keyed derivation that mirrors a real book bringing policy
    attributes onto risk."""
    pipe = PyfuncPricingPipeline(_context())
    pipe.options["show_progress"] = False
    pipe.add_parameter(ColumnCalculation(
        name="risk_age_band", target_table="risk", output_column="risk_age_band",
        calculation=(
            "pd.cut(df['driver_youngest_age'], bins=list(range(15, 66, 10)) + [np.inf], "
            "labels=['15-24','25-34','35-44','45-54','55-64','65+']).astype('object')"
        ),
        dependencies=[],
    ))
    pipe.add_parameter(RatingTablesLayer(
        name="demand_multiplier", target_table="risk", output_column="demand_multiplier",
        rating_tables={"age_mult": pd.DataFrame({
            "risk_age_band": pd.Series(_ALL_BANDS, dtype=object), "value": [1.0] * len(_ALL_BANDS),
        })},
        default_values={"age_mult": 1.0}, dependencies=["risk_age_band"],
    ))
    pipe.add_parameter(TableAggregation(
        name="policy_agg_tt", source_table="policy", output_table="policy_agg_tt",
        groupby_key="policy_number", aggregations={"transaction_type": ("transaction_type", "first")},
        dependencies=[],
    ))
    pipe.add_parameter(TableJoin(
        name="risk_tt_join", left_table="risk", right_table="policy_agg_tt",
        left_on="policy_number", columns=["transaction_type"], dependencies=["policy_agg_tt"],
    ))
    return _add_economics(pipe)


def _collision_optimizer(n_periods=2, id_columns=None):
    """Going-concern optimizer over a two-table (policy + risk) book where the
    fresh NB cohort reuses the carried policy's ``policy_number`` ('P1')."""
    pipe = _collision_pipeline()
    nb = {
        "policy": pd.DataFrame({"policy_number": ["P1"], "transaction_type": pd.Series(["NB"], dtype=object)}),
        "risk": pd.DataFrame({
            "policy_number": ["P1"], "driver_youngest_age": [25],
            "base_premium": [500.0], "technical_cost": [300.0], "previous_gwp": [490.0],
        }),
    }
    rn = {
        "policy": pd.DataFrame({"policy_number": ["Q1", "Q2"], "transaction_type": pd.Series(["RN", "RN"], dtype=object)}),
        "risk": pd.DataFrame({
            "policy_number": ["Q1", "Q2"], "driver_youngest_age": [40, 50],
            "base_premium": [550.0, 620.0], "technical_cost": [320.0, 350.0], "previous_gwp": [530.0, 600.0],
        }),
    }
    return PriceOptimizerRatingFactorMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        new_business_mode="going_concern",
        aging_rules_rn={
            "risk.driver_youngest_age": _src("risk", "driver_youngest_age", "df['driver_youngest_age'] + 1"),
            "policy.transaction_type": _src("policy", "transaction_type", "'RN'"),
            "risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']"),
        },
        id_columns=id_columns,
        **_MP_COLS,
    )


def test_going_concern_id_collision_without_rekey():
    # Documents the bug: fresh NB reuses the carried policy's policy_number, so
    # the policy→risk aggregation mis-attributes 'NB' to the carried renewal.
    opt = _collision_optimizer(n_periods=2, id_columns=None)
    pol1 = opt._demographic_trajectory()[1]["policy"]
    assert pol1["policy_number"].duplicated().any()                 # 'P1' appears twice
    df = opt._simulate_horizon(opt.input.initial_flat)
    carried = df[(df["__period"] == 1) & (df["__cohort"] == "rn")]
    assert (carried["transaction_type"] == "NB").any()             # carried mis-tagged NB


def test_going_concern_id_columns_graduate_carried_business():
    opt = _collision_optimizer(
        n_periods=2,
        id_columns={"policy_number": [("policy", "policy_number"), ("risk", "policy_number")]},
    )
    pol1 = opt._demographic_trajectory()[1]["policy"]
    assert not pol1["policy_number"].duplicated().any()            # fresh NB re-keyed -> unique
    df = opt._simulate_horizon(opt.input.initial_flat)
    carried = df[(df["__period"] == 1) & (df["__cohort"] == "rn")]
    # Carried business has fully graduated: every renewal row is RN, no NB leaks.
    assert (carried["transaction_type"] == "RN").all()
    # No cartesian blow-up: seed (1 NB + 2 RN), then +1 graduating cohort.
    assert df.groupby("__period").size().tolist() == [3, 4]


def test_id_columns_rekey_preserves_fk_consistency():
    opt = _collision_optimizer(
        n_periods=2,
        id_columns={"policy_number": [("policy", "policy_number"), ("risk", "policy_number")]},
    )
    traj = opt._demographic_trajectory()
    risk1, pol1 = traj[1]["risk"], traj[1]["policy"]
    fresh = risk1[risk1["__cohort"] == "nb"]
    carried_ids = set(risk1[risk1["__cohort"] == "rn"]["policy_number"])
    # Fresh NB ids carry the period token, stay joinable to their policy rows,
    # and are disjoint from the carried book.
    assert fresh["policy_number"].str.endswith("@nb1").all()
    assert fresh["policy_number"].isin(pol1["policy_number"]).all()
    assert not (set(fresh["policy_number"]) & carried_ids)


def test_id_columns_validation():
    data_nb = {
        "policy": pd.DataFrame({"policy_number": ["P1"]}),
        "risk": pd.DataFrame({"policy_number": ["P1"]}),
    }
    validate = PriceOptimizerRatingFactorMultiPeriod._validate_id_columns
    assert validate(None, data_nb) == {}
    with pytest.raises(TypeError):
        validate("notadict", data_nb)
    with pytest.raises(TypeError):
        validate({"policy_number": [("policy",)]}, data_nb)          # not a (table, column) pair
    with pytest.raises(ValueError, match="not in data_nb"):
        validate({"policy_number": [("nope", "policy_number")]}, data_nb)
    with pytest.raises(ValueError, match="not found"):
        validate({"policy_number": [("policy", "missing")]}, data_nb)


# --------------------------------------------------------------------------- #
# Pluggable objective & constraint metrics (named menu + callable escape hatch)
# --------------------------------------------------------------------------- #
#
# Both the objective (``objective_type``, at construction) and the lifetime
# constraint (``constraint_metric`` / ``constraint_kind``, at optimize) accept a
# name from ``_AGG_METRICS`` or a callable ``agg -> float``. Defaults reproduce
# the legacy "maximise total expected profit s.t. conversion == target" exactly.


def test_objective_named_volume_wiring():
    # A named non-profit objective ("total_volume") is maximised and reported.
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0, objective_type="total_volume")
    t = float(opt._constraint_value(opt.input.initial_flat))
    res = opt.optimize(target_constraint=t, bounds=(0.9, 1.1), method="SLSQP")
    assert res["result"].success
    assert res["summary"]["objective_type"].iloc[0] == "total_volume"
    # The reported objective IS the volume metric (objective == total_volume column).
    assert res["summary"]["objective"].iloc[0] == pytest.approx(res["summary"]["total_volume"].iloc[0])


def test_objective_callable_matches_named():
    # A callable objective equal to agg.npv reproduces the "total_expected_profit"
    # optimum exactly (same objective + constraint + warm start => same overlay).
    def total_npv(agg):
        return agg.npv

    named = _constraint_optimizer(n_periods=2, discount_rate=0.0, objective_type="total_expected_profit")
    call = _constraint_optimizer(n_periods=2, discount_rate=0.0, objective_type=total_npv)
    t = float(named._constraint_value(named.input.initial_flat))
    r_named = named.optimize(target_constraint=t, bounds=(0.9, 1.1), method="SLSQP")
    r_call = call.optimize(target_constraint=t, bounds=(0.9, 1.1), method="SLSQP")

    assert r_call["result"].success
    assert r_call["summary"]["objective_type"].iloc[0] == "total_npv"   # callable __name__
    assert r_call["summary"]["npv"].iloc[0] == pytest.approx(r_named["summary"]["npv"].iloc[0], rel=1e-4)
    for cat in r_named["optimal_multipliers"]["age_mult"]:
        assert r_call["optimal_multipliers"]["age_mult"][cat] == pytest.approx(
            r_named["optimal_multipliers"]["age_mult"][cat], abs=1e-3
        )


def test_lifetime_volume_target_slsqp():
    # Equality on a NON-conversion metric: hold lifetime volume at its baseline
    # while profit is maximised over the remaining freedom (2 DOF, 1 equality).
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    base = opt._simulate_and_aggregate(opt.input.initial_flat)
    res = opt.optimize(
        target_constraint=base.volume, constraint_metric="total_volume",
        constraint_kind="target", bounds=(0.9, 1.1), method="SLSQP",
    )
    assert res["result"].success
    s = res["summary"]
    assert s["constraint_metric"].iloc[0] == "total_volume"
    assert s["constraint_kind"].iloc[0] == "target"
    # constraint_gap is measured on the VOLUME metric (not conversion) and is ~0.
    assert s["constraint_gap"].iloc[0] == pytest.approx(0.0, abs=1e-3)
    assert s["total_volume"].iloc[0] == pytest.approx(base.volume, abs=1e-3)


def test_lifetime_constraint_callable_matches_named():
    # A callable constraint equal to agg.volume reproduces the named "total_volume".
    def total_vol(agg):
        return agg.volume

    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    base = opt._simulate_and_aggregate(opt.input.initial_flat)
    r_named = opt.optimize(
        target_constraint=base.volume, constraint_metric="total_volume",
        bounds=(0.9, 1.1), method="SLSQP",
    )
    r_call = opt.optimize(
        target_constraint=base.volume, constraint_metric=total_vol,
        bounds=(0.9, 1.1), method="SLSQP",
    )
    assert r_call["result"].success
    assert r_call["summary"]["constraint_metric"].iloc[0] == "total_vol"   # callable __name__
    assert r_call["summary"]["npv"].iloc[0] == pytest.approx(r_named["summary"]["npv"].iloc[0], rel=1e-4)


def test_lifetime_conversion_floor_respected():
    # Inequality (>=) on the lifetime metric: a floor above the natural level is
    # honoured. Exercised under SLSQP (hard ineq dict).
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    natural = float(opt._constraint_value(opt.input.initial_flat))
    floor = natural + 0.01
    res = opt.optimize(
        target_constraint=floor, constraint_metric="conversion", constraint_kind="floor",
        bounds=(0.9, 1.1), method="SLSQP",
    )
    assert res["result"].success
    assert res["summary"]["constraint"].iloc[0] >= floor - 2e-3
    assert res["summary"]["constraint_kind"].iloc[0] == "floor"


def test_lifetime_conversion_ceiling_respected():
    # Inequality (<=) on the lifetime metric: a ceiling below the natural level is
    # honoured.
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    natural = float(opt._constraint_value(opt.input.initial_flat))
    ceiling = natural - 0.01
    res = opt.optimize(
        target_constraint=ceiling, constraint_metric="conversion", constraint_kind="ceiling",
        bounds=(0.9, 1.1), method="SLSQP",
    )
    assert res["result"].success
    assert res["summary"]["constraint"].iloc[0] <= ceiling + 2e-3
    assert res["summary"]["constraint_kind"].iloc[0] == "ceiling"


def test_volume_floor_binds_under_penalty_method():
    # The penalty path (Nelder-Mead) honours an inequality via the squared-hinge:
    # a volume floor ABOVE baseline pulls volume up toward it (only the violated
    # side is penalised). A high weight keeps the soft constraint tight despite
    # volume's small magnitude here.
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    base = opt._simulate_and_aggregate(opt.input.initial_flat)
    floor = base.volume + 0.05
    res = opt.optimize(
        target_constraint=floor, constraint_metric="total_volume", constraint_kind="floor",
        bounds=(0.8, 1.2), method="Nelder-Mead", constraint_penalty_weight=1.0e6,
        optimizer_options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6},
    )
    # Volume is lifted from baseline toward the floor (the floor genuinely binds).
    assert res["summary"]["total_volume"].iloc[0] > base.volume + 0.02


def test_constraint_and_objective_metric_validation():
    opt = _constraint_optimizer(n_periods=1)
    t = float(opt._constraint_value(opt.input.initial_flat))
    with pytest.raises(ValueError, match="constraint_metric"):
        opt.optimize(target_constraint=t, constraint_metric="bogus")
    with pytest.raises(TypeError, match="constraint_metric"):
        opt.optimize(target_constraint=t, constraint_metric=123)
    with pytest.raises(ValueError, match="constraint_kind"):
        opt.optimize(target_constraint=t, constraint_kind="bogus")
    # Objective is validated at construction.
    with pytest.raises(ValueError, match="objective_type"):
        _constraint_optimizer(objective_type="bogus")
    with pytest.raises(TypeError, match="objective_type"):
        _constraint_optimizer(objective_type=123)


def test_constraint_defaults_match_legacy_conversion():
    # No new kwargs => conversion equality, byte-for-byte legacy behaviour.
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    t = float(opt._constraint_value(opt.input.initial_flat))
    res = opt.optimize(target_constraint=t, bounds=(0.9, 1.1), method="SLSQP")
    assert res["result"].success
    s = res["summary"]
    assert s["constraint_metric"].iloc[0] == "conversion"
    assert s["constraint_kind"].iloc[0] == "target"
    assert s["constraint"].iloc[0] == pytest.approx(t, abs=1e-3)
    assert abs(s["constraint_gap"].iloc[0]) < 1e-3


def test_progress_text_shows_constraint_metric_label(capsys):
    # A non-conversion metric labels the readout with its name (replacing "conv");
    # a binding volume floor forces the solver to iterate so a line is emitted.
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    base = opt._simulate_and_aggregate(opt.input.initial_flat)
    opt.optimize(
        target_constraint=base.volume * 1.02, constraint_metric="total_volume",
        constraint_kind="floor", bounds=(0.8, 1.2), method="SLSQP", progress="text",
    )
    out = capsys.readouterr().out
    assert "total_volume=" in out
    assert "conv=" not in out


def test_frontier_forwards_constraint_metric():
    opt = _constraint_optimizer(n_periods=2, discount_rate=0.0)
    v = opt._simulate_and_aggregate(opt.input.initial_flat).volume
    fr = opt.calculate_efficient_frontier(
        [v, v * 1.01], constraint_metric="total_volume", constraint_kind="target",
        bounds=(0.8, 1.2), method="SLSQP", progress=False,
    )
    assert len(fr) == 2
    assert (fr["constraint_metric"] == "total_volume").all()
    assert (fr["constraint_kind"] == "target").all()
    assert fr["result_success"].all()
