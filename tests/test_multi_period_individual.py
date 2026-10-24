"""Phase 2 tests for ``PriceOptimizerIndividualMultiPeriod`` — the per-policy
(lifetime-multiplier) multi-period optimizer.

The two keystones (design ``multi_period_individual_optimizer.md`` §15):

- **15.1 single-period reduction (R1):** with ``n_periods == 1`` and
  ``discount_rate == 0`` the per-policy optimum is identical to running
  :class:`PriceOptimizerIndividual` on the single-period grid built by
  :func:`simulate_adjustment_grid` on the same pipeline.
- **15.2 separability (S1):** perturbing one policy's multiplier moves *only* that
  policy's simulated conversion / premium — the property the whole decomposition
  rests on.

Everything runs on a small **no-H2O** pipeline whose demand overlay is a
per-policy ``risk_number`` rating table.
"""

import numpy as np
import pandas as pd
import pytest

from mlflow.pyfunc import PythonModelContext

from open_rating_engine import (
    ColumnCalculation,
    PyfuncPricingPipeline,
    RatingTablesLayer,
)
from open_rating_engine.optimization.individual_price_optimizer import PriceOptimizerIndividual
from open_rating_engine.optimization.multi_period_individual_price_optimizer import (
    PriceOptimizerIndividualMultiPeriod,
    make_log_multiplier_penalty,
)
from open_rating_engine.optimization._horizon_engine import _PERIOD_COL, _POLICY_ID_COL
from open_rating_engine.optimization.simulate_adjustment_grid import (
    make_demand_emulation,
    simulate_adjustment_grid,
)


# --------------------------------------------------------------------------- #
# A small no-H2O pipeline with a per-policy (risk_number) demand overlay
# --------------------------------------------------------------------------- #

_MP_COLS = dict(
    conversion_col="conversion_rate",
    profit_col="expected_profit",
    gwp_col="gwp",
    cost_col="technical_cost",
    sample_weight_col="sample_weights",
)


def _context():
    return PythonModelContext(artifacts={}, model_config={"environment": "local"})


def _add_economics(pipe):
    """gwp -> price_change -> (logistic) conversion -> profit. Conversion decreases
    in price, so there is a genuine conversion/profit trade-off per policy."""
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


def _build_individual_pipeline(risk_numbers):
    """Pipeline whose demand overlay is ONE rating table keyed on ``risk_number``
    (identity 1.0 initial values) — the per-policy 'individual' overlay."""
    pipe = PyfuncPricingPipeline(_context())
    pipe.options["show_progress"] = False
    pipe.add_parameter(RatingTablesLayer(
        name="demand_multiplier",
        target_table="risk",
        output_column="demand_multiplier",
        rating_tables={
            "risk_number": pd.DataFrame({
                "risk_number": pd.Series(risk_numbers, dtype=object),
                "value": [1.0] * len(risk_numbers),
            })
        },
        default_values={"risk_number": 1.0},
        dependencies=[],
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


# Heterogeneous book: per-policy price levels differ, so the constrained optimum
# genuinely redistributes the multiplier across policies (no vacuous all-ones).
def _books():
    nb = {"risk": pd.DataFrame({
        "risk_number": pd.Series(["P1", "P2"], dtype=object),
        "base_premium": [450.0, 520.0],
        "technical_cost": [280.0, 310.0],
        "previous_gwp": [430.0, 545.0],
        "sample_weights": [1.0, 2.0],
    })}
    rn = {"risk": pd.DataFrame({
        "risk_number": pd.Series(["P3", "P4", "P5"], dtype=object),
        "base_premium": [510.0, 480.0, 600.0],
        "technical_cost": [300.0, 350.0, 300.0],
        "previous_gwp": [500.0, 455.0, 625.0],
        "sample_weights": [1.5, 1.0, 3.0],
    })}
    return nb, rn


def _all_risk_numbers(nb, rn):
    return nb["risk"]["risk_number"].tolist() + rn["risk"]["risk_number"].tolist()


_FEEDBACK_RULE = {"risk.previous_gwp": _src("risk", "previous_gwp", "df['gwp']")}


def _build_optimizer(nb, rn, pipe, *, n_periods, discount_rate=0.0):
    return PriceOptimizerIndividualMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods,
        aging_rules_rn=_FEEDBACK_RULE, discount_rate=discount_rate, **_MP_COLS,
    )


def _baseline_conversion(pipe, book):
    scored = pipe.predict(book)["risk"]
    return float(np.average(scored["conversion_rate"], weights=book["risk"]["sample_weights"]))


# --------------------------------------------------------------------------- #
# 15.1 — single-period reduction (the keystone)
# --------------------------------------------------------------------------- #

def test_single_period_reduces_to_individual_optimum():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    grid = np.arange(0.85, 1.1501, 0.01)
    target = _baseline_conversion(pipe, combined)

    # Single-period reference: sweep the per-policy overlay, then λ-bisect.
    swept = simulate_adjustment_grid(
        data=combined, pipeline=pipe,
        premium_emulation=make_demand_emulation("demand_multiplier", "risk_number", mode="override"),
        output_table="risk",
        output_columns={"objective_array": "expected_profit", "constraint_array": "conversion_rate"},
        adjustment_factor=grid,
    )
    sp = PriceOptimizerIndividual(
        adjustment_factors=swept["adjustment_factor_array"],
        objective_array=swept["objective_array"],
        constraint_array=swept["constraint_array"],
        sample_weights=combined["risk"]["sample_weights"].to_numpy(),
    )
    # A 5-policy discrete grid gives a step-function constraint, so use a tolerance
    # the steps can satisfy. The SP==MP reduction holds at any *shared* tol (both
    # run the identical λ-bisection over identical arrays).
    sp_res = sp.optimize(target_constraint=target, tol=0.02)
    sp_map = dict(zip(combined["risk"]["risk_number"], sp_res["optimal_multipliers"]))

    # Multi-period at n_periods=1, r=0 — must reproduce it exactly. ENBP off, to match
    # the unconstrained single-period benchmark (the reduction is about the math).
    mp = _build_optimizer(nb, rn, pipe, n_periods=1, discount_rate=0.0)
    mp_res = mp.optimize(
        target_constraint=target, grid=grid, constraint_metric="lifetime_average",
        enbp_ceiling=None, tol=0.02,
    )
    mp_map = dict(zip(mp.to_per_policy_table()["risk_number"], mp.to_per_policy_table()["value"]))

    assert set(mp_map) == set(sp_map)
    for rnum in sp_map:                      # same grid node per policy
        assert mp_map[rnum] == pytest.approx(sp_map[rnum], abs=1e-6), rnum
    # Aggregates match tightly (robust even if a policy were near-indifferent).
    assert mp_res["summary"]["constraint"].iloc[0] == pytest.approx(sp_res["summary"]["constraint"].iloc[0], abs=1e-6)
    assert mp_res["summary"]["objective"].iloc[0] == pytest.approx(sp_res["summary"]["objective"].iloc[0], rel=1e-6)
    # Non-vacuous: the constrained optimum redistributes the overlay across policies.
    assert max(sp_map.values()) - min(sp_map.values()) > 0.02


# --------------------------------------------------------------------------- #
# 15.2 — separability (validates S1)
# --------------------------------------------------------------------------- #

def test_perturbing_one_policy_leaves_every_other_unchanged():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)

    cats = opt.input.category_orders[opt._policy_factor]
    x0 = np.ones(opt.input.param_total)
    x1 = x0.copy()
    x1[0] = 1.15                       # perturb exactly one policy's multiplier
    perturbed = cats[0]

    df0 = opt._simulate_horizon(x0)
    df1 = opt._simulate_horizon(x1)
    merged = df0.merge(df1, on=[_POLICY_ID_COL, _PERIOD_COL], suffixes=("_0", "_1"))

    others = merged[merged["risk_number_0"] != perturbed]
    for col in ("conversion_rate", "gwp", "sample_weights", "expected_profit", "previous_gwp"):
        assert np.allclose(others[f"{col}_0"].to_numpy(), others[f"{col}_1"].to_numpy()), col

    # ...and the perturbed policy genuinely moved (guards against a vacuous pass).
    moved = merged[merged["risk_number_0"] == perturbed]
    assert not np.allclose(moved["gwp_0"].to_numpy(), moved["gwp_1"].to_numpy())


# --------------------------------------------------------------------------- #
# Deployment + construction guards
# --------------------------------------------------------------------------- #

def test_to_per_policy_table_covers_book_with_grid_values():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    grid = np.linspace(0.9, 1.1, 11)
    opt.optimize(target_constraint=_baseline_conversion(pipe, combined), grid=grid, enbp_ceiling=None, tol=0.02)
    tbl = opt.to_per_policy_table()

    assert list(tbl.columns) == ["risk_number", "value"]
    assert set(tbl["risk_number"]) == set(_all_risk_numbers(nb, rn))
    assert len(tbl) == 5
    # every deployed multiplier is one of the grid nodes, inside the bounds
    assert np.isin(tbl["value"].to_numpy(), grid).all()


def test_from_pipeline_rejects_multi_factor_overlay():
    # A two-table (rating-factor) overlay is NOT the per-policy individual shape.
    pipe = PyfuncPricingPipeline(_context())
    pipe.options["show_progress"] = False
    pipe.add_parameter(RatingTablesLayer(
        name="demand_multiplier", target_table="risk", output_column="demand_multiplier",
        rating_tables={
            "risk_number": pd.DataFrame({"risk_number": pd.Series(["P3", "P4", "P5"], dtype=object), "value": [1.0, 1.0, 1.0]}),
            "band": pd.DataFrame({"band": pd.Series(["a", "b"], dtype=object), "value": [1.0, 1.0]}),
        },
        default_values={"risk_number": 1.0, "band": 1.0},
        dependencies=[],
    ))
    _add_economics(pipe)
    nb, rn = _books()
    rn["risk"]["band"] = pd.Series(["a", "b", "a"], dtype=object)
    nb["risk"]["band"] = pd.Series(["a", "b"], dtype=object)

    with pytest.raises(ValueError, match="single per-policy overlay"):
        _build_optimizer(nb, rn, pipe, n_periods=1)


def test_from_pipeline_rejects_a_col_absent_from_the_scored_data():
    # The bare test pipeline has no "risk.previous_gwp" column; passing it (as a user would
    # by mistake for a namespaced pipeline) must fail loudly at construction with an
    # actionable message naming the offending *_col — not silently break the feedback.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    with pytest.raises(ValueError, match="previous_gwp_col"):
        PriceOptimizerIndividualMultiPeriod.from_pipeline(
            pipeline=pipe, layer_name="demand_multiplier",
            scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
            data_nb=nb, data_rn=rn, n_periods=2,
            aging_rules_rn=_FEEDBACK_RULE,
            previous_gwp_col="risk.previous_gwp",  # absent from the bare scored data
            **_MP_COLS,
        )


# --------------------------------------------------------------------------- #
# Multi-period sanity
# --------------------------------------------------------------------------- #

def test_efficient_frontier_builds_arrays_once_and_is_monotone():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    frontier = opt.calculate_efficient_frontier(
        np.linspace(0.47, 0.52, 4), grid=np.linspace(0.9, 1.1, 11), enbp_ceiling=None, tol=0.02
    )
    assert len(frontier) == 4  # every (interior) target is achievable
    # Reaching a higher conversion target means charging less -> lower avg multiplier.
    assert frontier["constraint"].iloc[0] < frontier["constraint"].iloc[-1]
    assert frontier["price_multiplier"].iloc[0] > frontier["price_multiplier"].iloc[-1]


def test_zero_discount_lifetime_npv_is_undiscounted_survivorship_sum():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.0)

    arrays = opt._build_lifetime_arrays(np.array([1.0]))
    # At r=0 the per-exposure lifetime NPV is Σ_t (w_{i,t}/w0)·expected_profit_t,
    # i.e. the undiscounted survivorship-weighted profit. Reconstruct it directly.
    df = opt._simulate_and_aggregate(np.full(opt.input.param_total, 1.0)).df
    w0 = df[df[_PERIOD_COL] == 0].set_index(_POLICY_ID_COL)["sample_weights"]
    expected = (
        df.assign(contrib=df["sample_weights"] * df["expected_profit"])
        .groupby(_POLICY_ID_COL)["contrib"].sum()
        / w0
    ).reindex(arrays["policy_index"]).to_numpy()
    assert np.allclose(arrays["lifetime_npv"][:, 0], expected)


# --------------------------------------------------------------------------- #
# Robustness guards (no silently-wrong results)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Going-concern Phase 1 — valuation frame (in-force decision; NB held at baseline)
#
# Per docs/design/multi_period_individual_optimizer_going_concern.md §15 Phase 1:
# the per-policy decision is the in-force renewal book only; the NB stream
# (period-0 NB + every injected cohort) is rule-priced at the baseline layer
# default and valued in whole-book reporting. The renewal-book lifetime arrays are
# byte-identical to runoff (separability S1-G / proposition P-G), and the path-DP
# subclass must reject going-concern (it has no fixed policy set).
# --------------------------------------------------------------------------- #

# Re-key risk_number so each injected NB cohort is globally unique and falls through
# the per-policy table to the layer default (the recommended id_columns wiring).
_GC_ID_COLUMNS = {"risk_number": [("risk", "risk_number")]}


def _build_going_concern_optimizer(nb, rn, pipe, *, n_periods, discount_rate=0.0,
                                   nb_growth_rate=0.0, id_columns="default"):
    if id_columns == "default":
        id_columns = _GC_ID_COLUMNS
    return PriceOptimizerIndividualMultiPeriod.from_pipeline(
        pipeline=pipe, layer_name="demand_multiplier",
        scored_data_nb=pipe.predict(nb), scored_data_rn=pipe.predict(rn),
        data_nb=nb, data_rn=rn, n_periods=n_periods, aging_rules_rn=_FEEDBACK_RULE,
        discount_rate=discount_rate, new_business_mode="going_concern",
        nb_growth_rate=nb_growth_rate, id_columns=id_columns, **_MP_COLS,
    )


def test_going_concern_is_accepted_and_harvest_is_inforce_only():
    # GC-2: the period-0-centric NaN bug is fixed by restricting the harvest to the
    # in-force renewal book (P3,P4,P5); period-0 NB (P1,P2) is excluded.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    arr = gc._build_lifetime_arrays(np.linspace(0.9, 1.1, 11))
    assert set(arr["risk_keys"]) == {"P3", "P4", "P5"}        # in-force only
    assert not np.isnan(arr["birth_weights"]).any()
    assert not np.isnan(arr["lifetime_npv"]).any()


def test_going_concern_inforce_arrays_are_byte_identical_to_runoff():
    # GC-3 (keystone, S1-G / P-G): injecting the NB stream does NOT perturb any
    # in-force policy's harvested lifetime arrays. The renewal book's arrays match
    # runoff's exactly — which is why Phase 1 leaves the renewal decision unchanged.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    grid = np.linspace(0.9, 1.1, 11)
    runoff = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    ro, gca = runoff._build_lifetime_arrays(grid), gc._build_lifetime_arrays(grid)
    ro_keys, gc_keys = list(ro["risk_keys"]), list(gca["risk_keys"])
    for rnum in ("P3", "P4", "P5"):
        i, j = ro_keys.index(rnum), gc_keys.index(rnum)
        for key in ("lifetime_npv", "lifetime_conversion", "lifetime_gwp", "lifetime_volume"):
            assert np.array_equal(ro[key][i], gca[key][j]), (key, rnum)
        assert ro["birth_weights"][i] == gca["birth_weights"][j]


def test_going_concern_optimize_reports_whole_book_and_deploys_inforce_only():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    base = gc.baseline()
    res = gc.optimize(
        target_constraint=base["conversion"], grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )
    s = res["summary"]
    gc_cols = {"npv_inforce", "npv_nb", "npv_whole_book",
               "volume_inforce", "volume_nb", "volume_whole_book"}
    assert gc_cols <= set(s.columns)
    # baseline and optimize summaries still line up column-for-column under going-concern.
    assert list(base["summary"].columns) == list(s.columns)
    # The deployed per-policy table is the in-force book only — NB is rule-priced.
    assert set(gc.to_per_policy_table()["risk_number"]) == {"P3", "P4", "P5"}
    # whole-book = in-force + NB, and equals the whole-book per-period NPV sum.
    assert s["npv_whole_book"].iloc[0] == pytest.approx(s["npv_inforce"].iloc[0] + s["npv_nb"].iloc[0])
    assert gc.output.per_period_summary["npv"].sum() == pytest.approx(s["npv_whole_book"].iloc[0], rel=1e-7)
    # The headline npv stays the optimized in-force block (== npv_inforce, by S1-G).
    assert s["npv"].iloc[0] == pytest.approx(s["npv_inforce"].iloc[0], rel=1e-9)
    # New business is genuinely present every period and contributes value.
    assert (gc.output.per_period_summary["new_business_weight"] > 0).all()
    assert s["npv_nb"].iloc[0] != 0.0


def test_going_concern_allows_nb_ids_absent_from_overlay_table():
    # The coverage check is relaxed under going-concern: only in-force (data_rn) ids
    # must be table rows; NB ids fall through to the baseline default.
    nb, rn = _books()
    pipe = _build_individual_pipeline(rn["risk"]["risk_number"].tolist())  # P3,P4,P5 only
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)
    gc.optimize(gc.baseline()["conversion"], grid=np.linspace(0.9, 1.1, 11),
                enbp_ceiling=None, tol=0.02)
    assert set(gc.to_per_policy_table()["risk_number"]) == {"P3", "P4", "P5"}


def test_going_concern_inforce_id_absent_from_overlay_table_is_derived():
    # Under going-concern an in-force (data_rn) id absent from the table is now derived
    # from the scored in-force book and optimized — no pre-seeding required. NB stays excluded.
    nb, rn = _books()
    pipe = _build_individual_pipeline(["P3", "P4"])  # in-force P5 omitted; NB (P1,P2) also absent
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)
    gc.optimize(gc.baseline()["conversion"], grid=np.linspace(0.9, 1.1, 11),
                enbp_ceiling=None, tol=0.02)
    assert set(gc.to_per_policy_table()["risk_number"]) == {"P3", "P4", "P5"}  # P5 derived; NB excluded


def test_going_concern_rejects_nb_inforce_id_collision():
    nb, rn = _books()
    nb["risk"]["risk_number"] = pd.Series(["P3", "X9"], dtype=object)  # P3 collides with in-force
    pipe = _build_individual_pipeline(["P3", "P4", "P5", "X9"])
    with pytest.raises(ValueError, match="overlap"):
        _build_going_concern_optimizer(nb, rn, pipe, n_periods=2)


def test_going_concern_runs_without_id_columns():
    # id_columns is recommended (re-keying) but optional for a pipeline with no
    # id-keyed joins — the in-force arrays are unaffected by NB pricing (S1-G).
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=2, id_columns=None)
    res = gc.optimize(gc.baseline()["conversion"], grid=np.linspace(0.9, 1.1, 11),
                      enbp_ceiling=None, tol=0.02)
    assert set(gc.to_per_policy_table()["risk_number"]) == {"P3", "P4", "P5"}
    assert "npv_whole_book" in res["summary"].columns


def test_going_concern_nb_growth_scales_new_business_weight():
    # GC-9: nb_growth_rate scales each period's fresh NB cohort by (1+g)^t.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=3, nb_growth_rate=0.2)
    pp = gc.baseline()["per_period"].set_index("period")
    base_nb = pp.loc[0, "new_business_weight"]
    assert base_nb == pytest.approx(3.0)                       # P1 (w=1) + P2 (w=2)
    assert pp.loc[1, "new_business_weight"] == pytest.approx(base_nb * 1.2)
    assert pp.loc[2, "new_business_weight"] == pytest.approx(base_nb * 1.2 ** 2)


def test_going_concern_monte_carlo_runs_from_the_individual_class():
    # GC-10: the engine's going-concern MC pivot, previously unreachable from this
    # class (blocked at from_pipeline), now runs at the deployed overlay.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    gc.optimize(gc.baseline()["conversion"], grid=np.linspace(0.9, 1.1, 11),
                enbp_ceiling=None, tol=0.02)
    mc = gc.monte_carlo_npv(n_sims=4000, seed=0)
    assert mc.samples.shape == (4000,)
    assert mc.summary["new_business_mode"].iloc[0] == "going_concern"
    assert mc.summary["mean"].iloc[0] == pytest.approx(mc.deterministic_npv, rel=0.2)


def test_going_concern_at_n_periods_one_harvests_inforce_and_values_nb():
    # GC-11: at n_periods=1 no NB is injected after period 0, but period-0 NB is
    # still rule-priced (excluded from the per-policy harvest) and valued whole-book.
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    gc = _build_going_concern_optimizer(nb, rn, pipe, n_periods=1, discount_rate=0.0)
    arr = gc._build_lifetime_arrays(np.array([1.0]))
    assert set(arr["risk_keys"]) == {"P3", "P4", "P5"}        # period-0 NB excluded
    base = gc.baseline()
    # Whole-book NPV includes the period-0 NB stream, so it exceeds the in-force block.
    assert base["summary"]["npv_whole_book"].iloc[0] != base["summary"]["npv_inforce"].iloc[0]


def test_inforce_policy_absent_from_overlay_table_is_derived_and_optimized():
    # The overlay need not be pre-seeded per policy: the decision space is derived from
    # the scored in-force book. P5 is absent from the table yet is tuned (gets a swept
    # grid value) and deploys — instead of raising the old coverage error.
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(["P1", "P2", "P3", "P4"])  # omits P5
    opt = _build_optimizer(nb, rn, pipe, n_periods=1, discount_rate=0.0)  # runoff
    grid = np.linspace(0.9, 1.1, 11)
    opt.optimize(target_constraint=_baseline_conversion(pipe, combined), grid=grid,
                 enbp_ceiling=None, tol=0.02)
    tbl = opt.to_per_policy_table()
    assert set(tbl["risk_number"]) == set(_all_risk_numbers(nb, rn))  # all 5, incl. derived P5
    assert "P5" in set(tbl["risk_number"])
    assert np.isin(tbl["value"].to_numpy(), grid).all()              # P5 got a swept grid value


def test_derivation_matches_a_fully_pre_seeded_overlay():
    # Deriving the decision space from the book gives the SAME optimum as a fully
    # pre-seeded overlay (both start every policy at the 1.0 default) — so the change is
    # behaviour-preserving for already-seeded tables and "fills in" the rest identically.
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    grid = np.linspace(0.9, 1.1, 11)
    pipe_full = _build_individual_pipeline(_all_risk_numbers(nb, rn))  # every policy seeded
    pipe_partial = _build_individual_pipeline(["P3"])                  # rest are derived
    target = _baseline_conversion(pipe_full, combined)

    def _optimum(pipe):
        opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.05)
        opt.optimize(target_constraint=target, grid=grid, enbp_ceiling=None, tol=0.02)
        t = opt.to_per_policy_table()
        return dict(zip(t["risk_number"], t["value"]))

    full, partial = _optimum(pipe_full), _optimum(pipe_partial)
    assert set(full) == set(_all_risk_numbers(nb, rn))
    assert full == partial  # identical per-policy optimum, seeded vs derived
    assert max(full.values()) - min(full.values()) > 0.02  # non-vacuous: genuinely tuned


def test_monte_carlo_npv_reports_distribution_for_the_optimum():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    opt.optimize(
        _baseline_conversion(pipe, combined), grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )

    mc = opt.monte_carlo_npv(n_sims=5000, seed=0)
    # The adapter fed the engine a complete per-policy overlay (no missing-category error).
    assert mc.samples.shape == (5000,)
    # Mean MC NPV converges to the deterministic NPV (design §16.2); tails are ordered.
    assert mc.summary["mean"].iloc[0] == pytest.approx(mc.deterministic_npv, rel=0.15)
    assert mc.cvar <= mc.var <= mc.summary["mean"].iloc[0]
    # Reproducible under a fixed seed.
    assert np.array_equal(mc.samples, opt.monte_carlo_npv(n_sims=5000, seed=0).samples)


def test_monte_carlo_npv_requires_an_optimum_first():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2)
    with pytest.raises(RuntimeError, match="run optimize"):
        opt.monte_carlo_npv()


def test_year1_metric_targets_first_period_conversion():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    # Year-1 conversion at all-ones is just the single-period (period-0) conversion.
    target = _baseline_conversion(pipe, combined)
    res = opt.optimize(
        target_constraint=target, grid=np.linspace(0.9, 1.1, 11),
        constraint_metric="year1", enbp_ceiling=None, tol=0.02,
    )
    assert res["summary"]["constraint"].iloc[0] == pytest.approx(target, abs=0.02)


# --------------------------------------------------------------------------- #
# Phase 3 — ENBP ceiling + regularization (constraints & compliance, §9.3/§11)
# --------------------------------------------------------------------------- #

def test_adjust_objective_masks_multipliers_above_ceiling():
    grid = np.array([0.9, 1.0, 1.1])
    npv = np.array([[10.0, 11.0, 12.0], [5.0, 6.0, 7.0]])
    obj = PriceOptimizerIndividualMultiPeriod._adjust_objective(
        npv, grid, enbp_ceiling=1.0, regularizer=None
    )
    assert np.isneginf(obj[:, 2]).all()           # k = 1.1 > 1.0 forbidden
    assert np.allclose(obj[:, :2], npv[:, :2])     # k <= 1.0 untouched


def test_adjust_objective_subtracts_log_multiplier_penalty():
    grid = np.array([0.9, 1.0, 1.1])
    npv = np.zeros((2, 3))
    obj = PriceOptimizerIndividualMultiPeriod._adjust_objective(
        npv, grid, enbp_ceiling=None, regularizer=make_log_multiplier_penalty(l2_penalty=100.0),
    )
    assert obj[0, 1] == pytest.approx(0.0)         # no penalty at k = 1.0
    assert obj[0, 0] < 0.0 and obj[0, 2] < 0.0     # penalized away from 1.0 (both directions)


def test_enbp_ceiling_below_grid_raises():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=1)
    with pytest.raises(ValueError, match="forbids every grid multiplier"):
        opt.optimize(0.5, grid=np.linspace(1.05, 1.10, 6), enbp_ceiling=1.0, tol=0.02)


def test_default_ceiling_forbids_markups_that_are_otherwise_used():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)
    grid = np.linspace(0.9, 1.2, 31)  # markup headroom up to 1.2
    # Clearly below baseline (> tol below the ceiling-feasible min, which is the
    # all-ones conversion), so it needs markups and is infeasible under the ceiling.
    low_target = _baseline_conversion(pipe, combined) - 0.05

    # With the ceiling OFF, this lower-conversion (more-profit) target is reached by
    # marking some policies up above 1.0.
    free = opt.optimize(low_target, grid=grid, enbp_ceiling=None, tol=0.02)
    assert (np.asarray(free["optimal_multipliers"]) > 1.0 + 1e-9).any()

    # Under the default ENBP ceiling (1.0) markups are forbidden, so the same target
    # is infeasible — the ceiling genuinely removes the price-walking option.
    with pytest.raises(ValueError, match="No solution"):
        opt.optimize(low_target, grid=grid, tol=0.02)


def test_regularizer_shrinks_overlay_toward_one():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)
    grid = np.linspace(0.9, 1.1, 21)
    target = _baseline_conversion(pipe, combined)

    free = opt.optimize(target, grid=grid, enbp_ceiling=None, tol=0.02)
    penalized = opt.optimize(
        target, grid=grid, enbp_ceiling=None,
        regularizer=make_log_multiplier_penalty(l2_penalty=50000.0), tol=0.02,
    )
    # A strong log-multiplier penalty pulls every policy back toward 1.0, collapsing
    # the overlay spread relative to the unregularized optimum.
    assert np.ptp(np.asarray(penalized["optimal_multipliers"])) < np.ptp(np.asarray(free["optimal_multipliers"]))


# --------------------------------------------------------------------------- #
# Phase 4 — performance (parallel sweeps + cohorting, §12)
# --------------------------------------------------------------------------- #

def test_parallel_sweep_matches_sequential():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    grid = np.linspace(0.9, 1.1, 11)

    seq = opt._build_lifetime_arrays(grid, n_jobs=1)
    par = opt._build_lifetime_arrays(grid, n_jobs=4)
    for key in ("lifetime_npv", "lifetime_conversion", "lifetime_gwp", "lifetime_volume", "birth_weights"):
        assert np.allclose(seq[key], par[key]), key
    assert np.array_equal(seq["risk_keys"], par["risk_keys"])


def test_invalid_n_jobs_raises():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=1)
    with pytest.raises(ValueError, match="n_jobs"):
        opt.optimize(0.5, grid=np.linspace(0.9, 1.1, 5), enbp_ceiling=None, n_jobs=0)


def test_cohorting_matches_full_resolution_exactly():
    # Two distinct policy profiles across five policies (identical features, different
    # ids/weights), so cohorting collapses 5 policies -> 2 response cells.
    nb = {"risk": pd.DataFrame({
        "risk_number": pd.Series(["P1", "P2"], dtype=object),
        "base_premium": [500.0, 500.0],
        "technical_cost": [300.0, 300.0],
        "previous_gwp": [490.0, 490.0],
        "sample_weights": [1.0, 2.0],
    })}
    rn = {"risk": pd.DataFrame({
        "risk_number": pd.Series(["P3", "P4", "P5"], dtype=object),
        "base_premium": [600.0, 600.0, 600.0],
        "technical_cost": [340.0, 340.0, 340.0],
        "previous_gwp": [610.0, 610.0, 610.0],
        "sample_weights": [1.5, 1.0, 3.0],
    })}
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.05)
    grid = np.linspace(0.9, 1.1, 11)
    target = _baseline_conversion(pipe, combined)

    full = opt.optimize(target, grid=grid, enbp_ceiling=None, tol=0.02)
    n_full = opt._inner.input.objective_array.shape[0]
    full_opt = np.asarray(full["optimal_multipliers"]).copy()

    coh = opt.optimize(target, grid=grid, enbp_ceiling=None, cohort=True, tol=0.02)
    n_coh = opt._inner.input.objective_array.shape[0]

    assert n_full == 5 and n_coh == 2                       # 5 policies -> 2 response cells
    assert np.allclose(full_opt, np.asarray(coh["optimal_multipliers"]))  # exact same per-policy optimum
    # deployment still names every policy
    assert set(opt.to_per_policy_table()["risk_number"]) == {"P1", "P2", "P3", "P4", "P5"}


# --------------------------------------------------------------------------- #
# baseline() — the do-nothing horizon (achievable target_constraint helper)
# --------------------------------------------------------------------------- #

def test_baseline_is_an_achievable_do_nothing_target():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    base = opt.baseline()

    # conversion == the birth-weighted average of the per-policy lifetime conversions
    # — exactly the quantity the inner optimizer / optimize() constrains.
    arrays = opt._build_lifetime_arrays(np.array([1.0]))
    manual_conv = float(np.average(arrays["lifetime_conversion"][:, 0], weights=arrays["birth_weights"]))
    assert base["conversion"] == pytest.approx(manual_conv, abs=1e-12)

    # npv is the TOTAL book lifetime NPV: Σ_i w0·(NPV_i/w0) == Σ_i NPV_i, i.e. the
    # engine's row-summed expected_npv at the same do-nothing overlay (per-exposure
    # round-trip).
    agg = opt._simulate_and_aggregate(np.full(opt.input.param_total, 1.0))
    assert base["npv"] == pytest.approx(agg.npv, rel=1e-9)

    # per_policy names every policy with its do-nothing response.
    assert set(base["per_policy"]["risk_number"]) == set(_all_risk_numbers(nb, rn))
    assert len(base["per_policy"]) == 5

    # ...and it is an achievable equality target: optimizing TO the do-nothing
    # conversion converges there (the do-nothing overlay is itself a feasible point
    # hitting that exact constraint, so the λ-bisection reaches it).
    grid = np.linspace(0.9, 1.1, 11)   # includes 1.0
    res = opt.optimize(target_constraint=base["conversion"], grid=grid, enbp_ceiling=None, tol=0.02)
    assert res["summary"]["constraint"].iloc[0] == pytest.approx(base["conversion"], abs=0.02)


def test_baseline_reduces_to_single_period_snapshot_at_n1_r0():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=1, discount_rate=0.0)

    # At n_periods=1, r=0 the lifetime do-nothing conversion is exactly the
    # single-period snapshot (the layer is seeded at identity 1.0, so period-0
    # scoring == pipe.predict(combined)).
    assert opt.baseline()["conversion"] == pytest.approx(_baseline_conversion(pipe, combined), abs=1e-9)


def test_baseline_respects_constraint_metric():
    nb, rn = _books()
    combined = {"risk": pd.concat([nb["risk"], rn["risk"]], ignore_index=True)}
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    # year1 == the period-0 (single-period snapshot) birth-weighted conversion, exactly.
    year1 = opt.baseline(constraint_metric="year1")["conversion"]
    assert year1 == pytest.approx(_baseline_conversion(pipe, combined), abs=1e-9)

    # lifetime_average blends later periods (previous_gwp feedback changes renewal
    # conversion), so it moves off the year-1 snapshot.
    lifetime = opt.baseline(constraint_metric="lifetime_average")["conversion"]
    assert lifetime != pytest.approx(year1, abs=1e-6)


def test_baseline_summary_columns_match_optimize():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=2, discount_rate=0.0)

    base = opt.baseline()
    res = opt.optimize(
        target_constraint=base["conversion"], grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )
    # Same schema, so a baseline and an optimized run line up column-for-column.
    assert list(base["summary"].columns) == list(res["summary"].columns)
    assert len(base["summary"]) == 1


# --------------------------------------------------------------------------- #
# Per-period horizon summary (parity with the rating-factor optimizer)
# --------------------------------------------------------------------------- #

_PER_PERIOD_COLS = [
    "period", "conversion", "volume", "weight", "new_business_weight",
    "renewal_weight", "npv", "price_multiplier", "renewal_price_multiplier",
]


def test_optimize_and_baseline_expose_a_per_period_summary():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    # baseline returns the do-nothing per-period breakdown (mirrors RF baseline).
    base = opt.baseline()
    pp = base["per_period"]
    assert list(pp.columns) == _PER_PERIOD_COLS
    assert list(pp["period"]) == [0, 1, 2]
    assert np.allclose(pp["price_multiplier"].to_numpy(), 1.0)   # do-nothing overlay

    # optimize stores the per-period breakdown on self.output (mirrors RF).
    res = opt.optimize(
        target_constraint=base["conversion"], grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )
    opp = opt.output.per_period_summary
    assert list(opp.columns) == _PER_PERIOD_COLS
    assert len(opp) == 3
    # The per-period NPV sums to the total horizon NPV reported in the summary — a
    # cross-check that the optimum re-simulates to the same value the lifetime arrays
    # harvested (separability S1: each policy's NPV at its own multiplier is unchanged
    # by the others).
    assert opp["npv"].sum() == pytest.approx(res["summary"]["npv"].iloc[0], rel=1e-7)


def test_per_period_summary_reconciles_with_lifetime_at_single_period():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=1, discount_rate=0.0)

    base = opt.baseline()
    pp = base["per_period"]
    assert len(pp) == 1
    # At n_periods=1 the single period's pooled conversion equals the lifetime
    # (birth-weighted) conversion — both are Σ w0·p0 / Σ w0.
    assert pp["conversion"].iloc[0] == pytest.approx(base["conversion"], abs=1e-9)


def test_summary_is_enriched_with_horizon_kpis():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    base = opt.baseline()

    new_cols = {"npv", "total_volume", "constraint_metric", "constraint_gap"}
    assert new_cols <= set(base["summary"].columns)

    res = opt.optimize(
        target_constraint=base["conversion"], grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )
    s = res["summary"]
    assert new_cols <= set(s.columns)
    assert list(base["summary"].columns) == list(s.columns)   # baseline & optimize aligned

    # npv == total Σ NPV_i == the per-period NPV sum at the optimum.
    assert s["npv"].iloc[0] == pytest.approx(opt.output.per_period_summary["npv"].sum(), rel=1e-7)
    assert s["constraint_metric"].iloc[0] == "lifetime_average"
    # `constraint` stays the birth-weighted conversion we targeted (a reproducible
    # value), so the realised gap is within tolerance.
    assert abs(s["constraint_gap"].iloc[0]) <= 0.02

    # baseline: gap is exactly 0, and the enriched totals equal its own headline numbers.
    assert base["summary"]["constraint_gap"].iloc[0] == 0.0
    assert base["summary"]["npv"].iloc[0] == pytest.approx(base["npv"], rel=1e-12)
    assert base["summary"]["total_volume"].iloc[0] == pytest.approx(base["total_volume"], rel=1e-12)


# --------------------------------------------------------------------------- #
# output_dataframe = scored horizon frame (parity with the rating-factor optimizer)
# --------------------------------------------------------------------------- #

def test_baseline_output_dataframe_is_the_scored_horizon_frame():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)

    df = opt.baseline()["output_dataframe"]
    assert _PERIOD_COL in df.columns and "expected_npv" in df.columns
    assert sorted(df[_PERIOD_COL].unique().tolist()) == [0, 1, 2]
    # It is exactly the engine's scored horizon at the do-nothing overlay.
    ref = opt._simulate_and_aggregate(np.full(opt.input.param_total, 1.0)).df
    assert len(df) == len(ref)
    assert df["expected_npv"].sum() == pytest.approx(ref["expected_npv"].sum(), rel=1e-12)


def test_optimize_output_dataframe_is_the_horizon_frame_not_the_selection_table():
    nb, rn = _books()
    pipe = _build_individual_pipeline(_all_risk_numbers(nb, rn))
    opt = _build_optimizer(nb, rn, pipe, n_periods=3, discount_rate=0.05)
    base = opt.baseline()

    res = opt.optimize(
        target_constraint=base["conversion"], grid=np.linspace(0.9, 1.1, 11),
        enbp_ceiling=None, tol=0.02,
    )
    df = res["output_dataframe"]
    # Horizon frame (5 policies × 3 periods), NOT the 5-row per-policy selection table.
    assert _PERIOD_COL in df.columns
    assert len(df) == 5 * 3
    # Same object stored on self.output (mirrors RF's output_optimized).
    assert opt.output.output_optimized is df
