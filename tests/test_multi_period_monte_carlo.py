"""Phase 7 tests for ``PriceOptimizerMultiPeriod.monte_carlo_npv`` — the post-hoc
Monte-Carlo survivorship risk report (design §10.2).

These reuse the no-H2O mock-pipeline fixtures from ``test_multi_period_forward_sim``
(sibling test module; ``tests/`` is on the path under pytest's default import mode)
so the whole thing runs offline. The keystone is that the MC mean NPV converges to
the deterministic NPV (§16.2); the rest lock reproducibility, the survival math
(inclusive product × bare offered margin), going-concern identity, tail metrics,
and validation.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")  # the sibling module imports the numba-JIT single-period optimizer

from open_rating_engine.optimization.multi_period_rating_factor_price_optimizer import (
    PriceOptimizerRatingFactorMultiPeriod,
)

# Reuse the no-H2O mock-pipeline fixtures from the sibling test module. Ensure its
# directory is importable regardless of pytest's import mode.
sys.path.insert(0, os.path.dirname(__file__))
from test_multi_period_forward_sim import (  # noqa: E402
    _going_concern_optimizer,
    _invariant_optimizer,
    _src,
)

_POLICY_ID_COL = "__policy_id"


def _overlay(opt):
    """The seed overlay (all-ones multipliers) as a stress-test target — avoids a
    slow optimize() call where MC behaviour, not the optimum, is under test."""
    return opt.input.initial_multipliers


# --------------------------------------------------------------------------- #
# Reproducibility (design R5)
# --------------------------------------------------------------------------- #

def test_same_seed_reproduces_samples():
    opt = _invariant_optimizer(n_periods=3)
    a = opt.monte_carlo_npv(_overlay(opt), n_sims=300, seed=7)
    b = opt.monte_carlo_npv(_overlay(opt), n_sims=300, seed=7)
    assert np.array_equal(a.samples, b.samples)


def test_different_seed_changes_samples():
    opt = _invariant_optimizer(n_periods=3)
    a = opt.monte_carlo_npv(_overlay(opt), n_sims=300, seed=1)
    b = opt.monte_carlo_npv(_overlay(opt), n_sims=300, seed=2)
    assert not np.array_equal(a.samples, b.samples)
    assert a.summary["mean"].iloc[0] != b.summary["mean"].iloc[0]


def test_chunk_size_does_not_change_results():
    # Chunking is a memory optimization only — same seed, any chunking, same draws.
    opt = _invariant_optimizer(n_periods=3)
    a = opt.monte_carlo_npv(_overlay(opt), n_sims=250, seed=3, chunk_size=1000)
    b = opt.monte_carlo_npv(_overlay(opt), n_sims=250, seed=3, chunk_size=10)
    assert np.array_equal(a.samples, b.samples)


# --------------------------------------------------------------------------- #
# Mean convergence to the deterministic NPV (keystone, §16.2)
# --------------------------------------------------------------------------- #

def test_mc_mean_converges_to_deterministic_runoff():
    opt = _invariant_optimizer(n_periods=3, discount_rate=0.05)
    mc = opt.monte_carlo_npv(_overlay(opt), n_sims=4000, seed=0)
    mean, std = mc.summary["mean"].iloc[0], mc.summary["std"].iloc[0]
    # Seeded -> deterministic test; a 4-sigma band on the sample mean is safe.
    assert abs(mean - mc.deterministic_npv) < 4.0 * std / np.sqrt(4000)


def test_mc_mean_converges_to_deterministic_going_concern_with_growth():
    opt = _going_concern_optimizer(n_periods=3, nb_growth_rate=0.2)
    mc = opt.monte_carlo_npv(_overlay(opt), n_sims=4000, seed=0)
    mean, std = mc.summary["mean"].iloc[0], mc.summary["std"].iloc[0]
    assert abs(mean - mc.deterministic_npv) < 4.0 * std / np.sqrt(4000)


# --------------------------------------------------------------------------- #
# Survival math — the MC inputs reconstruct the deterministic NPV exactly
# (no sampling): guards the inclusive cumprod, bare-offered, and weight0 choices.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("make", [
    lambda: _invariant_optimizer(n_periods=3, discount_rate=0.05),
    lambda: _going_concern_optimizer(n_periods=3, nb_growth_rate=0.15),
])
def test_mc_inputs_reconstruct_deterministic_npv(make):
    opt = make()
    x = opt._flatten(_overlay(opt))
    P, off_over_disc, weight0 = opt._build_mc_inputs(x)
    # E[NPV] = Σ_i weight0_i · Σ_t (Π_{s<=t} p) · offered/disc — inclusive cumprod
    # over BARE offered. Must equal the deterministic forward-sim NPV.
    analytic = float((weight0[:, None] * np.cumprod(P, axis=1) * off_over_disc).sum())
    assert analytic == pytest.approx(opt._simulate_and_aggregate(x).npv, rel=1e-9)


def test_one_period_mean_is_p_times_offered():
    # Single period: survival product is just B_0, so E[NPV] = Σ p·offered/disc.
    opt = _invariant_optimizer(n_periods=1, discount_rate=0.0)
    x = opt._flatten(_overlay(opt))
    P, off_over_disc, weight0 = opt._build_mc_inputs(x)
    expected = float((weight0[:, None] * P * off_over_disc).sum())
    mc = opt.monte_carlo_npv(_overlay(opt), n_sims=6000, seed=0)
    assert mc.summary["mean"].iloc[0] == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------- #
# Going-concern identity (the __policy_id pivot)
# --------------------------------------------------------------------------- #

def test_going_concern_policy_id_count():
    opt = _going_concern_optimizer(n_periods=3)
    df = opt._simulate_horizon(opt._flatten(_overlay(opt)))
    # Seed RN (2 rows) + one fresh NB cohort (1 row) per period (3) = 5 distinct policies.
    assert df[_POLICY_ID_COL].nunique() == 5
    # Each policy persists from birth onward (deterministic carry), so the
    # row count grows 3, 4, 5 across periods.
    assert df.groupby("__period").size().tolist() == [3, 4, 5]


# --------------------------------------------------------------------------- #
# Distribution / tail metrics
# --------------------------------------------------------------------------- #

def test_percentiles_and_tail_metrics_ordered():
    opt = _invariant_optimizer(n_periods=3, discount_rate=0.05)
    mc = opt.monte_carlo_npv(_overlay(opt), n_sims=4000, seed=0, var_alpha=0.05)
    vals = [mc.percentiles[p] for p in (5, 25, 50, 75, 95)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))            # monotone percentiles
    assert mc.cvar <= mc.var <= mc.summary["mean"].iloc[0]        # expected shortfall <= VaR <= mean
    assert mc.var == pytest.approx(np.quantile(mc.samples, 0.05))  # VaR is the alpha-quantile
    assert mc.summary["std"].iloc[0] > 0                          # genuine dispersion


def test_summary_columns_and_return_samples_flag():
    opt = _invariant_optimizer(n_periods=2)
    mc = opt.monte_carlo_npv(_overlay(opt), n_sims=200, seed=0, return_samples=False)
    assert mc.samples is None
    row = mc.summary.iloc[0]
    for col in ("mean", "std", "deterministic_npv", "mean_minus_deterministic",
                "p5", "p95", "var_0.05", "cvar_0.05", "n_sims", "new_business_mode"):
        assert col in mc.summary.columns
    assert row["n_sims"] == 200
    assert row["new_business_mode"] == "runoff"


# --------------------------------------------------------------------------- #
# Default resolution + persistence
# --------------------------------------------------------------------------- #

def test_defaults_to_optimize_result_and_can_save():
    opt = _invariant_optimizer(n_periods=2)
    # Mimic a completed optimize() without the slow solve.
    opt.output.optimal_multipliers = opt.input.initial_multipliers
    mc = opt.monte_carlo_npv(n_sims=200, seed=0, save_to_output=True)
    assert mc.samples is not None
    assert opt.output.monte_carlo is mc


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_requires_optimal_multipliers():
    opt = _invariant_optimizer(n_periods=2)  # optimize() not run, none passed
    with pytest.raises(RuntimeError, match="optimize"):
        opt.monte_carlo_npv(n_sims=10)


def test_requires_from_pipeline():
    bare = PriceOptimizerRatingFactorMultiPeriod(
        data_nb={"risk": pd.DataFrame({"a": [1.0]})},
        data_rn={"risk": pd.DataFrame({"a": [1.0]})},
        n_periods=1,
        aging_rules_rn={"risk.a": _src("risk", "a", "df['a']")},
    )
    with pytest.raises(RuntimeError, match="from_pipeline"):
        bare.monte_carlo_npv(optimal_multipliers={"x": {"y": 1.0}}, n_sims=10)


def test_argument_validation():
    opt = _invariant_optimizer(n_periods=2)
    ov = _overlay(opt)
    with pytest.raises(ValueError, match="n_sims"):
        opt.monte_carlo_npv(ov, n_sims=0)
    with pytest.raises(ValueError, match="var_alpha"):
        opt.monte_carlo_npv(ov, var_alpha=0.0)
    with pytest.raises(ValueError, match="var_alpha"):
        opt.monte_carlo_npv(ov, var_alpha=1.0)
    with pytest.raises(ValueError, match="percentiles"):
        opt.monte_carlo_npv(ov, percentiles=(5, 150))
    with pytest.raises(ValueError, match="chunk_size"):
        opt.monte_carlo_npv(ov, chunk_size=0)


def test_flatten_roundtrip_and_missing_key():
    opt = _invariant_optimizer(n_periods=2)
    x = opt.input.initial_flat
    assert np.allclose(opt._flatten(opt._unflatten(x)), x)
    with pytest.raises(KeyError, match="factor"):
        opt._flatten({})
