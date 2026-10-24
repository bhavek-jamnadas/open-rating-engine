"""Characterization tests for ``PriceOptimizerRatingFactor``.

These lock in the single-period optimizer's behaviour so the Phase-1 refactor
(extracting ``_RatingFactorDecisionSpace``) can be proven behaviour-preserving:
the same assertions must pass before and after the extraction. They run fully
offline -- no S3, no H2O, no real pipeline -- building a real ``RatingTablesLayer``
plus a tiny stub pipeline for the ``from_pipeline`` path.

Imported via the submodule path (not the package ``__init__``) so the file is
agnostic to the package-level export wiring.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")  # the optimizer JIT-compiles its interpolator

from open_rating_engine.optimization.rating_factor_price_optimizer import (
    PriceOptimizerRatingFactor,
)
from open_rating_engine import RatingTablesLayer


# The complete set of attributes the optimizer packs into ``self.input``. Locking
# this exact set is the single most valuable guard for the extraction: the base
# builds 8 of them, the subclass adds the 7 grid attrs, and the merge must
# reproduce all 15 with no duplicates or omissions.
_INPUT_ATTRS = {
    "adjustment_factors", "af_grid", "objective_array", "constraint_array",
    "sample_weights", "reporting_arrays", "regularizer",
    "factor_names", "category_orders", "row_category_indices", "default_contrib",
    "param_offsets", "param_total", "initial_flat", "initial_multipliers",
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny():
    """Direct-``__init__`` kwargs: N=3 rows, K=3 adjustment factors, 1 factor
    with two decision categories (rows map A, B, A)."""
    return dict(
        adjustment_factors=np.array([0.8, 1.0, 1.2]),
        objective_array=np.array([[10., 9., 8.], [20., 18., 16.], [5., 4.5, 4.]]),
        constraint_array=np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75], [0.8, 0.7, 0.6]]),
        rating_factors={"f": np.array(["A", "B", "A"], dtype=object)},
        initial_multipliers={"f": {"A": 1.0, "B": 1.0}},
    )


@pytest.fixture
def layer():
    """A real multiplicative ``RatingTablesLayer`` with one table, three bands."""
    rt = pd.DataFrame({
        "risk_age_band": pd.Series(["A", "B", "C"], dtype=object),
        "value": [1.1, 1.0, 0.9],
    })
    return RatingTablesLayer(
        name="risk.demand_multiplier",
        target_table="risk",
        output_column="demand_multiplier",
        rating_tables={"age_tbl": rt},
        default_values={"age_tbl": 1.0},
    )


class _StubPipeline:
    """Minimal stand-in exposing only what ``from_pipeline`` reads: ``.layers``."""

    def __init__(self, layer):
        self.layers = {layer.name: layer}


def _grid_5x3():
    """Well-conditioned (N=5, K=3) objective/constraint grids for the
    ``from_pipeline`` book (rows A, B, A, C, D -- 'D' exercises the table
    default)."""
    af = np.array([0.8, 1.0, 1.2])
    obj = np.array([
        [10., 9., 8.],
        [12., 11., 10.],
        [8., 7., 6.],
        [15., 13., 11.],
        [9., 8., 7.],
    ])
    con = np.array([
        [0.90, 0.80, 0.70],
        [0.92, 0.82, 0.72],
        [0.88, 0.78, 0.68],
        [0.95, 0.85, 0.75],
        [0.91, 0.81, 0.71],
    ])
    return af, obj, con


def _scored_5():
    return {"risk": pd.DataFrame(
        {"risk_age_band": pd.Series(["A", "B", "A", "C", "D"], dtype=object)}
    )}


# --------------------------------------------------------------------------- #
# Group (i): direct __init__ -- decision space + objective evaluation
# --------------------------------------------------------------------------- #

def test_init_populates_input_namespace(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)

    assert set(vars(opt.input)) == _INPUT_ATTRS
    assert opt.input.factor_names == ("f",)
    assert opt.input.param_total == 2
    assert opt.input.param_offsets == {"f": 0}
    assert opt.input.category_orders == {"f": ("A", "B")}
    assert np.array_equal(opt.input.row_category_indices["f"], [0, 1, 0])
    assert np.array_equal(opt.input.initial_flat, [1.0, 1.0])
    assert opt.input.af_grid.shape == (3, 3)
    assert opt.input.default_contrib == {}


def test_effective_multiplier_single_factor(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    # x = [A=1.1, B=0.9]; rows map A, B, A -> [1.1, 0.9, 1.1].
    eff = opt._effective_multiplier(np.array([1.1, 0.9]))
    assert eff == pytest.approx([1.1, 0.9, 1.1])


def test_effective_multiplier_is_product_of_two_factors():
    opt = PriceOptimizerRatingFactor(
        adjustment_factors=np.array([0.8, 1.0, 1.2]),
        objective_array=np.array([[10., 9., 8.], [20., 18., 16.], [5., 4.5, 4.]]),
        constraint_array=np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75], [0.8, 0.7, 0.6]]),
        rating_factors={
            "f1": np.array(["A", "B", "A"], dtype=object),
            "f2": np.array(["X", "X", "Y"], dtype=object),
        },
        initial_multipliers={
            "f1": {"A": 1.0, "B": 1.0},
            "f2": {"X": 1.0, "Y": 1.0},
        },
    )
    assert opt.input.param_total == 4
    assert opt.input.param_offsets == {"f1": 0, "f2": 2}
    # x = [f1.A=1.1, f1.B=0.9, f2.X=1.05, f2.Y=0.95].
    # f1 per row -> [1.1, 0.9, 1.1]; f2 per row -> [1.05, 1.05, 0.95].
    eff = opt._effective_multiplier(np.array([1.1, 0.9, 1.05, 0.95]))
    assert eff == pytest.approx([1.1 * 1.05, 0.9 * 1.05, 1.1 * 0.95])


def test_unflatten_roundtrip(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    out = opt._unflatten(np.array([1.1, 0.9]))
    assert out == {"f": {"A": 1.1, "B": 0.9}}
    assert all(isinstance(v, float) for v in out["f"].values())


def test_build_bounds_tuple_and_dict(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    assert opt._build_bounds((0.8, 1.2)) == [(0.8, 1.2)] * 2
    assert opt._build_bounds({"f": (0.9, 1.1)}) == [(0.9, 1.1)] * 2
    # A factor omitted from the dict falls back to the (0.8, 1.2) default.
    assert opt._build_bounds({}) == [(0.8, 1.2)] * 2
    with pytest.raises(ValueError):
        opt._build_bounds({"ghost": (1.0, 1.0)})
    with pytest.raises(TypeError):
        opt._build_bounds((1.0,))


def test_optimize_slsqp_invariants(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    res = opt.optimize(target_constraint=0.80, bounds=(0.8, 1.2), method="SLSQP")

    assert bool(res["result"].success) is True
    summary = res["summary"]
    assert abs(summary["constraint_gap"].iloc[0]) < 1e-4
    assert summary["method"].iloc[0] == "SLSQP"

    # Decision variables stay within bounds.
    for v in res["optimal_multipliers"]["f"].values():
        assert 0.8 - 1e-9 <= v <= 1.2 + 1e-9

    df = res["output_dataframe"]
    assert list(df.columns[:4]) == [
        "effective_multiplier", "objective", "constraint", "sample_weights",
    ]
    # Single factor => effective multiplier equals the chosen per-row multiplier.
    assert np.allclose(df["effective_multiplier"], df["f_multiplier"])
    # save_to_log defaults True.
    assert opt.output.optimal_multipliers == res["optimal_multipliers"]


def test_optimize_nelder_mead_penalty(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    res = opt.optimize(
        target_constraint=0.80, bounds=(0.8, 1.2),
        method="Nelder-Mead", constraint_penalty_weight=1000.0,
    )
    # Soft penalty -- looser tolerance than the SLSQP equality constraint.
    assert abs(res["summary"]["constraint_gap"].iloc[0]) < 1e-2
    assert res["summary"]["method"].iloc[0] == "Nelder-Mead"
    for v in res["optimal_multipliers"]["f"].values():
        assert 0.8 - 1e-6 <= v <= 1.2 + 1e-6


def test_default_multipliers_for_unmatched(tiny):
    kw = dict(tiny)
    kw["rating_factors"] = {"f": np.array(["A", "B", "C"], dtype=object)}  # C absent
    kw["initial_multipliers"] = {"f": {"A": 1.0, "B": 1.0}}

    opt = PriceOptimizerRatingFactor(**kw, default_multipliers={"f": 1.0})
    assert opt.input.default_contrib == {"f": 1.0}
    assert np.array_equal(opt.input.row_category_indices["f"], [0, 1, -1])
    # Row C (idx -1) takes the fixed default, not a decision variable.
    contrib = opt._factor_contribution("f", np.array([1.2, 0.9]))
    assert contrib[2] == pytest.approx(1.0)

    # Without a default for the unmatched category, construction must fail loudly.
    with pytest.raises(KeyError, match="default_multipliers"):
        PriceOptimizerRatingFactor(**kw)


def test_save_to_log_false_leaves_output(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    opt.optimize(target_constraint=0.80, save_to_log=False)
    assert not hasattr(opt.output, "optimal_multipliers")
    assert len(opt.output.optimization_scoring_history) == 0


def test_slsqp_options_deprecation(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    with pytest.warns(DeprecationWarning, match="slsqp_options"):
        opt.optimize(target_constraint=0.80, slsqp_options={"maxiter": 50})


def test_calculate_efficient_frontier(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    ef = opt.calculate_efficient_frontier([0.78, 0.80, 0.82])
    assert isinstance(ef, pd.DataFrame)
    assert len(ef) >= 1
    for col in ("target_constraint", "constraint", "objective", "constraint_gap"):
        assert col in ef.columns
    assert opt.output.efficient_frontier is ef


# --------------------------------------------------------------------------- #
# Group (ii): from_pipeline -- layer parsing + deploy round-trip
# --------------------------------------------------------------------------- #

def test_from_pipeline_builds_and_parses(layer):
    af, obj, con = _grid_5x3()
    opt = PriceOptimizerRatingFactor.from_pipeline(
        pipeline=_StubPipeline(layer),
        layer_name="risk.demand_multiplier",
        scored_data=_scored_5(),
        adjustment_factors=af,
        objective_array=obj,
        constraint_array=con,
    )

    assert opt.input.factor_names == ("age_tbl",)
    assert opt.input.initial_multipliers["age_tbl"] == {"A": 1.1, "B": 1.0, "C": 0.9}
    assert opt.input.default_contrib == {"age_tbl": 1.0}
    # Rows A, B, A, C, D -> 0, 1, 0, 2, -1 ('D' unmatched -> fixed default).
    assert np.array_equal(opt.input.row_category_indices["age_tbl"], [0, 1, 0, 2, -1])

    meta = opt._layer_meta
    assert meta["name"] == "risk.demand_multiplier"
    assert meta["target_table"] == "risk"
    assert meta["output_column"] == "demand_multiplier"
    assert meta["formula"] is None
    assert meta["default_values"] == {"age_tbl": 1.0}
    assert meta["dependencies"] == []
    assert opt._layer_schema["age_tbl"]["key_cols"] == ["risk_age_band"]


def test_from_pipeline_to_rating_tables_roundtrip(layer):
    af, obj, con = _grid_5x3()
    opt = PriceOptimizerRatingFactor.from_pipeline(
        pipeline=_StubPipeline(layer), layer_name="risk.demand_multiplier",
        scored_data=_scored_5(), adjustment_factors=af,
        objective_array=obj, constraint_array=con,
    )
    opt.optimize(target_constraint=0.80, bounds=(0.8, 1.2))

    tables = opt.to_rating_tables()
    age = tables["age_tbl"]
    assert list(age.columns) == ["risk_age_band", "value"]
    # Only the three decision categories appear; 'D' is a fixed default, never a row.
    assert set(age["risk_age_band"]) == {"A", "B", "C"}
    assert age["value"].dtype.kind == "f"
    assert age["value"].between(0.8 - 1e-9, 1.2 + 1e-9).all()


def test_from_pipeline_to_layer_reconstructs(layer):
    af, obj, con = _grid_5x3()
    opt = PriceOptimizerRatingFactor.from_pipeline(
        pipeline=_StubPipeline(layer), layer_name="risk.demand_multiplier",
        scored_data=_scored_5(), adjustment_factors=af,
        objective_array=obj, constraint_array=con,
    )
    opt.optimize(target_constraint=0.80, bounds=(0.8, 1.2))

    rebuilt = opt.to_layer()
    assert isinstance(rebuilt, RatingTablesLayer)
    assert rebuilt.name == "risk.demand_multiplier"
    assert rebuilt.target_table == "risk"
    assert rebuilt.output_column == "demand_multiplier"
    assert rebuilt.formula is None
    assert rebuilt.default_values == {"age_tbl": 1.0}
    assert rebuilt.dependencies == []
    assert set(rebuilt.rating_tables) == {"age_tbl"}

    assert opt.to_layer(name="override").name == "override"


def test_to_rating_tables_requires_from_pipeline(tiny):
    opt = PriceOptimizerRatingFactor(**tiny)
    with pytest.raises(RuntimeError, match="from_pipeline"):
        opt.to_rating_tables()
    with pytest.raises(RuntimeError, match="from_pipeline"):
        opt.to_layer()


def test_from_pipeline_rejects_missing_layer(layer):
    af, obj, con = _grid_5x3()
    with pytest.raises(KeyError, match="not found"):
        PriceOptimizerRatingFactor.from_pipeline(
            pipeline=_StubPipeline(layer), layer_name="ghost",
            scored_data=_scored_5(), adjustment_factors=af,
            objective_array=obj, constraint_array=con,
        )
    # Layer target table absent from scored_data.
    with pytest.raises(KeyError):
        PriceOptimizerRatingFactor.from_pipeline(
            pipeline=_StubPipeline(layer), layer_name="risk.demand_multiplier",
            scored_data={"wrong": pd.DataFrame({"risk_age_band": ["A"]})},
            adjustment_factors=af, objective_array=obj, constraint_array=con,
        )
