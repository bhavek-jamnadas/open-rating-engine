import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from tqdm.notebook import tqdm

from ._decision_space import _RatingFactorDecisionSpace, _RESERVED_SUMMARY_COLS

__all__ = [
    "PriceOptimizerRatingFactor",
]


@njit
def _fast_linear_interpolator(price_points, values, target_prices):
    n = values.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.interp(target_prices[i], price_points[i], values[i])
    return out


class PriceOptimizerRatingFactor(_RatingFactorDecisionSpace):
    """Optimize price multipliers for rating factors to achieve a target constraint.

    This class finds optimal multipliers for categorical rating factors that maximize
    an objective function while satisfying a constraint. It uses SLSQP optimization
    with linear interpolation to evaluate objectives and constraints at effective
    multiplier values.

    The decision-variable handling (flatten/unflatten, effective-multiplier
    expansion, bounds, ``from_pipeline`` parsing, and ``to_rating_tables`` /
    ``to_layer`` deployment) is inherited from
    :class:`_RatingFactorDecisionSpace`; this class supplies the precomputed-grid
    objective.
    """

    def __init__(
        self,
        adjustment_factors,
        objective_array,
        constraint_array,
        rating_factors,
        initial_multipliers,
        sample_weights=None,
        reporting_arrays=None,
        regularizer=None,
        default_multipliers=None,
    ):
        adjustment_factors = self._coerce_1d(adjustment_factors, "adjustment_factors")
        if np.any(adjustment_factors <= 0):
            raise ValueError("adjustment_factors must all be positive.")

        objective_array = self._coerce_2d(objective_array, "objective_array")
        constraint_array = self._coerce_2d(constraint_array, "constraint_array")

        K = adjustment_factors.shape[0]
        if objective_array.shape[1] != K:
            raise ValueError(
                f"objective_array has {objective_array.shape[1]} columns but adjustment_factors has length {K}."
            )
        if constraint_array.shape != objective_array.shape:
            raise ValueError("constraint_array must have the same shape as objective_array.")

        N = objective_array.shape[0]
        if sample_weights is None:
            sample_weights = np.ones(N)
        else:
            sample_weights = self._coerce_1d(sample_weights, "sample_weights")
            if sample_weights.shape[0] != N:
                raise ValueError(
                    f"sample_weights length ({sample_weights.shape[0]}) must match number of rows ({N})."
                )

        # Decision-space construction (validation + flattening) lives on the base
        # so the multi-period optimizer can reuse it; N comes from the grid here.
        decision = self._compute_decision_space(
            rating_factors, initial_multipliers, default_multipliers, n_rows=N
        )

        if reporting_arrays is None:
            reporting_arrays = {}
        else:
            if not isinstance(reporting_arrays, dict):
                raise TypeError("reporting_arrays must be a dict[str, np.ndarray].")
            coerced = {}
            for key, arr in reporting_arrays.items():
                if not isinstance(key, str):
                    raise TypeError(f"reporting_arrays keys must be strings; got {type(key).__name__}.")
                if key in _RESERVED_SUMMARY_COLS:
                    raise ValueError(
                        f"reporting_arrays key '{key}' collides with reserved summary column."
                    )
                arr = self._coerce_2d(arr, f"reporting_arrays['{key}']")
                if arr.shape != objective_array.shape:
                    raise ValueError(
                        f"reporting_arrays['{key}'] shape {arr.shape} must match objective_array shape {objective_array.shape}."
                    )
                coerced[key] = arr
            reporting_arrays = coerced

        if regularizer is not None and not callable(regularizer):
            raise TypeError("regularizer must be None or callable.")

        af_grid = np.broadcast_to(adjustment_factors, (N, K))

        self.input = SimpleNamespace(
            **decision,
            adjustment_factors=adjustment_factors,
            af_grid=af_grid,
            objective_array=objective_array,
            constraint_array=constraint_array,
            sample_weights=sample_weights,
            reporting_arrays=reporting_arrays,
            regularizer=regularizer,
        )
        self.output = SimpleNamespace(
            optimization_scoring_history=pd.DataFrame({
                "target_constraint": [],
                "constraint": [],
                "objective": [],
                "price_multiplier": [],
            })
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline,
        layer_name,
        scored_data,
        adjustment_factors,
        objective_array,
        constraint_array,
        reporting_arrays=None,
        sample_weights=None,
        regularizer=None,
    ):
        """Build an optimizer with ``rating_factors``/``initial_multipliers``
        derived from a ``RatingTablesLayer`` registered on ``pipeline``.

        Each rating table becomes one rating factor; its key column(s) define the
        categories (folded into a single label for multi-column joins). Decision
        variables are the table's rows, seeded from their current ``value``; data
        rows whose key combination is absent from a table take that table's
        ``default_values`` entry (fixed, not optimized).

        Parameters
        ----------
        pipeline : PyfuncPricingPipeline
            Pipeline holding the layer.
        layer_name : str
            Name of a multiplicative ``RatingTablesLayer`` (``formula`` omitted, or
            a pure product of all tables' ``value`` columns).
        scored_data : dict[str, pd.DataFrame]
            Output of ``pipeline.predict(...)`` — must contain the layer's
            ``target_table`` with the tables' key columns.
        adjustment_factors, objective_array, constraint_array, reporting_arrays, sample_weights, regularizer
            Passed through to ``__init__`` (precompute these as today, e.g. via
            ``simulate_adjustment_grid``).
        """
        parsed = cls._parse_rating_layer(pipeline, layer_name, scored_data)
        obj = cls(
            adjustment_factors=adjustment_factors,
            objective_array=objective_array,
            constraint_array=constraint_array,
            rating_factors=parsed["rating_factors"],
            initial_multipliers=parsed["initial_multipliers"],
            sample_weights=sample_weights,
            reporting_arrays=reporting_arrays,
            regularizer=regularizer,
            default_multipliers=parsed["default_multipliers"],
        )
        obj._layer_schema = parsed["layer_schema"]
        obj._layer_meta = parsed["layer_meta"]
        return obj

    def _interp_at_eff(self, array_2d, eff):
        return _fast_linear_interpolator(self.input.af_grid, array_2d, eff)

    def _objective_fn(self, x):
        eff = self._effective_multiplier(x)
        row_obj = self._interp_at_eff(self.input.objective_array, eff)
        score = np.average(row_obj, weights=self.input.sample_weights)
        if self.input.regularizer is not None:
            score = score - float(self.input.regularizer(self._unflatten(x)))
        return -score

    def _constraint_value(self, x):
        eff = self._effective_multiplier(x)
        row_con = self._interp_at_eff(self.input.constraint_array, eff)
        return np.average(row_con, weights=self.input.sample_weights)

    _SUPPORTED_METHODS = ("SLSQP", "L-BFGS-B", "Nelder-Mead")

    def optimize(
        self,
        target_constraint,
        bounds=(0.8, 1.2),
        method="SLSQP",
        constraint_penalty_weight=1000.0,
        optimizer_options=None,
        save_to_log=True,
        slsqp_options=None,
    ):
        if not isinstance(target_constraint, (int, float)) or isinstance(target_constraint, bool):
            raise TypeError("target_constraint must be a numeric value.")
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {list(self._SUPPORTED_METHODS)}; got {method!r}."
            )
        if not isinstance(constraint_penalty_weight, (int, float)) or isinstance(constraint_penalty_weight, bool):
            raise TypeError("constraint_penalty_weight must be a numeric value.")
        if constraint_penalty_weight < 0:
            raise ValueError("constraint_penalty_weight must be non-negative.")

        if slsqp_options is not None:
            warnings.warn(
                "`slsqp_options` is deprecated; use `optimizer_options` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if optimizer_options is None:
                optimizer_options = slsqp_options
        if optimizer_options is not None and not isinstance(optimizer_options, dict):
            raise TypeError("optimizer_options must be None or a dict.")

        flat_bounds = self._build_bounds(bounds)
        sample_weights = self.input.sample_weights

        if method == "SLSQP":
            constraints = [{
                "type": "eq",
                "fun": lambda x, t=target_constraint: self._constraint_value(x) - t,
            }]
            fn = self._objective_fn
        else:
            constraints = ()
            weight = float(constraint_penalty_weight)
            fn = (
                lambda x, t=target_constraint, w=weight: (
                    self._objective_fn(x) + w * (self._constraint_value(x) - t) ** 2
                )
            )

        result = minimize(
            fn,
            x0=self.input.initial_flat,
            method=method,
            bounds=flat_bounds,
            constraints=constraints,
            options=optimizer_options or {},
        )

        if not result.success:
            warnings.warn(f"{method} did not converge: {result.message}")

        x_opt = result.x
        optimal_multipliers = self._unflatten(x_opt)
        eff = self._effective_multiplier(x_opt)

        row_objective = self._interp_at_eff(self.input.objective_array, eff)
        row_constraint = self._interp_at_eff(self.input.constraint_array, eff)
        selected_reporting = {
            key: self._interp_at_eff(arr, eff) for key, arr in self.input.reporting_arrays.items()
        }

        avg_objective = np.average(row_objective, weights=sample_weights)
        avg_constraint = np.average(row_constraint, weights=sample_weights)
        price_multiplier_avg = np.average(eff, weights=sample_weights)

        per_factor_chosen = {}
        for factor in self.input.factor_names:
            offset = self.input.param_offsets[factor]
            n_cats = len(self.input.category_orders[factor])
            slice_ = x_opt[offset:offset + n_cats]
            per_factor_chosen[f"{factor}_multiplier"] = self._factor_contribution(factor, slice_)

        output_df = pd.DataFrame({
            "effective_multiplier": eff,
            "objective": row_objective,
            "constraint": row_constraint,
            "sample_weights": sample_weights,
            **selected_reporting,
            **per_factor_chosen,
        })

        summary_data = {
            "target_constraint": [target_constraint],
            "constraint": [avg_constraint],
            "objective": [avg_objective],
            "price_multiplier": [price_multiplier_avg],
        }
        for key, per_row in selected_reporting.items():
            summary_data[key] = [np.average(per_row, weights=sample_weights)]
        summary_data["method"] = [method]
        summary_data["constraint_gap"] = [avg_constraint - target_constraint]
        summary_data["result_success"] = [bool(result.success)]
        summary_data["result_message"] = [str(result.message)]
        summary_data["result_nit"] = [int(getattr(result, "nit", -1))]
        summary = pd.DataFrame(summary_data)

        if save_to_log:
            self.output.optimal_array = x_opt
            self.output.optimal_multipliers = optimal_multipliers
            self.output.output_optimized = output_df
            self.output.optimization_scoring_history = pd.concat([
                self.output.optimization_scoring_history,
                pd.DataFrame({
                    "target_constraint": [target_constraint],
                    "constraint": [avg_constraint],
                    "objective": [avg_objective],
                    "price_multiplier": [price_multiplier_avg],
                }),
            ])

        return {
            "optimal_multipliers": optimal_multipliers,
            "output_dataframe": output_df,
            "summary": summary,
            "result": result,
        }

    def calculate_efficient_frontier(
        self,
        target_constraint_range,
        bounds=(0.8, 1.2),
        method="SLSQP",
        constraint_penalty_weight=1000.0,
        optimizer_options=None,
    ):
        if not isinstance(target_constraint_range, (list, tuple, np.ndarray)):
            raise TypeError("target_constraint_range must be a list, tuple, or NumPy array.")
        if len(target_constraint_range) == 0:
            raise ValueError("target_constraint_range cannot be empty.")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target_constraint_range):
            raise TypeError("All values in target_constraint_range must be numeric.")

        summary_rows = []
        for target in tqdm(target_constraint_range, desc="Optimising rating factors", total=len(target_constraint_range)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.optimize(
                        target_constraint=target,
                        bounds=bounds,
                        method=method,
                        constraint_penalty_weight=constraint_penalty_weight,
                        optimizer_options=optimizer_options,
                        save_to_log=False,
                    )
                if result["summary"]["result_success"].iloc[0]:
                    summary_rows.append(result["summary"])
                else:
                    print(f"Warning: {method} did not converge at target {target:.4f}.")
            except Exception as exc:
                print(f"Warning: optimization failed at target {target:.4f}: {exc}")

        if not summary_rows:
            raise RuntimeError("No valid results were produced. All targets may have failed.")

        self.output.efficient_frontier = pd.concat(summary_rows, ignore_index=True)
        return self.output.efficient_frontier
