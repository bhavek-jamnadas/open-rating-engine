from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

__all__ = [
    "PriceOptimizerIndividual",
]


_RESERVED_SUMMARY_COLS = ("target_constraint", "constraint", "objective", "price_multiplier")


class PriceOptimizerIndividual:
    """
    Per-record price optimiser. The caller supplies the objective array they
    want maximised (with any penalty terms already baked in) and a constraint
    array (typically conversion rate). For each row, the optimiser picks the
    adjustment factor that maximises ``objective - λ * constraint``, where λ
    is found by bisection so that the sample-weighted average of the chosen
    constraint values matches ``target_constraint``.
    """

    def __init__(
        self,
        adjustment_factors,
        objective_array,
        constraint_array,
        sample_weights=None,
        reporting_arrays=None,
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

        self.input = SimpleNamespace(
            adjustment_factors=adjustment_factors,
            objective_array=objective_array,
            constraint_array=constraint_array,
            sample_weights=sample_weights,
            reporting_arrays=reporting_arrays,
        )
        self.output = SimpleNamespace(
            optimization_scoring_history=pd.DataFrame({
                "target_constraint": [],
                "constraint": [],
                "objective": [],
                "price_multiplier": [],
            })
        )

    @staticmethod
    def _coerce_1d(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array.")
        return arr

    @staticmethod
    def _coerce_2d(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array.")
        return arr

    def optimize(
        self,
        target_constraint,
        tol=1e-4,
        max_iter=50,
        lagrangian_lambda=(-1e3, 1e3),
        save_to_log=True,
    ):
        """
        Lagrangian bisection: pick one adjustment factor per row to maximise
        ``objective_array - λ * constraint_array``, with λ chosen so the
        sample-weighted average of the selected constraint values equals
        ``target_constraint``.
        """
        if not isinstance(target_constraint, (int, float)) or isinstance(target_constraint, bool):
            raise TypeError("target_constraint must be a numeric value.")

        adjustment_factors = self.input.adjustment_factors
        objective_array = self.input.objective_array
        constraint_array = self.input.constraint_array
        sample_weights = self.input.sample_weights
        reporting_arrays = self.input.reporting_arrays

        N = objective_array.shape[0]
        row_idx = np.arange(N)
        lam_low, lam_high = lagrangian_lambda

        for _ in range(max_iter):
            lam = (lam_low + lam_high) / 2
            score_matrix = objective_array - lam * constraint_array
            best_indices = np.argmax(score_matrix, axis=1)

            selected_constraint = constraint_array[row_idx, best_indices]
            selected_objective = objective_array[row_idx, best_indices]
            selected_multipliers = adjustment_factors[best_indices]

            avg_constraint = np.average(selected_constraint, weights=sample_weights)

            if abs(avg_constraint - target_constraint) <= tol:
                avg_objective = np.average(selected_objective, weights=sample_weights)
                price_multiplier_avg = np.average(selected_multipliers, weights=sample_weights)

                selected_reporting = {
                    key: arr[row_idx, best_indices] for key, arr in reporting_arrays.items()
                }

                output_df = pd.DataFrame({
                    "adjustment_factor": selected_multipliers,
                    "objective": selected_objective,
                    "constraint": selected_constraint,
                    "sample_weights": sample_weights,
                    **selected_reporting,
                })

                summary_data = {
                    "target_constraint": [target_constraint],
                    "constraint": [avg_constraint],
                    "objective": [avg_objective],
                    "price_multiplier": [price_multiplier_avg],
                }
                for key, per_row in selected_reporting.items():
                    summary_data[key] = [np.average(per_row, weights=sample_weights)]
                summary = pd.DataFrame(summary_data)

                if save_to_log:
                    self.output.optimal_array = selected_multipliers
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
                    "optimal_multipliers": selected_multipliers,
                    "output_dataframe": output_df,
                    "summary": summary,
                }

            if avg_constraint < target_constraint:
                lam_high = lam
            else:
                lam_low = lam

        raise ValueError("No solution found within tolerance after max iterations.")

    def calculate_efficient_frontier(
        self,
        target_constraint_range,
        tol=1e-4,
        max_iter=50,
        lagrangian_lambda=(-1e3, 1e3),
    ):
        """
        Sweep ``target_constraint_range`` through :meth:`optimize` and
        concatenate the resulting summary rows into an efficient-frontier
        DataFrame.
        """
        if not isinstance(target_constraint_range, (list, tuple, np.ndarray)):
            raise TypeError("target_constraint_range must be a list, tuple, or NumPy array.")
        if len(target_constraint_range) == 0:
            raise ValueError("target_constraint_range cannot be empty.")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target_constraint_range):
            raise TypeError("All values in target_constraint_range must be numeric.")

        summary_rows = []
        for target in tqdm(target_constraint_range, desc="Optimising portfolios", total=len(target_constraint_range)):
            try:
                result = self.optimize(
                    target_constraint=target,
                    tol=tol,
                    max_iter=max_iter,
                    lagrangian_lambda=lagrangian_lambda,
                    save_to_log=False,
                )
                summary_rows.append(result["summary"])
            except ValueError:
                print(f"Warning: Target {target:.4f} not achievable within tolerance.")

        if not summary_rows:
            raise RuntimeError("No valid results were produced. All targets may have failed.")

        self.output.efficient_frontier = pd.concat(summary_rows, ignore_index=True)
        return self.output.efficient_frontier
