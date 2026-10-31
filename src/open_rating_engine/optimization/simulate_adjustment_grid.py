from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # the notebook widget in Jupyter, a console bar anywhere else

__all__ = ["simulate_adjustment_grid", "make_demand_emulation"]


def simulate_adjustment_grid(
    data,
    pipeline,
    premium_emulation,
    output_table: str,
    output_columns: dict,
    adjustment_factor=None,
    parallel_n_jobs=1,
):
    """
    Score a ``PyfuncPricingPipeline`` across a grid of adjustment factors and
    return per-record arrays for any columns the caller asks for.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Pipeline input — a dict mapping table names to DataFrames.
    pipeline : PyfuncPricingPipeline
        Pre-built pipeline whose ``predict`` produces the requested columns
        on ``output_table``.
    premium_emulation : callable
        ``(data_dict, pipeline, factor) -> dict[str, pd.DataFrame]``. The
        caller decides how to inject the adjustment factor (e.g. setting an
        override column on ``risk`` or scaling ``risk.previous_gwp``) and
        returns the modified input dict to be scored.
    output_table : str
        Table in the predict output to read metrics from.
    output_columns : dict[str, str]
        Mapping from output-array name to column name on ``output_table``.
        Each entry produces one ``(N, K)`` array in the result dict under the
        given key. Example::

            {
                "offered_gwp_array":         "risk.gwp",
                "expected_conversion_array": "risk.conversion_rate",
                "offered_profit_array":      "risk.reporting_offered_profit",
            }
    adjustment_factor : array-like of float, optional
        1-D array of positive adjustment factors. Defaults to
        ``np.arange(0.2, 2.01, 0.05)``.
    parallel_n_jobs : int
        Threads used to score factors concurrently. ``-1`` uses all cores.

    Returns
    -------
    dict
        ``{"adjustment_factor_array": (K,), **{key: (N, K) for key in output_columns}}``
        where N is the number of rows in ``data[output_table]`` and K is
        ``len(adjustment_factor)``.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict[str, pd.DataFrame]")

    if output_table not in data:
        raise KeyError(f"output_table '{output_table}' not found in data; got keys {list(data)}")

    if not isinstance(data[output_table], pd.DataFrame):
        raise TypeError(f"data['{output_table}'] must be a pandas DataFrame")

    if data[output_table].empty:
        raise ValueError(f"data['{output_table}'] is empty")

    if not hasattr(pipeline, "predict"):
        raise TypeError("pipeline must be a PyfuncPricingPipeline (no .predict method found)")

    if not callable(premium_emulation):
        raise TypeError("premium_emulation must be callable")

    if not isinstance(output_columns, dict) or not output_columns:
        raise TypeError("output_columns must be a non-empty dict mapping array name -> column name")

    if "adjustment_factor_array" in output_columns:
        raise ValueError("'adjustment_factor_array' is reserved and cannot be used as an output_columns key")

    if adjustment_factor is None:
        adjustment_factor = np.arange(0.2, 2.01, 0.05)
    elif isinstance(adjustment_factor, list):
        adjustment_factor = np.array(adjustment_factor)
    elif not isinstance(adjustment_factor, np.ndarray):
        raise TypeError("adjustment_factor must be None, a list, or a numpy array")

    if adjustment_factor.ndim != 1:
        raise ValueError("adjustment_factor must be a 1-dimensional array")

    if np.any(adjustment_factor <= 0):
        raise ValueError("All adjustment_factor values must be positive numbers")

    if not isinstance(parallel_n_jobs, int):
        raise TypeError("parallel_n_jobs must be an integer")

    column_names = list(output_columns.values())

    def simulate_factor(factor):
        pipeline_copy = deepcopy(pipeline)
        data_for_scoring = premium_emulation(data, pipeline_copy, factor)
        predicted = pipeline_copy.predict(data_for_scoring)
        out_df = predicted[output_table]
        return tuple(out_df[col].values for col in column_names)

    max_workers = parallel_n_jobs if parallel_n_jobs > 0 else None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_factor, f) for f in adjustment_factor]
        for future in tqdm(
            as_completed(futures),
            desc="scoring adjustment factors",
            total=len(futures),
        ):
            # Advance the bar as each factor finishes; surface errors eagerly.
            future.result()
        results = [future.result() for future in futures]

    columns_by_index = list(zip(*results))

    result = {"adjustment_factor_array": adjustment_factor}
    for (array_name, _col_name), column_results in zip(output_columns.items(), columns_by_index):
        result[array_name] = np.stack(column_results, axis=1)

    return result


def make_demand_emulation(layer_name, table_key, mode="scale"):
    """Build a ``premium_emulation`` callback for :func:`simulate_adjustment_grid`.

    The returned callback drives one rating table inside an existing
    ``RatingTablesLayer`` by the swept adjustment factor, so you can trace how
    ``conversion_rate``/``expected_profit`` respond to a uniform demand shift
    without rebuilding the pipeline on every factor.

    Parameters
    ----------
    layer_name : str
        Name of the ``RatingTablesLayer`` already registered on the pipeline
        (e.g. ``"risk.demand_multiplier"``).
    table_key : str
        Key into that layer's ``rating_tables`` to drive (e.g. ``"risk_age_band"``).
    mode : {"scale", "override"}
        - ``"override"``: set that table's ``value`` column **and** its
          ``default_value`` to the factor. The output of that term equals the
          factor (use a single-table demand layer for a pure
          "output == adjustment_factor" elasticity curve).
        - ``"scale"``: multiply that table's ``value`` column **and** its
          ``default_value`` by the factor, scaling that term — and hence the
          layer output — by the factor on top of the existing rating structure.

    Returns
    -------
    callable
        ``(data, pipeline, factor) -> dict[str, pd.DataFrame]`` suitable as the
        ``premium_emulation`` argument to :func:`simulate_adjustment_grid`. It
        returns ``data`` unchanged: the adjustment lives in the mutated
        pipeline, which :func:`simulate_adjustment_grid` then scores once.

    Notes
    -----
    The callback mutates the layer in place. This is safe because
    :func:`simulate_adjustment_grid` deepcopies the pipeline before each call,
    so every factor (and every thread) starts from the unmodified baseline.
    """
    if mode not in ("override", "scale"):
        raise ValueError("mode must be 'override' or 'scale'")

    def premium_emulation(data, pipeline, factor):
        if layer_name not in pipeline.layers:
            raise KeyError(f"layer '{layer_name}' not found in pipeline")
        layer = pipeline.layers[layer_name]
        if table_key not in getattr(layer, "rating_tables", {}):
            raise KeyError(
                f"table_key '{table_key}' not in layer '{layer_name}'.rating_tables"
            )
        factor = float(factor)
        tbl = layer.rating_tables[table_key]
        if mode == "override":
            tbl["value"] = factor
            layer.default_values[table_key] = factor
        else:  # scale
            tbl["value"] = tbl["value"].to_numpy() * factor
            layer.default_values[table_key] = layer.default_values[table_key] * factor
        return data

    return premium_emulation
