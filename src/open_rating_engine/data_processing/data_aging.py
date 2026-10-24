"""Aging preprocessor — applies SourceColumn aging rules to input frames.

One step forward in time per call. A single-cycle score path invokes this
once between ``input_frames = {table: pd.DataFrame(...)}`` construction
and ``pipeline.predict(...)``. A multi-year driver wraps the predict in a
loop and calls this once per cycle.

The function is pure (no DB, no I/O) and operates on dicts that the
caller has already resolved into ``layer_configs`` and ``input_frames``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "apply_aging",
    "AgingRuleError",
]


SOURCE_COLUMN_LAYER_TYPE = "SourceColumn"


def apply_aging(
    layer_configs: dict[str, dict],
    input_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Mutate ``input_frames`` in place by applying each enabled SourceColumn
    aging rule. Returns the same dict for chaining.

    For every layer config of type ``SourceColumn`` with an
    ``aging_rule.enabled == True`` and a non-empty ``expression``, the
    expression is evaluated with ``{df, np, pd}`` in scope where ``df`` is
    ``input_frames[source_table]``. The result is assigned to
    ``df[source_column]`` in place. Layers without a rule, with the rule
    disabled, or whose ``source_table`` is absent from ``input_frames`` are
    silently skipped.

    Order of evaluation within a single call is dict-insertion order.
    Cross-column reads between aged columns are undefined within a cycle —
    use a ``ColumnCalculation`` layer if you need ordering guarantees.
    """
    for layer_name, cfg in layer_configs.items():
        if cfg.get("type") != SOURCE_COLUMN_LAYER_TYPE:
            continue
        rule = cfg.get("aging_rule")
        if not isinstance(rule, dict):
            continue
        if not rule.get("enabled", True):
            continue
        expression = rule.get("expression")
        if not isinstance(expression, str) or not expression:
            continue

        table = cfg.get("source_table")
        column = cfg.get("source_column")
        if not isinstance(table, str) or not table:
            continue
        if not isinstance(column, str) or not column:
            continue
        if table not in input_frames:
            continue

        df = input_frames[table]
        try:
            result = eval(expression, {"df": df, "np": np, "pd": pd})
        except Exception as exc:
            raise AgingRuleError(
                f"aging rule on '{table}.{column}' raised: {exc}"
            ) from exc

        if not isinstance(result, pd.Series):
            try:
                result = pd.Series(result, index=df.index)
            except Exception as exc:
                raise AgingRuleError(
                    f"aging rule on '{table}.{column}' returned "
                    f"{type(result).__name__}; expected pandas Series or "
                    f"a scalar broadcastable to one ({exc})"
                ) from exc

        if len(result) != len(df):
            raise AgingRuleError(
                f"aging rule on '{table}.{column}' returned a Series of "
                f"length {len(result)}; expected {len(df)} (matching df)"
            )

        df[column] = result

    return input_frames


class AgingRuleError(ValueError):
    """Raised when an aging expression fails to evaluate or returns an
    incompatible shape."""
