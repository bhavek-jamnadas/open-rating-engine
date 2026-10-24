"""Thermometer (cumulative) encoding for monotone-across-bins predictors.

A binned numeric predictor entered into a GLM as a plain factor cannot have a
monotone ordering enforced through H2O ``beta_constraints`` (box bounds act on
individual coefficients, not on differences between adjacent bins). The standard
fix is **thermometer / cumulative encoding**: replace the K-bin factor with
``K-1`` cumulative indicators

    {name}__ge_1, {name}__ge_2, ..., {name}__ge_{K-1}

where ``{name}__ge_k = 1`` iff the value falls in bin ``k`` or higher (bins are
ordered ascending by the cut edges). The linear predictor for bin ``j`` is then
``sum_{k<=j} beta_k`` — a step function — so a monotone-increasing shape is
exactly ``beta_k >= 0`` for every increment (a per-coefficient sign/box
constraint), and decreasing is ``beta_k <= 0``.

This module is the single source of truth for the encoding so that **training**
(a host application's fit pipeline, which imports these functions) and **scoring**
(the ``H2OScoringParameter`` MOJO preprocessing path below) produce byte-identical
columns. Out-of-range / NA values get all indicators ``0`` → they score as the
reference (lowest) bin, matching the closed-form ``GlmScoring`` path where an
out-of-range bin label falls back to the baseline contribution of ``0``.

Pure pandas/numpy — no ``h2o`` import — so it stays light to import from either
repo and trivially unit-testable.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Suffix for the k-th cumulative indicator. Double underscore keeps it clear of
# real column names (same convention as the fit pipeline's synthetic columns).
THERMOMETER_SUFFIX = "__ge_"


def n_bins_from_edges(edges) -> int:
    """Number of in-range bins for ``edges`` (``len(edges) - 1``)."""
    return max(0, len(list(edges)) - 1)


def thermometer_column_names(name: str, edges) -> List[str]:
    """Cumulative-indicator column names for ``name`` given its cut ``edges``.

    Returns ``K-1`` names (``{name}__ge_1`` … ``{name}__ge_{K-1}``) for ``K``
    in-range bins; an empty list when there are fewer than two bins (nothing to
    order).
    """
    n_bins = n_bins_from_edges(edges)
    return [f"{name}{THERMOMETER_SUFFIX}{k}" for k in range(1, n_bins)]


def apply_thermometer_encoding(
    df: pd.DataFrame, name: str, edges, *, drop_source: bool = False
) -> pd.DataFrame:
    """Add cumulative-indicator columns for ``name`` to ``df`` (in place).

    Mirrors fit-time binning's ``pd.cut(numeric, bins=edges, include_lowest=True)``
    so bin codes line up with the labels used elsewhere. Each ``{name}__ge_k`` is
    ``1.0`` when the row's bin index is ``>= k`` and ``0.0`` otherwise; rows whose
    value is out-of-range, non-numeric or NA get every indicator ``0.0`` (they
    behave as the reference bin). Returns ``df`` for chaining.

    ``drop_source`` removes the original ``name`` column after encoding; the
    default keeps it (the fit pipeline still wants it as a bin label for
    diagnostics, and the MOJO simply ignores the unused raw column at score time).
    """
    names = thermometer_column_names(name, edges)
    if not names:
        return df
    numeric = pd.to_numeric(df[name], errors="coerce")
    # ``cat.codes`` is 0..K-1 for in-range bins (ascending) and -1 for NA / OOR.
    codes = pd.cut(numeric, bins=list(edges), include_lowest=True).cat.codes.to_numpy()
    for k, col in enumerate(names, start=1):
        df[col] = (codes >= k).astype("float64")
    if drop_source:
        df = df.drop(columns=[name])
    return df
