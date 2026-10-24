"""Thermometer (cumulative) encoding — the shared train/score binning contract.

Covers the pure ``rating_engine.binning`` helpers (the single source of truth used
by a host application at fit time *and* by ``H2OScoringParameter`` at score time) and
the ``bin_thermometer`` branch of ``H2OScoringParameter._apply_preprocessing``.
No JVM / MOJO — just the encoding.
"""
import numpy as np
import pandas as pd
import pytest

from open_rating_engine.rating_engine.binning import (
    apply_thermometer_encoding,
    n_bins_from_edges,
    thermometer_column_names,
)
from open_rating_engine.rating_engine.pricing_layers import H2OScoringParameter

EDGES = [0, 5, 10, 20]  # 3 in-range bins → 2 indicator columns


def test_column_names_and_bin_count():
    assert n_bins_from_edges(EDGES) == 3
    assert thermometer_column_names("age", EDGES) == ["age__ge_1", "age__ge_2"]


def test_fewer_than_two_bins_has_no_indicators():
    assert thermometer_column_names("age", [0, 10]) == []


def test_encoding_in_range_bins_are_cumulative():
    df = pd.DataFrame({"age": [3, 7, 15]})  # bin0, bin1, bin2
    out = apply_thermometer_encoding(df, "age", EDGES)
    # bin0 → (0,0); bin1 → (1,0); bin2 → (1,1)
    assert out["age__ge_1"].tolist() == [0.0, 1.0, 1.0]
    assert out["age__ge_2"].tolist() == [0.0, 0.0, 1.0]


def test_out_of_range_and_na_map_to_reference():
    df = pd.DataFrame({"age": [25, -1, np.nan, "junk"]})  # above, below, NA, non-numeric
    out = apply_thermometer_encoding(df, "age", EDGES)
    assert out["age__ge_1"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert out["age__ge_2"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_drop_source_removes_original_column():
    df = pd.DataFrame({"age": [3, 7]})
    out = apply_thermometer_encoding(df, "age", EDGES, drop_source=True)
    assert "age" not in out.columns
    assert {"age__ge_1", "age__ge_2"} <= set(out.columns)


def _thermometer_layer():
    return H2OScoringParameter(
        name="score", target_table="risk", output_column="prediction",
        column_map={"age": "age"},
        mojo_artifact_key="m", genmodel_artifact_key="g",
        preprocessing=[{"type": "bin_thermometer", "column": "age", "edges": EDGES}],
    )


def test_apply_preprocessing_bin_thermometer_regenerates_indicators():
    """The score-time recipe must produce the same indicator columns the MOJO saw
    at train time, so the raw column is expanded into ``age__ge_k``."""
    layer = _thermometer_layer()
    out = layer._apply_preprocessing(pd.DataFrame({"age": [3, 7, 15, 25]}))
    assert out["age__ge_1"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert out["age__ge_2"].tolist() == [0.0, 0.0, 1.0, 0.0]
