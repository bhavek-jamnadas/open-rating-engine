"""Lightweight fakes so H2O / JPype never need to start a JVM in tests.

``H2OScoringParameter`` ultimately calls ``_load_model`` (JPype + the ``hex``
genmodel classes) and ``_mojo_predict_pandas`` (per-row scoring through the
wrapper). We patch those two seams: a sentinel wrapper object and a pure
pandas implementation of the prediction frame.
"""

import numpy as np
import pandas as pd

SENTINEL_WRAPPER = object()


def fake_load_model(mojo_zip_path: str, genmodel_jar_path: str):
    """Stand-in for ``H2OScoringParameter._load_model`` — no JVM."""
    return SENTINEL_WRAPPER


def make_fake_mojo_predict_pandas(model_type: str):
    """Return a function matching ``_mojo_predict_pandas``'s signature.

    Produces a deterministic predictions frame whose column layout matches
    what ``_score`` expects for each model type.
    """

    def _fake(dataframe: pd.DataFrame, wrapper, model_type=model_type, n_threads=4):
        n = len(dataframe)
        # Deterministic pseudo-predictions derived from row position.
        base = np.linspace(0.1, 0.9, num=n) if n else np.array([])
        if model_type == "regression":
            return pd.DataFrame({"predict": base * 1000.0}, index=dataframe.index)
        if model_type == "binomial":
            return pd.DataFrame(
                {"predict": (base > 0.5).astype(int), "p0": 1 - base, "p1": base},
                index=dataframe.index,
            )
        # multinomial: predict + p0..p2
        return pd.DataFrame(
            {
                "predict": np.zeros(n, dtype=int),
                "p0": base,
                "p1": 1 - base,
                "p2": np.zeros(n),
            },
            index=dataframe.index,
        )

    return _fake
