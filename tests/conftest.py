"""Shared fixtures for the open_rating_engine test suite.

No test here should touch the network, S3, a JVM, or an MLflow server — the
fixtures below provide in-memory stand-ins so the whole suite runs offline.
"""

import numpy as np
import pandas as pd
import pytest
from mlflow.pyfunc import PythonModelContext

from tests.data.sample_data import make_data, make_rating_tables


@pytest.fixture
def mock_context():
    """A minimal MLflow ``PythonModelContext`` (local environment)."""
    return PythonModelContext(
        artifacts={
            "model_cost": "/tmp/model_cost.zip",
            "h2o_gen_model": "/tmp/h2o-genmodel.jar",
        },
        model_config={"environment": "local", "java_options": "-Xmx1g"},
    )


@pytest.fixture
def sample_data():
    """Fresh copy of the synthetic policy/risk/cover/claims dict per test."""
    return make_data()


@pytest.fixture
def sample_rating_tables():
    return make_rating_tables()


@pytest.fixture
def pipeline(mock_context):
    """An empty pipeline wired to the mock context, progress bar disabled."""
    from open_rating_engine.rating_engine import PyfuncPricingPipeline

    p = PyfuncPricingPipeline(mock_context)
    p.options["show_progress"] = False
    return p


@pytest.fixture(autouse=True)
def _block_jvm(monkeypatch):
    """Fail loudly if any test accidentally tries to start a real JVM."""
    import jpype

    def _boom(*args, **kwargs):  # pragma: no cover - only hit on misuse
        raise AssertionError("A test attempted to start a real JVM")

    monkeypatch.setattr(jpype, "startJVM", _boom, raising=False)
    yield
