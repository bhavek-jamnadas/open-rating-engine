"""Smoke tests: the package and every submodule import without optional extras.

These guard against regressions in the lazy optional imports — none of these
imports should require the ``h2o``, ``onnx`` or ``viz`` extras, or a
running JVM.
"""

import importlib
import pkgutil

import pytest

import open_rating_engine


def _all_submodules():
    mods = []
    for m in pkgutil.walk_packages(open_rating_engine.__path__, "open_rating_engine."):
        name = m.name
        mods.append(name)
    return mods


SUBMODULES = _all_submodules()

# Top-level names of optional pip extras. Modules that import these at module
# level legitimately fail to import without the extra installed, so skip rather
# than fail for them. (Today every one of them is imported lazily.)
OPTIONAL_EXTRA_MODULES = {
    "h2o",
    "jpype",
    "onnxruntime",
    "plotly",
    "networkx",
    "ipywidgets",
}


def test_top_level_import_has_core_symbols():
    for symbol in (
        "PyfuncPricingPipeline",
        "ColumnCalculation",
        "TableJoin",
        "RatingTablesLayer",
        "apply_aging",
    ):
        assert hasattr(open_rating_engine, symbol), f"missing export: {symbol}"


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_submodule_imports(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in OPTIONAL_EXTRA_MODULES:
            pytest.skip(f"optional extra not installed: {missing}")
        raise


def test_layer_registry_complete():
    from open_rating_engine.rating_engine.layer_registry import LAYER_REGISTRY

    assert set(LAYER_REGISTRY) == {
        "ColumnCalculation",
        "TableTransform",
        "TableAggregation",
        "TableJoin",
        "RenameColumns",
        "DropColumns",
        "TableDrop",
        "RatingTablesLayer",
        "CustomLayer",
        "H2OScoringParameter",
        "OnnxScoring",
        "GlmScoring",
        "SourceColumn",
    }

