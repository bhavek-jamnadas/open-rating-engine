"""Smoke tests: the package and every submodule import without optional extras.

These guard against regressions in the lazy optional imports — none of these
imports should require the ``azure``, ``h2o``, ``onnx`` or ``viz`` extras, or a
running JVM.
"""

import importlib
import importlib.util
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
    "azure",
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


def test_azure_import_is_lazy_not_at_module_level():
    """``download_blob_to_temp`` must raise a clear error, not ImportError at
    module import, when the optional azure extra is absent."""
    from open_rating_engine.rating_engine import pricing_pipeline

    try:
        installed = importlib.util.find_spec("azure.storage.blob") is not None
    except ModuleNotFoundError:
        installed = False
    if installed:
        pytest.skip("azure extra is installed in this environment")

    with pytest.raises(ImportError, match="azure"):
        pricing_pipeline.download_blob_to_temp(
            "https://acct.blob.core.windows.net/c/blob.zip", conn_str="x"
        )
