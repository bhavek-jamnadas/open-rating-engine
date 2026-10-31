"""Guards for the package's public API and its lazy loading.

Package ``__init__`` modules re-export names lazily (``open_rating_engine._lazy``),
each module declares ``__all__``, and importing the package must stay cheap.
"""

import ast
import importlib
import os
import pathlib
import subprocess
import sys

import pytest

import open_rating_engine

PACKAGES = [
    "open_rating_engine",
    "open_rating_engine.rating_engine",
    "open_rating_engine.optimization",
    "open_rating_engine.data_processing",
]

MODULES = [
    "open_rating_engine.rating_engine.pricing_pipeline",
    "open_rating_engine.rating_engine.pricing_layers",
    "open_rating_engine.rating_engine.binning",
    "open_rating_engine.rating_engine.layer_registry",
    "open_rating_engine.data_processing.data_aging",
    "open_rating_engine.optimization.individual_price_optimizer",
    "open_rating_engine.optimization.rating_factor_price_optimizer",
    "open_rating_engine.optimization.multi_period_individual_price_optimizer",
    "open_rating_engine.optimization.multi_period_rating_factor_price_optimizer",
    "open_rating_engine.optimization.simulate_adjustment_grid",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exported_name_resolves(package):
    pkg = importlib.import_module(package)
    for name in pkg.__all__:
        assert getattr(pkg, name) is not None, f"{package}.{name}"


@pytest.mark.parametrize("package", PACKAGES)
def test_dir_lists_the_public_api(package):
    pkg = importlib.import_module(package)
    assert set(pkg.__all__) - {"__version__"} <= set(dir(pkg))


@pytest.mark.parametrize("package", PACKAGES)
def test_unknown_attribute_raises(package):
    pkg = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no_such_name"):
        pkg.no_such_name


@pytest.mark.parametrize("package", PACKAGES)
def test_reexports_are_the_defining_objects(package):
    """A re-export is the object from the defining module, not a copy."""
    pkg = importlib.import_module(package)
    for name in pkg.__all__:
        value = getattr(pkg, name)
        if callable(value):
            assert getattr(importlib.import_module(value.__module__), name) is value


@pytest.mark.parametrize(
    "code",
    [
        # The sibling is resolved first, and its __getattr__ loads the shared submodule.
        "from open_rating_engine.optimization import make_demand_emulation, simulate_adjustment_grid",
        # A direct submodule import before the package re-export is read.
        "import open_rating_engine.optimization.simulate_adjustment_grid\n"
        "from open_rating_engine.optimization import simulate_adjustment_grid",
    ],
)
def test_a_name_shared_with_its_submodule_is_the_function_in_any_import_order(code):
    """`simulate_adjustment_grid` is both a function and the module defining it.

    Importing the module binds it on the package under that name, after which the lazy
    __getattr__ never runs -- so the asterisk-core worker received the MODULE and failed
    with "'module' object is not callable". The checks above cannot see it: a module is
    not None, and `test_reexports_are_the_defining_objects` skips anything not callable.
    Run in a subprocess because the outcome depends on what this process imported first.
    """
    out = subprocess.run(
        [sys.executable, "-c", f"{code}\nprint(type(simulate_adjustment_grid).__name__)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert out.stdout.strip() == "function"


@pytest.mark.parametrize("module", MODULES)
def test_module_declares_all(module):
    mod = importlib.import_module(module)
    assert hasattr(mod, "__all__"), f"{module} has no __all__"
    for name in mod.__all__:
        assert hasattr(mod, name), f"{module}.__all__ lists missing name {name!r}"


def _type_checking_names(init_path):
    """Names bound under ``if TYPE_CHECKING:`` in an ``__init__.py``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom):
                    names.update(alias.asname or alias.name for alias in stmt.names)
    return names


@pytest.mark.parametrize("package", PACKAGES)
def test_type_checking_imports_match_exports(package):
    """The TYPE_CHECKING block (for IDEs/type checkers) must mirror the lazy exports."""
    pkg = importlib.import_module(package)
    init = pathlib.Path(pkg.__file__)
    assert _type_checking_names(init) == set(pkg.__all__) - {"__version__"}


def test_leaked_implementation_names_are_gone():
    for name in ("np", "pd", "os", "sys", "tqdm", "List", "ThreadPoolExecutor"):
        assert not hasattr(open_rating_engine, name), name


def test_version_is_a_string():
    assert isinstance(open_rating_engine.__version__, str)


def _modules_loaded_by(code):
    out = subprocess.run(
        [sys.executable, "-c", f"{code}\nimport sys\nprint('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        check=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return set(out.stdout.split())


def test_importing_the_package_loads_nothing_heavy():
    loaded = _modules_loaded_by("import open_rating_engine")
    assert not {"mlflow", "pandas", "numpy", "scipy", "numba"} & loaded


def test_binning_import_does_not_load_mlflow():
    """The fit-time encoding helpers only need numpy/pandas."""
    loaded = _modules_loaded_by(
        "from open_rating_engine.rating_engine.binning import thermometer_column_names"
    )
    assert "pandas" in loaded
    assert "mlflow" not in loaded


def test_data_aging_import_does_not_load_mlflow():
    loaded = _modules_loaded_by("from open_rating_engine import apply_aging")
    assert "mlflow" not in loaded
