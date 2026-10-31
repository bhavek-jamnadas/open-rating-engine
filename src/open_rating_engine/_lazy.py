"""Lazy re-exports for package ``__init__`` modules (PEP 562).

A package ``__init__`` that eagerly imports its submodules makes every import of
anything in the package pay for all of them: ``from open_rating_engine.rating_engine
.binning import ...`` would load mlflow just to run the parent ``__init__``. Instead
each ``__init__`` declares which names it re-exports from which submodule, and the
submodule is imported the first time one of those names is accessed.

Each ``__init__`` also repeats the re-exports under ``if TYPE_CHECKING:`` so IDEs and
type checkers see real imports. ``tests/test_public_api.py`` keeps the two in sync.
"""

import sys
from importlib import import_module


def attach(package, exports, submodules=()):
    """Return ``(__getattr__, __dir__, __all__)`` for ``package``.

    Parameters
    ----------
    package : str
        The package's ``__name__``.
    exports : dict[str, list[str]]
        Maps a module path relative to ``package`` (e.g. ``".pricing_layers"``) to
        the public names re-exported from it.
    submodules : iterable of str
        Child module names (e.g. ``"optimization"``) made reachable as attributes,
        so ``open_rating_engine.optimization`` works after a bare
        ``import open_rating_engine``.
    """
    name_to_module = {}
    for module, names in exports.items():
        for name in names:
            if name in name_to_module:
                raise ValueError(
                    f"{package}: {name!r} is exported from both "
                    f"{name_to_module[name]!r} and {module!r}"
                )
            name_to_module[name] = module
    submodules = frozenset(submodules)
    __all__ = sorted(set(name_to_module) | submodules)

    # A name that is also the name of the submodule defining it cannot be lazy. Importing
    # that submodule -- from anywhere: a sibling name's lookup below, or a caller's direct
    # `import package.name` -- makes the import system bind the MODULE on the package
    # under that name, and __getattr__ only runs for a MISSING attribute, so it never runs
    # again. `from package import name` then returns the module, depending on import order.
    # Binding the defining object now, during the package's own import, is the only order
    # that cannot lose: a later import of the already-loaded submodule does not rebind it.
    for name, module in name_to_module.items():
        if module == f".{name}":
            setattr(sys.modules[package], name, getattr(import_module(module, package), name))

    def __getattr__(name):
        if name in name_to_module:
            value = getattr(import_module(name_to_module[name], package), name)
        elif name in submodules:
            value = import_module(f".{name}", package)
        else:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        # Cache on the package so __getattr__ runs at most once per name.
        setattr(sys.modules[package], name, value)
        return value

    def __dir__():
        return list(__all__)

    return __getattr__, __dir__, __all__
