"""The ONE place the engine evaluates a formula written by a user.

Every formula the engine runs — a ``ColumnCalculation`` calculation, a
``RatingTablesLayer`` formula, a GLM/ONNX derived column, a ``SourceColumn`` aging
rule, and the scoring-plan exporter's constant folding — is user-authored text. On
a shared instance, the user is any tenant. So each one goes through
:func:`safe_eval`, and nothing else in the package may call :func:`eval`
(``tests/test_safe_eval.py`` greps for it).

Why an allowlist, and why restricted builtins are not enough on their own
------------------------------------------------------------------------
Removing ``open`` / ``__import__`` from ``__builtins__`` does not make ``eval``
safe. Any object in scope leads back to a module: ``pd.io.common.os.system(...)``
and ``().__class__.__base__.__subclasses__()`` both evaluate under the restricted
builtins this engine used before. So the source is parsed and every node checked
against what formulas actually use (a census of every formula in the product's
tests, docs and stored configs, 2026-11-16) BEFORE it is compiled:

* **Node types** — arithmetic, comparisons, boolean logic, ``x if c else y``,
  subscripts and slices, literals, names and calls. No lambda, comprehension,
  walrus, f-string, ``*args`` or ``**kwargs``: none appears in a real formula,
  and each is a way to build a value the checks below cannot see.
* **Names** — any name not starting with ``_``. A name can only resolve to what
  the caller put in the namespace or to :data:`SAFE_BUILTINS`, because
  ``__builtins__`` is always replaced.
* **Attributes** — never one starting with ``_``. ``np.<name>`` and
  ``pd.<name>`` must be in :data:`NUMPY_ATTRIBUTES` / :data:`PANDAS_ATTRIBUTES`;
  every other ``<value>.<name>`` must be in :data:`VALUE_ATTRIBUTES`. The value
  is unknown until run time, so that list holds only names that are harmless on
  ANY type: no ``to_*`` writer, no ``eval`` / ``query`` / ``apply`` / ``pipe`` /
  ``agg`` / ``transform`` (each runs code or dispatches by a string name), no
  ``format``.
* **Calls** — the callee is a name or an allowed attribute, never the result of
  a subscript or another call.

Not a goal: bounding CPU or memory. ``10 ** 10 ** 10`` passes, and so does a huge
``range``. Containing a runaway formula needs a time limit and process isolation
around scoring, not a syntax check.
"""
from __future__ import annotations

import ast
import builtins
from functools import lru_cache
from types import CodeType
from typing import Any, Mapping, Union

__all__ = [
    "MAX_EXPRESSION_LENGTH",
    "NUMPY_ATTRIBUTES",
    "PANDAS_ATTRIBUTES",
    "SAFE_BUILTINS",
    "UnsafeExpressionError",
    "VALUE_ATTRIBUTES",
    "check_expression",
    "safe_eval",
]


class UnsafeExpressionError(ValueError):
    """A formula uses a construct outside the allowlist. The message names it."""


MAX_EXPRESSION_LENGTH = 20_000

# The builtins a formula may call. Unchanged from the engine's earlier
# `_safe_builtins()`, so no working formula loses a name.
SAFE_BUILTINS: Mapping[str, Any] = {
    "int": int, "float": float, "str": str,
    "bool": bool, "len": len, "abs": abs,
    "round": round, "min": min, "max": max,
    "sum": sum, "zip": zip, "map": map,
    "enumerate": enumerate, "range": range,
    "list": list, "dict": dict, "tuple": tuple,
    "set": set, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}

NUMPY_ATTRIBUTES = frozenset({
    # elementwise maths
    "abs", "absolute", "sign", "sqrt", "square", "power", "exp", "expm1",
    "log", "log1p", "log2", "log10", "floor", "ceil", "round", "around", "rint",
    "trunc", "clip", "maximum", "minimum", "fmax", "fmin", "mod", "divide",
    # selection and tests
    "where", "select", "isnan", "isinf", "isfinite", "nan_to_num", "isin",
    "logical_and", "logical_or", "logical_not", "digitize",
    # reductions
    "sum", "mean", "median", "min", "max", "prod", "cumsum", "cumprod",
    "nansum", "nanmean", "nanmin", "nanmax",
    # construction
    "array", "arange", "linspace", "zeros_like", "ones_like", "full_like",
    # constants and dtypes
    "nan", "inf", "pi", "e",
    "float64", "float32", "int64", "int32", "int8", "bool_", "str_",
    "datetime64", "timedelta64",
})

PANDAS_ATTRIBUTES = frozenset({
    "cut", "qcut", "isna", "isnull", "notna", "notnull",
    "to_numeric", "to_datetime", "to_timedelta",
    "Series", "Categorical", "Timestamp", "Timedelta", "DateOffset",
    "NA", "NaT",
})

# Methods and properties allowed on ANY value — a Series, a DataFrame, a scalar,
# a string literal, a dict. ⚠️ Before adding a name, check it on every type it
# could meet: a name harmless on a Series may be a writer, a code runner or a
# getattr-by-string dispatcher on another.
VALUE_ATTRIBUTES = frozenset({
    # missing values
    "fillna", "isna", "isnull", "notna", "notnull", "dropna", "ffill", "bfill",
    # types
    "astype", "dtype", "dtypes",
    # arithmetic and comparison methods
    "abs", "round", "clip", "pow", "add", "sub", "mul", "div", "truediv",
    "floordiv", "mod", "radd", "rsub", "rmul", "rdiv", "rtruediv",
    "lt", "le", "gt", "ge", "eq", "ne",
    # reductions and running totals
    "sum", "mean", "median", "min", "max", "std", "var", "count", "nunique",
    "prod", "any", "all", "cumsum", "cumprod", "cummax", "cummin",
    "idxmax", "idxmin", "rank", "diff", "pct_change", "shift",
    # selection
    "where", "mask", "isin", "between", "loc", "iloc", "at", "iat", "get",
    "head", "tail", "first", "last", "nlargest", "nsmallest",
    # shape and labels
    "values", "to_numpy", "to_list", "tolist", "item", "keys", "index",
    "columns", "shape", "size", "ndim", "empty", "name", "T",
    "set_index", "reset_index", "rename", "sort_values", "sort_index",
    "unique", "duplicated", "drop_duplicates", "value_counts", "copy",
    # mapping values
    "map", "replace", "combine_first",
    # accessors, and what they expose
    "str", "dt", "cat",
    "upper", "lower", "title", "capitalize", "strip", "lstrip", "rstrip",
    "len", "contains", "startswith", "endswith", "slice", "split", "zfill",
    "pad", "match", "fullmatch", "extract", "isnumeric", "isdigit", "isalpha",
    "year", "month", "day", "dayofweek", "weekday", "dayofyear", "quarter",
    "hour", "minute", "second", "days", "date", "normalize", "floor", "ceil",
    "strftime", "days_in_month", "is_month_end", "is_month_start",
    "total_seconds", "codes", "categories",
})

_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.UAdd, ast.USub, ast.Invert, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)
_STRUCTURE = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.Name,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Load,
)
_ALLOWED_NODES = _STRUCTURE + _OPERATORS
_CONSTANT_TYPES = (str, int, float, bool, type(None))

# A plain-language name for the node types formulas most often reach for.
_NODE_LABELS = {
    "Lambda": "a lambda", "ListComp": "a list comprehension",
    "SetComp": "a set comprehension", "DictComp": "a dict comprehension",
    "GeneratorExp": "a generator expression", "NamedExpr": "the := operator",
    "JoinedStr": "an f-string", "FormattedValue": "an f-string",
    "Starred": "*args", "Await": "await", "Yield": "yield",
}


def _describe(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001 - only used to word a message
        return type(node).__name__
    return text if len(text) <= 60 else text[:57] + "..."


def _check_attribute(node: ast.Attribute) -> None:
    # Inside out, so `pd.io.common.os.getpid` is reported at `pd.io` — where the
    # chain first leaves the allowlist — not at whichever link the walk met first.
    if isinstance(node.value, ast.Attribute):
        _check_attribute(node.value)
    attr = node.attr
    if attr.startswith("_"):
        raise UnsafeExpressionError(
            f"`{_describe(node)}` reads `{attr}`; a formula may not use a name starting with '_'"
        )
    receiver = node.value
    if isinstance(receiver, ast.Name) and receiver.id in ("np", "pd"):
        allowed = NUMPY_ATTRIBUTES if receiver.id == "np" else PANDAS_ATTRIBUTES
        if attr not in allowed:
            raise UnsafeExpressionError(
                f"`{receiver.id}.{attr}` is not one of the {receiver.id} functions a formula may use"
            )
        return
    if attr not in VALUE_ATTRIBUTES:
        hint = ""
        if isinstance(receiver, ast.Name) and receiver.id == "df":
            hint = f"; to read a column write df['{attr}']"
        raise UnsafeExpressionError(
            f"`.{attr}` in `{_describe(node)}` is not a method or property a formula may use{hint}"
        )


def _check_tree(tree: ast.AST) -> None:
    # A '_' name is the clearest sign of an escape attempt, so name it first rather
    # than whichever attribute the walk happens to reach before it.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise UnsafeExpressionError(
                f"`{node.id}`: a formula may not use a name starting with '_'"
            )
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            label = _NODE_LABELS.get(type(node).__name__, type(node).__name__)
            raise UnsafeExpressionError(f"a formula may not use {label}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, _CONSTANT_TYPES):
            raise UnsafeExpressionError(
                f"a formula may not use a {type(node.value).__name__} literal"
            )
        if isinstance(node, ast.Attribute):
            _check_attribute(node)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise UnsafeExpressionError(
                    f"`{_describe(node)}`: a formula may only call a function or method by name"
                )
            if any(k.arg is None for k in node.keywords):
                raise UnsafeExpressionError("a formula may not use **kwargs")


def _parse(source: str) -> ast.Expression:
    if not isinstance(source, str):
        raise UnsafeExpressionError(f"a formula must be a string, not {type(source).__name__}")
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"a formula may be at most {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        return ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        raise  # a syntax error stays a SyntaxError: callers already word those
    except (RecursionError, MemoryError, ValueError) as exc:
        raise UnsafeExpressionError(f"a formula could not be parsed: {exc}") from exc


def check_expression(source: Union[str, ast.AST]) -> None:
    """Raise :class:`UnsafeExpressionError` if ``source`` is outside the allowlist.

    ``source`` is formula text, or an already-parsed node (the scoring-plan
    exporter folds sub-expressions it has parsed itself). A syntax error is
    raised as :class:`SyntaxError`, unchanged. Returns ``None`` when the formula
    is allowed; it does not check that the names it reads exist.
    """
    _check_tree(source if isinstance(source, ast.AST) else _parse(source))


@lru_cache(maxsize=2048)
def _compile_text(source: str) -> CodeType:
    tree = _parse(source)
    _check_tree(tree)
    return compile(tree, "<formula>", "eval")


def _compile(source: Union[str, ast.AST]) -> CodeType:
    if isinstance(source, str):
        return _compile_text(source)
    node = source.body if isinstance(source, ast.Expression) else source
    tree = ast.Expression(body=node)
    _check_tree(tree)
    return compile(ast.fix_missing_locations(tree), "<formula>", "eval")


def safe_eval(source: Union[str, ast.AST], namespace: Mapping[str, Any]) -> Any:
    """Check ``source`` against the allowlist, then evaluate it.

    ``namespace`` is every name the formula may read (``df``, ``np``, rating
    tables, bare column names...). ``__builtins__`` is always
    :data:`SAFE_BUILTINS`, whatever ``namespace`` holds. Raises
    :class:`UnsafeExpressionError` before running anything outside the
    allowlist; any error the formula itself raises propagates unchanged.
    """
    code = _compile(source)
    scope = dict(namespace)
    scope["__builtins__"] = dict(SAFE_BUILTINS)
    return builtins.eval(code, scope)  # the one sanctioned eval: see the module docstring
