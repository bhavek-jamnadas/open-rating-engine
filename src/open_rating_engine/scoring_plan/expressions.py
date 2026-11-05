"""Compile a layer's expression string into the scoring plan's expression tree.

A ``ColumnCalculation`` expression is Python evaluated against pandas Series. The
scoring plan holds the same expression as a small JSON tree that the Rust scorer
evaluates without Python. Only a fixed vocabulary is accepted: column references,
arithmetic, comparisons, ``&``/``|``/``~``, a handful of numpy and pandas functions,
a few Series methods and ``aggregate()``. Anything else raises
:class:`UnsupportedExpression`, which the exporter reports as a blocking layer.

Any part of an expression that reads no row data (a catalog lookup, ``np.inf``,
``list(range(15, 66, 10)) + [np.inf]``) is evaluated once here, with the same names
in scope as at prediction time, and stored as a constant.
"""

import ast
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..rating_engine.pricing_layers import (
    ColumnCalculation,
    LayerCalculationError,
    _ALLOWED_AGG_FUNCS,
    _resolve_structural_join,
)

__all__ = [
    "ExpressionScope",
    "UnsupportedExpression",
    "compile_expression",
    "const",
    "encode_float",
]


class UnsupportedExpression(Exception):
    """Raised when an expression uses something the scoring plan cannot express."""


# Attribute names that make an otherwise row-free expression unsafe to evaluate once
# at export time: its value would differ between export and prediction.
_NOT_CONSTANT_ATTRS = frozenset({
    "random", "now", "today", "utcnow", "read_csv", "read_parquet", "read_json",
    "read_excel", "read_sql", "to_csv", "to_parquet", "to_json",
})

_ARITH_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_CMP_OPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
_LOGIC_OPS = {ast.BitAnd: "and", ast.BitOr: "or"}

# name -> (positional parameter names, defaults). Mirrors the numpy/pandas signatures
# for the arguments the plan supports.
_NP_FUNCS = {
    "where": (["condition", "x", "y"], {}),
    "maximum": (["x1", "x2"], {}),
    "minimum": (["x1", "x2"], {}),
    "abs": (["x"], {}),
    "absolute": (["x"], {}),
    "log": (["x"], {}),
    "exp": (["x"], {}),
    "sqrt": (["x"], {}),
    "floor": (["x"], {}),
    "ceil": (["x"], {}),
    "round": (["a", "decimals"], {"decimals": 0}),
    "around": (["a", "decimals"], {"decimals": 0}),
    "isnan": (["x"], {}),
    "clip": (["a", "a_min", "a_max"], {}),
    "power": (["x1", "x2"], {}),
}
_PD_FUNCS = {
    "cut": (
        ["x", "bins", "right", "labels", "retbins", "precision", "include_lowest"],
        {"right": True, "labels": None, "retbins": False, "precision": 3, "include_lowest": False},
    ),
    "isna": (["obj"], {}),
    "isnull": (["obj"], {}),
    "notna": (["obj"], {}),
    "notnull": (["obj"], {}),
}
_SERIES_METHODS = {
    "fillna": (["value"], {}),
    "isna": ([], {}),
    "isnull": ([], {}),
    "notna": ([], {}),
    "notnull": ([], {}),
    "astype": (["dtype"], {}),
    "abs": ([], {}),
    "round": (["decimals"], {"decimals": 0}),
    "clip": (["lower", "upper"], {"lower": None, "upper": None}),
    "to_numpy": ([], {}),
}
_AGGREGATE_SIG = (
    ["source", "agg_func", "source_column", "group_by", "fill_value"],
    {"source_column": None, "group_by": None, "fill_value": None},
)

_INT_DTYPES = (int, np.int64, "int", "int64")
_FLOAT_DTYPES = (float, np.float64, "float", "float64")


class ExpressionScope:
    """The names an expression can see, and how bare names resolve.

    Parameters
    ----------
    kind : str
        ``"column"`` for a ``ColumnCalculation`` (``df``, ``data``, ``catalog``,
        ``options`` and ``aggregate`` in scope), ``"rating_formula"`` for a
        ``RatingTablesLayer`` formula (each rating table is a name), or
        ``"glm_derived"`` for a ``GlmScoring`` or ``OnnxScoring`` formula column (bare
        names are columns).
    """

    def __init__(
        self,
        kind: str,
        *,
        catalog: Optional[Dict] = None,
        options: Optional[Dict] = None,
        target_table: Optional[str] = None,
        rating_tables: Iterable[str] = (),
    ):
        self.kind = kind
        self.catalog = catalog or {}
        self.options = options or {}
        self.target_table = target_table
        self.rating_tables = frozenset(rating_tables)

    def constant_namespace(self) -> Dict:
        """Names available when folding a row-free subexpression."""
        namespace = {"__builtins__": ColumnCalculation._safe_builtins(), "np": np, "pd": pd}
        if self.kind == "column":
            namespace["catalog"] = self.catalog
            namespace["options"] = self.options
        return namespace

    def is_row_name(self, name: str) -> bool:
        """True when ``name`` refers to row data, so expressions using it can't fold."""
        if self.kind == "column":
            return name in ("df", "data", "aggregate")
        if self.kind == "rating_formula":
            return name in self.rating_tables
        # glm_derived: every name that isn't a module or a builtin is a column.
        return name not in self.constant_namespace() and name not in ColumnCalculation._safe_builtins()


def encode_float(value: float):
    """JSON form of a float in the plan: a number, or "inf" / "-inf" / "nan"."""
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def const(value, strong: bool = False) -> dict:
    """Plan node for a scalar constant.

    Constants are "weak" like Python scalars in numpy: ``float32 * 2.0`` stays
    float32. ``strong=True`` makes the constant behave like a float64 column.
    """
    if value is None:
        node = {"k": "const", "t": "null"}
    elif isinstance(value, (bool, np.bool_)):
        node = {"k": "const", "t": "bool", "v": bool(value)}
    elif isinstance(value, (int, np.integer)):
        if not -(2 ** 63) <= int(value) < 2 ** 63:
            raise UnsupportedExpression(f"integer constant {value} does not fit in 64 bits")
        node = {"k": "const", "t": "int", "v": int(value)}
    elif isinstance(value, (float, np.floating)):
        node = {"k": "const", "t": "float", "v": encode_float(value)}
    elif isinstance(value, (str, np.str_)):
        node = {"k": "const", "t": "str", "v": str(value)}
    else:
        raise UnsupportedExpression(f"constant of type {type(value).__name__} is not supported")
    if strong:
        node["strong"] = True
    return node


def compile_expression(source: str, scope: ExpressionScope, data_structure=None) -> dict:
    """Compile ``source`` to a plan expression node.

    Raises :class:`UnsupportedExpression` naming the first construct the plan
    cannot express.
    """
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsupportedExpression(f"invalid Python syntax: {exc.msg}") from exc
    return _Compiler(scope, data_structure or []).expr(tree.body)


class _Compiler:
    def __init__(self, scope: ExpressionScope, data_structure: List[Dict]):
        self.scope = scope
        self.data_structure = data_structure

    # -- constant folding -------------------------------------------------

    def _reads_rows(self, node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and self.scope.is_row_name(sub.id):
                return True
            if isinstance(sub, ast.Attribute) and sub.attr in _NOT_CONSTANT_ATTRS:
                return True
            if isinstance(sub, (ast.Lambda, ast.NamedExpr, ast.ListComp, ast.GeneratorExp,
                                ast.DictComp, ast.SetComp)):
                return True
        return False

    def _evaluate(self, node: ast.AST):
        code = compile(ast.Expression(body=node), "<plan-constant>", "eval")
        try:
            return eval(code, self.scope.constant_namespace())  # noqa: S307 - same namespace as predict
        except Exception as exc:
            raise UnsupportedExpression(
                f"could not evaluate {ast.unparse(node)!r} at export time: {exc}"
            ) from exc

    def constant_value(self, node: ast.AST):
        """Python value of a row-free subexpression, or raise."""
        if self._reads_rows(node):
            raise UnsupportedExpression(f"{ast.unparse(node)!r} must not depend on row data")
        return self._evaluate(node)

    def _folded(self, node: ast.AST) -> Optional[dict]:
        if self._reads_rows(node):
            return None
        value = self._evaluate(node)
        if isinstance(value, (np.ndarray, list, tuple, dict, set, pd.Series, pd.DataFrame)):
            return None
        return const(value)

    # -- expressions -------------------------------------------------------

    def expr(self, node: ast.AST) -> dict:
        folded = self._folded(node)
        if folded is not None:
            return folded

        if isinstance(node, ast.Subscript):
            return self._subscript(node)
        if isinstance(node, ast.Name):
            if self.scope.kind == "glm_derived" and self.scope.is_row_name(node.id):
                return {"k": "col", "name": node.id}
            raise UnsupportedExpression(f"name {node.id!r} cannot be used as a value here")
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op in _ARITH_OPS:
                return {"k": "arith", "op": _ARITH_OPS[op], "l": self.expr(node.left), "r": self.expr(node.right)}
            if op in _LOGIC_OPS:
                return {"k": "logic", "op": _LOGIC_OPS[op], "l": self.expr(node.left), "r": self.expr(node.right)}
            raise UnsupportedExpression(f"operator {type(node.op).__name__} is not supported")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return {"k": "neg", "x": self.expr(node.operand)}
            if isinstance(node.op, ast.UAdd):
                return self.expr(node.operand)
            if isinstance(node.op, ast.Invert):
                return {"k": "not", "x": self.expr(node.operand)}
            raise UnsupportedExpression("'not' cannot be applied to a column; use ~")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP_OPS:
                raise UnsupportedExpression(f"comparison {ast.unparse(node)!r} is not supported")
            return {
                "k": "cmp",
                "op": _CMP_OPS[type(node.ops[0])],
                "l": self.expr(node.left),
                "r": self.expr(node.comparators[0]),
            }
        if isinstance(node, ast.Attribute):
            if node.attr == "values":
                return self.expr(node.value)
            raise UnsupportedExpression(f"attribute .{node.attr} is not supported")
        if isinstance(node, ast.Call):
            return self._call(node)
        raise UnsupportedExpression(f"{type(node).__name__} expressions are not supported")

    def _subscript(self, node: ast.Subscript) -> dict:
        base = node.value
        if isinstance(base, ast.Name):
            key = self.constant_value(node.slice)
            if self.scope.kind in ("column", "glm_derived") and base.id == "df":
                if not isinstance(key, str):
                    raise UnsupportedExpression("df[...] must select one column by name")
                return {"k": "col", "name": key}
            if self.scope.kind == "rating_formula" and base.id in self.scope.rating_tables:
                if key != "value":
                    raise UnsupportedExpression(f"rating table {base.id!r} only has a 'value' column")
                return {"k": "rt", "table": base.id}
        if _root_name(node) == "data":
            raise UnsupportedExpression("data[...] is not supported; use df[...] or aggregate()")
        raise UnsupportedExpression(f"subscript {ast.unparse(node)!r} is not supported")

    # -- calls -------------------------------------------------------------

    @staticmethod
    def _bind(node: ast.Call, signature, what: str) -> Dict[str, ast.AST]:
        names, defaults = signature
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise UnsupportedExpression(f"*args/**kwargs are not supported in {what}")
        if len(node.args) > len(names):
            raise UnsupportedExpression(f"too many arguments to {what}")
        bound = dict(zip(names, node.args))
        for kw in node.keywords:
            if kw.arg not in names:
                raise UnsupportedExpression(f"argument {kw.arg!r} of {what} is not supported")
            if kw.arg in bound:
                raise UnsupportedExpression(f"{what} got {kw.arg!r} twice")
            bound[kw.arg] = kw.value
        for name in names:
            if name not in bound and name not in defaults:
                raise UnsupportedExpression(f"{what} is missing argument {name!r}")
        return bound

    def _arg_value(self, bound, name, defaults):
        return self.constant_value(bound[name]) if name in bound else defaults[name]

    def _call(self, node: ast.Call) -> dict:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "np":
            if func.attr not in _NP_FUNCS:
                raise UnsupportedExpression(f"np.{func.attr} is not supported")
            return self._np_call(func.attr, self._bind(node, _NP_FUNCS[func.attr], f"np.{func.attr}"))
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pd":
            if func.attr not in _PD_FUNCS:
                raise UnsupportedExpression(f"pd.{func.attr} is not supported")
            return self._pd_call(func.attr, node)
        if isinstance(func, ast.Attribute) and _root_name(func) in ("np", "pd"):
            raise UnsupportedExpression(f"{ast.unparse(func)} is not supported")
        if isinstance(func, ast.Attribute):
            if func.attr not in _SERIES_METHODS:
                raise UnsupportedExpression(f"method .{func.attr}() is not supported")
            return self._method(func.attr, func.value, node)
        if isinstance(func, ast.Name):
            if func.id == "aggregate" and self.scope.kind == "column":
                return self._aggregate(node)
            if func.id == "abs":
                bound = self._bind(node, (["x"], {}), "abs()")
                return {"k": "call", "f": "abs", "args": [self.expr(bound["x"])]}
            if func.id == "round":
                bound = self._bind(node, (["number", "ndigits"], {"ndigits": None}), "round()")
                digits = self._arg_value(bound, "ndigits", {"ndigits": None})
                return self._round(self.expr(bound["number"]), 0 if digits is None else digits)
        raise UnsupportedExpression(f"call {ast.unparse(node.func)}() is not supported")

    def _round(self, x: dict, decimals) -> dict:
        if not isinstance(decimals, (int, np.integer)) or isinstance(decimals, bool):
            raise UnsupportedExpression("round() decimals must be an integer constant")
        return {"k": "call", "f": "round", "args": [x], "decimals": int(decimals)}

    def _np_call(self, name: str, bound: Dict[str, ast.AST]) -> dict:
        defaults = _NP_FUNCS[name][1]
        if name == "where":
            return {"k": "call", "f": "where",
                    "args": [self.expr(bound["condition"]), self.expr(bound["x"]), self.expr(bound["y"])]}
        if name in ("maximum", "minimum"):
            return {"k": "call", "f": name, "args": [self.expr(bound["x1"]), self.expr(bound["x2"])]}
        if name == "power":
            return {"k": "arith", "op": "pow", "l": self.expr(bound["x1"]), "r": self.expr(bound["x2"])}
        if name in ("abs", "absolute"):
            return {"k": "call", "f": "abs", "args": [self.expr(bound["x"])]}
        if name in ("log", "exp", "sqrt", "floor", "ceil"):
            return {"k": "call", "f": name, "args": [self.expr(bound["x"])]}
        if name == "isnan":
            return {"k": "call", "f": "isna", "args": [self.expr(bound["x"])]}
        if name in ("round", "around"):
            return self._round(self.expr(bound["a"]), self._arg_value(bound, "decimals", defaults))
        if name == "clip":
            return self._clip(self.expr(bound["a"]), bound.get("a_min"), bound.get("a_max"))
        raise UnsupportedExpression(f"np.{name} is not supported")  # pragma: no cover

    def _clip(self, x: dict, lower: Optional[ast.AST], upper: Optional[ast.AST]) -> dict:
        node = {"k": "call", "f": "clip", "args": [x]}
        for key, arg in (("lower", lower), ("upper", upper)):
            value = None if arg is None else self.constant_value(arg)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    raise UnsupportedExpression("clip bounds must be numeric constants")
                node[key] = const(value)
        return node

    def _pd_call(self, name: str, node: ast.Call) -> dict:
        signature = _PD_FUNCS[name]
        bound = self._bind(node, signature, f"pd.{name}")
        if name in ("isna", "isnull"):
            return {"k": "call", "f": "isna", "args": [self.expr(bound["obj"])]}
        if name in ("notna", "notnull"):
            return {"k": "call", "f": "notna", "args": [self.expr(bound["obj"])]}
        return self._cut(bound, signature[1])

    def _cut(self, bound: Dict[str, ast.AST], defaults: Dict) -> dict:
        bins = self._arg_value(bound, "bins", defaults)
        labels = self._arg_value(bound, "labels", defaults)
        right = self._arg_value(bound, "right", defaults)
        include_lowest = self._arg_value(bound, "include_lowest", defaults)
        if self._arg_value(bound, "retbins", defaults):
            raise UnsupportedExpression("pd.cut(retbins=True) is not supported")
        edges = _numeric_list(bins, "pd.cut bins")
        if len(edges) < 2:
            raise UnsupportedExpression("pd.cut needs at least two bin edges")
        if any(not a < b for a, b in zip(edges, edges[1:])):
            raise UnsupportedExpression("pd.cut bins must be strictly increasing")
        if labels is None:
            raise UnsupportedExpression(
                "pd.cut without labels produces intervals; pass labels=[...] or labels=False"
            )
        if labels is False:
            out_labels = None
        else:
            out_labels = list(labels) if isinstance(labels, (list, tuple, np.ndarray, pd.Index)) else None
            if out_labels is None or not all(isinstance(v, str) for v in out_labels):
                raise UnsupportedExpression("pd.cut labels must be a list of strings or False")
            if len(out_labels) != len(edges) - 1:
                raise UnsupportedExpression("pd.cut needs one label per bin")
        return {
            "k": "cut",
            "x": self.expr(bound["x"]),
            "edges": [encode_float(e) for e in edges],
            "labels": out_labels,
            "right": bool(right),
            "include_lowest": bool(include_lowest),
        }

    def _method(self, name: str, receiver: ast.AST, node: ast.Call) -> dict:
        signature = _SERIES_METHODS[name]
        bound = self._bind(node, signature, f".{name}()")
        x = self.expr(receiver)
        if name == "to_numpy":
            return x
        if name == "fillna":
            value = self.constant_value(bound["value"])
            if isinstance(value, (list, tuple, dict, np.ndarray, pd.Series)):
                raise UnsupportedExpression(".fillna() value must be a scalar constant")
            return {"k": "call", "f": "fillna", "args": [x, const(value)]}
        if name in ("isna", "isnull"):
            return {"k": "call", "f": "isna", "args": [x]}
        if name in ("notna", "notnull"):
            return {"k": "call", "f": "notna", "args": [x]}
        if name == "abs":
            return {"k": "call", "f": "abs", "args": [x]}
        if name == "round":
            return self._round(x, self._arg_value(bound, "decimals", signature[1]))
        if name == "clip":
            return self._clip(x, bound.get("lower"), bound.get("upper"))
        dtype = self.constant_value(bound["dtype"])
        if any(dtype is d or dtype == d for d in _INT_DTYPES):
            return {"k": "call", "f": "astype_int", "args": [x]}
        if any(dtype is d or dtype == d for d in _FLOAT_DTYPES):
            return {"k": "call", "f": "astype_float", "args": [x]}
        raise UnsupportedExpression(f".astype({dtype!r}) is not supported; use int or float")

    def _aggregate(self, node: ast.Call) -> dict:
        bound = self._bind(node, _AGGREGATE_SIG, "aggregate()")
        defaults = _AGGREGATE_SIG[1]
        source = self._arg_value(bound, "source", defaults)
        agg_func = self._arg_value(bound, "agg_func", defaults)
        source_column = self._arg_value(bound, "source_column", defaults)
        group_by = self._arg_value(bound, "group_by", defaults)
        fill_value = self._arg_value(bound, "fill_value", defaults)
        if not isinstance(source, str):
            raise UnsupportedExpression("aggregate() source must be a table name")
        if agg_func not in _ALLOWED_AGG_FUNCS:
            raise UnsupportedExpression(f"aggregate() agg_func {agg_func!r} is not supported")
        if source_column is None and agg_func != "count":
            raise UnsupportedExpression(f"aggregate() needs source_column for agg_func={agg_func!r}")
        if group_by is not None:
            if not isinstance(group_by, str):
                raise UnsupportedExpression("aggregate() group_by must be one column name")
            target_key = source_key = group_by
        else:
            try:
                target_key, source_key = _resolve_structural_join(
                    self.data_structure, self.scope.target_table, source
                )
            except LayerCalculationError as exc:
                raise UnsupportedExpression(str(exc)) from exc
        if fill_value is None:
            fill = const(0) if agg_func == "count" else const(float("nan"))
        else:
            fill = const(fill_value)
        return {
            "k": "agg",
            "source": source,
            "func": agg_func,
            "column": source_column,
            "target_key": target_key,
            "source_key": source_key,
            "fill": fill,
        }


def _root_name(node: ast.AST) -> Optional[str]:
    """The name at the root of ``a.b[c].d``, if any."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _numeric_list(value, what: str) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        items = list(value)
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in items):
            return [float(v) for v in items]
    raise UnsupportedExpression(f"{what} must be a list of numbers")
