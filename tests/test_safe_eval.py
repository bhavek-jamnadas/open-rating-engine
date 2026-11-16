"""The formula allowlist (`open_rating_engine.safe_eval`).

Three things are pinned here:

1. Every formula shape the product actually uses still passes — the corpus is the
   2026-11-16 census of stored configs, tests and docs. A name dropped from an
   allowlist breaks a customer's algorithm, so this list is the regression guard.
2. Each known way out of a restricted-builtins `eval` is refused BEFORE anything runs.
3. Nothing else in the package calls `eval` / `exec` / `compile` — a new call site
   that skipped `safe_eval` would reopen the hole without failing any other test.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from open_rating_engine.safe_eval import (
    UnsafeExpressionError,
    check_expression,
    safe_eval,
)

ALLOWED = [
    # stored ColumnCalculation formulas
    "df['risk.premium_stat_prem_total']",
    "np.where(df['risk.previous_gwp'] != 0, (df['risk.gwp'] - df['risk.previous_gwp']) / df['risk.previous_gwp'], np.nan)",
    "pd.cut(df['risk.customer_tenure'], bins=[-1, 5, 10, np.inf], labels=['0-5', '6-10', '10+'])",
    "pd.cut(df['age'], bins=list(range(15, 66, 10)) + [np.inf], labels=['15-24', '25-34', '35-44', '45-54', '55-64', '65+'])",
    "aggregate(source='policy', source_column='policy.customer_tenure', agg_func='first')",
    "aggregate(source='policy', source_column='policy.premium', agg_func='sum') * df['risk.pred'] / df['risk.policy_pred']",
    "catalog['assumptions_expenses'][catalog['assumptions_expenses']['expense_type'] == 'expense_fixed']['value'].values[0] * df['risk.cost']",
    "- catalog['exp'][catalog['exp']['t'] == 'x']['value'].values[0] * df['risk.cost']",
    "np.where(df['prev'].isna(), df['total'], np.maximum(np.minimum(df['total'], df['prev'] * (1 + df['cap'])), df['prev'] * (1 + df['cup'])))",
    "df['cup_percentage'].fillna(-0.5)",
    "(df['risk.vehicle_usage'] == 'Rideshare').astype(int)",
    "np.where(df['cover.cover_type']=='hire_car', 1, 0)",
    "np.abs(df['risk.gwp'] - df['risk.competitor_premiums_average'])",
    "df['risk.gwp'] / df['risk.previous_gwp'] - 1",
    # RatingTablesLayer formulas
    "risk_number['value']",
    "risk_age_band['value'] * customer_tenure_band['value'] + 0.05",
    # GLM / ONNX derived columns (bare names are columns)
    "np.clip(age, 0, 50)",
    "np.minimum(age_raw, 60) + 1",
    "np.log(df['value']) + np.sqrt(df['age'])",
    "np.floor(df['age'] / 7) + np.ceil(df['age'] / 7)",
    "1.0 / (1.0 + np.exp(3.0 * df['price_change']))",
    "np.round(df['ratio'], 2)",
    # aging rules
    "df['eff_date'] + pd.DateOffset(years=1)",
    "df['age'] + 1",
    "df['sum_insured'] * 0.9",
    # other shapes seen in tests and docs
    "df['name'].str.upper()",
    "df['a'].str.len()",
    "df['x'].map({'a': 1, 'b': 2})",
    "df['premium'] / df.shape[0]",
    "df['base_premium'].head(2)",
    "catalog['factors'].set_index('name').loc['base', 'value'] * df['value']",
    "df[['gwp']].sum(axis=1)",
    "df['a'] if True else df['b']",
    "(df['a'] > 1) & (df['b'] < 2) | ~df['c'].isna()",
    "df['x'].isin(['a', 'b'])",
    "df['d'].dt.year - 2000",
]

ESCAPES = [
    "pd.io.common.os.getpid()",                       # module reached through pandas
    "().__class__.__base__.__subclasses__()",        # the classic dunder walk
    "df.__class__",
    "aggregate.__globals__",
    "__import__('os')",
    "__builtins__",
    "np.load('x.npy', allow_pickle=True)",           # pickle via numpy
    "np.random.rand() * df['x']",
    "np.lib.npyio",
    "pd.read_csv('/etc/passwd')",
    "pd.read_pickle('https://example.invalid/x')",
    "pd.eval('1+1')",
    "df.to_csv('/tmp/x')",
    "data['policy'].to_pickle('/tmp/x')",
    "df.eval('a + 1')",
    "df.query('a > 1')",
    "df['a'].apply(len)",
    "df.pipe(len)",
    "df['a'].agg('sum')",
    "df['a'].transform('sum')",
    "'{0.__class__}'.format(df)",                     # attribute walk inside str.format
    "df.gi_frame",                                     # frame attributes have no underscore
    "df.age",                                          # column-as-attribute: write df['age']
    "(lambda: 1)()",
    "[x for x in range(3)]",
    "{x: 1 for x in range(3)}",
    "(x for x in range(3))",
    "(y := 1)",
    "f'{df}'",
    "max(*[1, 2])",
    "max(**{'a': 1})",
    "df['a'][0]()",                                    # call on a subscript
    "max(1, 2)(3)",                                    # call on a call result
    "b'bytes'",
]


@pytest.mark.parametrize("source", ALLOWED)
def test_real_formulas_pass(source):
    check_expression(source)


@pytest.mark.parametrize("source", ESCAPES)
def test_escapes_are_refused(source):
    with pytest.raises(UnsafeExpressionError):
        check_expression(source)


@pytest.mark.parametrize("source", ESCAPES)
def test_safe_eval_refuses_before_running(source):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(source, {"df": pd.DataFrame({"a": [1]}), "np": np, "pd": pd})


def test_message_names_where_the_chain_leaves_the_allowlist():
    with pytest.raises(UnsafeExpressionError, match=r"`pd\.io`"):
        check_expression("pd.io.common.os.getpid()")
    with pytest.raises(UnsafeExpressionError, match=r"df\['age'\]"):
        check_expression("df.age * 2")


def test_names_resolve_only_to_the_namespace_and_safe_builtins():
    # `getattr` and `open` are plain names, so the syntax check passes them; what
    # stops them is that __builtins__ is always the restricted set.
    for source in ("getattr(df, 'to_csv')", "open('x')", "eval('1')", "exec('1')"):
        check_expression(source)
        with pytest.raises(NameError):
            safe_eval(source, {"df": pd.DataFrame()})


def test_namespace_cannot_smuggle_full_builtins():
    import builtins

    ns = {"__builtins__": builtins.__dict__, "df": pd.DataFrame()}
    with pytest.raises(NameError):
        safe_eval("open('x')", ns)


def test_safe_eval_computes_a_real_formula():
    df = pd.DataFrame({"gwp": [100.0, 200.0], "prev": [80.0, 0.0]})
    out = safe_eval(
        "np.where(df['prev'] != 0, (df['gwp'] - df['prev']) / df['prev'], np.nan)",
        {"df": df, "np": np, "pd": pd},
    )
    assert out[0] == pytest.approx(0.25) and np.isnan(out[1])


def test_safe_eval_accepts_a_parsed_node():
    node = ast.parse("1 + 2", mode="eval").body
    assert safe_eval(node, {}) == 3
    with pytest.raises(UnsafeExpressionError):
        safe_eval(ast.parse("pd.io", mode="eval").body, {"pd": pd})


def test_syntax_errors_stay_syntax_errors():
    with pytest.raises(SyntaxError):
        check_expression("df['a'] +")


def test_overlong_formula_is_refused():
    with pytest.raises(UnsafeExpressionError):
        check_expression("1 + " * 6000 + "1")


# ── The layers refuse an unsafe formula when they are BUILT ───────────────────


def test_column_calculation_refuses_at_construction():
    from open_rating_engine.rating_engine.pricing_layers import ColumnCalculation

    with pytest.raises(UnsafeExpressionError, match="Layer 'probe'"):
        ColumnCalculation(
            name="probe", target_table="policy", output_column="probe",
            calculation="df['a'] * 0 + pd.io.common.os.getpid()",
        )


def test_column_calculation_still_reports_syntax_errors_when_run():
    from open_rating_engine.rating_engine.pricing_layers import (
        ColumnCalculation,
        LayerCalculationError,
    )

    layer = ColumnCalculation(name="bad", target_table="t", output_column="x", calculation="df['a'] +")
    with pytest.raises(LayerCalculationError):
        layer._compute_result({"t": pd.DataFrame({"a": [1]})}, {}, {})


def test_rating_tables_formula_refused_at_construction(sample_rating_tables):
    from open_rating_engine.rating_engine.pricing_layers import RatingTablesLayer

    with pytest.raises(UnsafeExpressionError):
        RatingTablesLayer(
            "rt", "risk", "factor",
            rating_tables=sample_rating_tables,
            formula="risk_usage_mult['value'].__class__",
        )


def test_aging_rule_has_no_full_builtins():
    from open_rating_engine.data_processing.data_aging import AgingRuleError, apply_aging

    frames = {"policy": pd.DataFrame({"age": [30]})}
    for expression in ("__import__('os').getpid()", "open('x')"):
        layers = {
            "policy.age": {
                "type": "SourceColumn", "source_table": "policy", "source_column": "age",
                "aging_rule": {"expression": expression, "enabled": True},
            }
        }
        with pytest.raises(AgingRuleError):
            apply_aging(layers, frames)


# ── Nothing else may evaluate code ────────────────────────────────────────────

_PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "open_rating_engine"
_CODE_RUNNERS = {"eval", "exec", "compile"}


def _code_runner_calls(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _CODE_RUNNERS:
            yield node.lineno, func.id
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in _CODE_RUNNERS
            and isinstance(func.value, ast.Name)
            and func.value.id == "builtins"
        ):
            yield node.lineno, f"builtins.{func.attr}"


def test_safe_eval_is_the_only_place_code_is_evaluated():
    assert _PACKAGE.is_dir(), _PACKAGE  # premise: we are scanning the real package
    files = sorted(_PACKAGE.rglob("*.py"))
    assert any(f.name == "pricing_layers.py" for f in files)
    offenders = [
        f"{path.relative_to(_PACKAGE)}:{line} {name}"
        for path in files
        if path.name != "safe_eval.py"
        for line, name in _code_runner_calls(path)
    ]
    assert offenders == [], "evaluate formulas through open_rating_engine.safe_eval: " + ", ".join(offenders)
