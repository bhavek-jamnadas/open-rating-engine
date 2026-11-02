"""Pipelines and inputs shared by the scoring-plan tests and the Rust scorer's fixtures.

``kitchen_sink_pipeline`` uses every layer type and expression form the scoring plan
supports, on inputs with the awkward cases: missing numbers and text, keys that
don't match, values outside the bins, and an ONNX model fed text and integers.
``scorer/tests/make_fixtures.py`` writes its pandas output as the expected result
for the Rust scorer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext

from open_rating_engine.rating_engine import (
    ColumnCalculation,
    DropColumns,
    GlmScoring,
    OnnxScoring,
    PyfuncPricingPipeline,
    RatingTablesLayer,
    RenameColumns,
    SourceColumn,
    TableAggregation,
    TableDrop,
    TableJoin,
)

DATA_STRUCTURE = [
    {"name": "policy", "parent": None, "parent_key": None, "child_key": None, "cardinality": None},
    {"name": "risk", "parent": "policy", "parent_key": "policy_number", "child_key": "policy_number",
     "cardinality": "one_to_many"},
    {"name": "cover", "parent": "risk", "parent_key": "risk_number", "child_key": "risk_number",
     "cardinality": "one_to_many"},
]


def kitchen_sink_inputs() -> dict:
    """``{table: records}``, as a serving endpoint receives them."""
    nan = None
    return {
        "policy": [
            {"policy_number": "P1", "tenure": 0, "previous_gwp": 950.0, "region": "north"},
            {"policy_number": "P2", "tenure": 7, "previous_gwp": nan, "region": "south"},
            {"policy_number": "P3", "tenure": 12, "previous_gwp": 2400.5, "region": nan},
            {"policy_number": "P4", "tenure": 3, "previous_gwp": 0.0, "region": "north"},
        ],
        "risk": [
            {"policy_number": "P1", "risk_number": 1, "age": 19, "value": 15000.0, "usage": "Private"},
            {"policy_number": "P1", "risk_number": 2, "age": 44, "value": 23950.75, "usage": "Rideshare"},
            {"policy_number": "P2", "risk_number": 3, "age": 31, "value": 8000.0, "usage": nan},
            {"policy_number": "P3", "risk_number": 4, "age": 70, "value": 55000.0, "usage": "Private"},
            {"policy_number": "P3", "risk_number": 5, "age": 12, "value": 1234.5, "usage": "Commercial"},
            {"policy_number": "P3", "risk_number": 6, "age": 25, "value": 99999.99, "usage": "Private"},
            {"policy_number": "P9", "risk_number": 7, "age": 55, "value": 4100.0, "usage": "Private"},
        ],
        "cover": [
            {"risk_number": 1, "cover_type": "core", "limit": 1000},
            {"risk_number": 1, "cover_type": "hire_car", "limit": 250},
            {"risk_number": 2, "cover_type": "core", "limit": 1500},
            {"risk_number": 4, "cover_type": "windscreen", "limit": 300},
            {"risk_number": 4, "cover_type": "core", "limit": 2000},
            {"risk_number": 6, "cover_type": "core", "limit": 800},
        ],
    }


def write_tiny_onnx(path: Path) -> Path:
    """A small ONNX graph fed floats, text and integers-as-text:
    ``age * 0.5 + value * 0.001 + code(usage) + code(str(risk_number))``."""
    import onnx
    from onnx import TensorProto, helper

    def value_info(name, kind):
        return helper.make_tensor_value_info(name, kind, [None, 1])

    nodes = [
        helper.make_node("Mul", ["age", "half"], ["age_term"]),
        helper.make_node("Mul", ["value", "milli"], ["value_term"]),
        helper.make_node("LabelEncoder", ["usage"], ["usage_code"], domain="ai.onnx.ml",
                         keys_strings=["Private", "Rideshare", "None"], values_floats=[1.0, 2.5, -1.0],
                         default_float=0.0),
        helper.make_node("LabelEncoder", ["risk_id"], ["risk_code"], domain="ai.onnx.ml",
                         keys_strings=["1", "4", "7"], values_floats=[0.125, 0.25, 0.375], default_float=0.0),
        helper.make_node("Add", ["age_term", "value_term"], ["s1"]),
        helper.make_node("Add", ["s1", "usage_code"], ["s2"]),
        helper.make_node("Add", ["s2", "risk_code"], ["score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [value_info("age", TensorProto.FLOAT), value_info("value", TensorProto.FLOAT),
         value_info("usage", TensorProto.STRING), value_info("risk_id", TensorProto.STRING)],
        [value_info("score", TensorProto.FLOAT)],
        initializer=[
            helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("milli", TensorProto.FLOAT, [], [0.001]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 2)])
    # ONNX Runtime 1.20 (the Rust scorer's) reads IR versions up to 10.
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def kitchen_sink_pipeline(model_path: Path) -> PyfuncPricingPipeline:
    """Every exportable layer type and expression form, wired to ``model_path``."""
    context = PythonModelContext(
        artifacts={"tiny_model": str(model_path)},
        model_config={"environment": "local"},
    )
    p = PyfuncPricingPipeline(context)
    p.options["show_progress"] = False
    p.options["data_structure"] = DATA_STRUCTURE
    p.set_catalog("factors", pd.DataFrame({"name": ["base", "loading"], "value": [1.07, 0.15]}))
    p.set_catalog("region_factors", pd.DataFrame({
        "region": ["north", "south"], "region_factor": [1.1, 0.9], "tenure": [100, 200],
    }))

    def add(layer):
        p.add_parameter(layer)

    add(SourceColumn("risk.value", "risk", "value", dtype="numeric"))
    add(SourceColumn("risk.age", "risk", "age", dtype="numeric"))
    # Override a source column; the raw value stays readable as value{}.
    add(ColumnCalculation("value_override", "risk", "value", "df['value{}'] * 1.1",
                          dependencies=["risk.value"], override_source=True))
    add(ColumnCalculation("value_raw", "risk", "value_raw", "df['value{}']", dependencies=["value_override"]))
    add(ColumnCalculation("age_band", "risk", "age_band",
                          "pd.cut(df['age'], bins=list(range(15, 66, 10)) + [np.inf], "
                          "labels=['15-24', '25-34', '35-44', '45-54', '55-64', '65+'])"))
    add(ColumnCalculation("age_code", "risk", "age_code", "pd.cut(df['age'], bins=[0, 30, 60], labels=False)"))
    add(ColumnCalculation("is_private", "risk", "is_private", "np.where(df['usage'] == 'Private', 1, 0)"))
    add(ColumnCalculation("usage_missing", "risk", "usage_missing", "df['usage'].isna()"))
    add(ColumnCalculation("usage_filled", "risk", "usage_filled", "df['usage'].fillna('Unknown')"))
    add(ColumnCalculation("ratio", "risk", "ratio", "df['value'] / df['age']", dependencies=["value_override"]))
    add(ColumnCalculation("ratio_clipped", "risk", "ratio_clipped", "df['ratio'].clip(lower=100, upper=1000)",
                          dependencies=["ratio"]))
    add(ColumnCalculation("ratio_rounded", "risk", "ratio_rounded", "np.round(df['ratio'], 2)", dependencies=["ratio"]))
    add(ColumnCalculation("log_value", "risk", "log_value", "np.log(df['value']) + np.sqrt(df['age'])",
                          dependencies=["value_override"]))
    add(ColumnCalculation("based_value", "risk", "based_value",
                          "catalog['factors'].set_index('name').loc['base', 'value'] * df['value'] "
                          "+ options.get('flat_fee', 25)",
                          dependencies=["value_override"]))
    add(ColumnCalculation("older_private", "risk", "older_private",
                          "((df['age'] > 30) & (df['usage'] == 'Private') | ~(df['is_private'] == 1)).astype(int)",
                          dependencies=["is_private"]))
    add(ColumnCalculation("poly", "risk", "poly",
                          "df['age'] ** 2 - abs(df['value'] - 20000) / 1000 + -df['ratio'].astype(float)",
                          dependencies=["ratio"]))
    add(ColumnCalculation("rounded_ages", "risk", "rounded_ages", "np.floor(df['age'] / 7) + np.ceil(df['age'] / 7)"))
    add(ColumnCalculation("tenure", "risk", "tenure",
                          "aggregate(source='policy', source_column='tenure', agg_func='first')"))
    add(ColumnCalculation("region", "risk", "region",
                          "aggregate(source='policy', source_column='region', agg_func='first')"))
    for name, column, func in [
        ("risk_count", None, "count"), ("total_value", "value", "sum"), ("mean_value", "value", "mean"),
        ("max_age", "age", "max"), ("min_usage", "usage", "min"), ("first_usage", "usage", "first"),
        ("last_usage", "usage", "last"), ("usage_kinds", "usage", "nunique"), ("value_std", "value", "std"),
        ("value_median", "value", "median"), ("age_sum", "age", "sum"),
    ]:
        args = f"source='risk', agg_func='{func}'" + (f", source_column='{column}'" if column else "")
        add(ColumnCalculation(name, "policy", name, f"aggregate({args})", dependencies=["value_override"]))
    add(ColumnCalculation("capped", "policy", "capped",
                          "np.where(df['previous_gwp'].isna(), df['total_value'], "
                          "np.minimum(df['total_value'], df['previous_gwp'] * 1.2))",
                          dependencies=["total_value"]))
    add(ColumnCalculation("prev_or_zero", "policy", "prev_or_zero", "df['previous_gwp'].fillna(0)"))
    add(ColumnCalculation("cover_total", "risk", "cover_total",
                          "aggregate(source='cover', source_column='limit', agg_func='sum', fill_value=-1)"))
    add(RatingTablesLayer(
        "band_factor", "risk", "band_factor",
        rating_tables={
            "age_factor": pd.DataFrame({"age_band": ["15-24", "25-34", "35-44", "65+"], "value": [1.3, 1.0, 0.95, 1.2]}),
            "usage_factor": pd.DataFrame({"usage": ["Private", "Rideshare"], "value": [1.0, 1.4]}),
            "usage_add": pd.DataFrame({"usage": ["Rideshare"], "value": [0.05]}),
        },
        formula="age_factor['value'] * usage_factor['value'] + usage_add['value']",
        default_values={"age_factor": 1.15, "usage_factor": 1.0, "usage_add": 0.0},
        dependencies=["age_band"],
    ))
    add(RatingTablesLayer(
        "band_product", "risk", "band_product",
        rating_tables={
            "age_factor": pd.DataFrame({"age_band": ["15-24", "25-34"], "value": [2, 3]}),
            "usage_factor": pd.DataFrame({"usage": ["Private"], "value": [1.5]}),
        },
        dependencies=["age_band"],
    ))
    add(OnnxScoring(
        "model_score", "risk", "model_score",
        column_map={"age": "age", "value": "value", "usage": "usage", "risk_id": "risk_number"},
        model_artifact_key="tiny_model",
        dependencies=["value_override"],
    ))
    add(ColumnCalculation("model_scaled", "risk", "model_scaled", "df['model_score'] * 1.5 + 2",
                          dependencies=["model_score"]))
    add(ColumnCalculation("model_total", "policy", "model_total",
                          "aggregate(source='risk', source_column='model_score', agg_func='sum')",
                          dependencies=["model_score"]))
    add(ColumnCalculation("model_mixed", "risk", "model_mixed", "df['model_score'] * df['age']",
                          dependencies=["model_score"]))
    add(GlmScoring(
        "glm_frequency", "risk", "glm_frequency",
        intercept=-2.1, link="log",
        terms=[
            {"name": "value", "treatment": "log_transform", "coef": 0.12},
            {"name": "age2", "treatment": "raw", "coef": -0.004},
            {"name": "usage", "treatment": "as_factor", "levels": {"Private": 0.0, "Rideshare": 0.31}},
            {"name": "age", "treatment": "binned", "bin_edges": [15, 25, 40, 100],
             "levels": {"(14.999, 25.0]": 0.4, "(25.0, 40.0]": 0.1, "out_of_range": 0.7}},
        ],
        column_map={"value": "value", "age": "age", "usage": "usage"},
        derived=[{"name": "age2", "expression": "age * 2"}],
        dependencies=["value_override"],
    ))
    add(GlmScoring(
        "glm_legacy", "risk", "glm_legacy",
        intercept=0.3, link="logit",
        terms=[{"name": "age", "treatment": "raw", "coef": 0.01},
               {"name": "risk_number", "treatment": "as_factor", "levels": {"1": 0.2, "4": -0.3}}],
        offset_column="age_code",
        dependencies=["age_code"],
    ))
    add(TableJoin("join_region", "policy", "region_factors", left_on="region", right_on="region",
                  right_source="catalog"))
    add(TableJoin("join_region_keys", "risk", "region_factors", left_on="region", right_on="region",
                  columns=["region_factor"], right_source="catalog", dependencies=["region"]))
    add(TableAggregation("cover_agg", "cover", "cover_agg", groupby_key="risk_number",
                         aggregations={"limit_total": ("limit", "sum"), "n_covers": ("cover_type", "count"),
                                       "first_cover": ("cover_type", "first")}))
    add(TableJoin("join_cover", "risk", "cover_agg", left_on="risk_number", dependencies=["cover_agg", "join_region_keys"]))
    add(TableDrop("drop_cover_agg", "cover_agg", dependencies=["join_cover"]))
    add(ColumnCalculation("limit_or_zero", "risk", "limit_or_zero", "df['limit_total'].fillna(0)",
                          dependencies=["join_cover"]))
    add(RenameColumns("rename_risk", "risk", {"first_cover": "main_cover"}, dependencies=["join_cover"]))
    add(DropColumns("drop_usage_missing", "risk", ["usage_missing"], dependencies=["usage_missing", "rename_risk"]))
    return p
