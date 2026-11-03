//! The scoring plan written by `open_rating_engine.scoring_plan.exporter`.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer};

use crate::column::{Column, Table};

pub const PLAN_FORMAT: &str = "open-rating-plan";
pub const PLAN_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct Plan {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub source_accessors: Vec<(String, String)>,
    #[serde(default)]
    pub models: IndexMap<String, ModelRef>,
    pub steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
pub struct ModelRef {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Step {
    Column {
        layer: String,
        table: String,
        column: String,
        expr: Expr,
    },
    RatingTables {
        layer: String,
        table: String,
        column: String,
        tables: Vec<RatingTable>,
        formula: Option<Expr>,
    },
    Onnx {
        layer: String,
        table: String,
        column: String,
        model: String,
        inputs: Vec<(String, String)>,
    },
    Glm {
        layer: String,
        table: String,
        column: String,
        inputs: Option<Vec<(String, String)>>,
        derived: Vec<Derived>,
        eta: Expr,
        offset: Option<String>,
        link: Link,
    },
    Join {
        layer: String,
        left: String,
        right: JoinRight,
        left_on: Vec<String>,
        right_on: Vec<String>,
        output: String,
        columns: Option<Vec<String>>,
        suffixes: (String, String),
    },
    AggregateTable {
        layer: String,
        source: String,
        output: String,
        keys: Vec<String>,
        aggs: Vec<(String, String, AggFunc)>,
    },
    Rename {
        layer: String,
        table: String,
        columns: Vec<(String, String)>,
    },
    DropColumns {
        layer: String,
        table: String,
        columns: Vec<String>,
    },
    DropTables {
        layer: String,
        tables: Vec<String>,
    },
}

impl Step {
    pub fn layer(&self) -> &str {
        match self {
            Step::Column { layer, .. }
            | Step::RatingTables { layer, .. }
            | Step::Onnx { layer, .. }
            | Step::Glm { layer, .. }
            | Step::Join { layer, .. }
            | Step::AggregateTable { layer, .. }
            | Step::Rename { layer, .. }
            | Step::DropColumns { layer, .. }
            | Step::DropTables { layer, .. } => layer,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RatingTable {
    pub name: String,
    pub keys: Vec<String>,
    pub key_rows: Vec<Vec<serde_json::Value>>,
    pub value_dtype: String,
    pub values: Vec<PlanF64>,
    pub default: PlanF64,
}

#[derive(Debug, Deserialize)]
pub struct Derived {
    pub name: String,
    pub expr: Expr,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Link {
    Identity,
    Log,
    Logit,
    Inverse,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinRight {
    Table(String),
    Catalog(TableData),
}

/// A constant table embedded in the plan (a catalog table).
#[derive(Debug, Deserialize)]
pub struct TableData {
    pub columns: Vec<ColumnData>,
}

#[derive(Debug, Deserialize)]
pub struct ColumnData {
    pub name: String,
    pub dtype: String,
    pub values: Vec<serde_json::Value>,
}

impl TableData {
    pub fn to_table(&self) -> Result<Table, String> {
        let mut table = Table::default();
        for (i, col) in self.columns.iter().enumerate() {
            let column = col.to_column()?;
            if i == 0 {
                table.nrows = column.len();
            }
            table.set(&col.name, column)?;
        }
        Ok(table)
    }
}

impl ColumnData {
    fn to_column(&self) -> Result<Column, String> {
        let bad = |v: &serde_json::Value| format!("column '{}': bad {} value {v}", self.name, self.dtype);
        Ok(match self.dtype.as_str() {
            "f64" => Column::F64(self.values.iter().map(|v| json_f64(v).ok_or_else(|| bad(v))).collect::<Result<_, _>>()?),
            "f32" => Column::F32(
                self.values.iter().map(|v| json_f64(v).map(|x| x as f32).ok_or_else(|| bad(v))).collect::<Result<_, _>>()?,
            ),
            "i64" => Column::I64(self.values.iter().map(|v| v.as_i64().ok_or_else(|| bad(v))).collect::<Result<_, _>>()?),
            "bool" => Column::Bool(self.values.iter().map(|v| v.as_bool().ok_or_else(|| bad(v))).collect::<Result<_, _>>()?),
            "str" => Column::Str(
                self.values
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::Null => Ok(None),
                        serde_json::Value::String(s) => Ok(Some(s.clone())),
                        other => Err(bad(other)),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            other => return Err(format!("column '{}': unknown dtype '{other}'", self.name)),
        })
    }
}

/// A float in the plan: a JSON number, or "inf" / "-inf" / "nan".
#[derive(Debug, Clone, Copy)]
pub struct PlanF64(pub f64);

pub fn json_f64(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => match s.as_str() {
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => None,
        },
        _ => None,
    }
}

impl<'de> Deserialize<'de> for PlanF64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        json_f64(&value)
            .map(PlanF64)
            .ok_or_else(|| serde::de::Error::custom(format!("expected a number, \"inf\", \"-inf\" or \"nan\", got {value}")))
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "k", rename_all = "snake_case")]
pub enum Expr {
    Col {
        name: String,
    },
    Const {
        #[serde(flatten)]
        value: ConstValue,
        #[serde(default)]
        strong: bool,
    },
    Rt {
        table: String,
    },
    Arith {
        op: ArithOp,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Cmp {
        op: CmpOp,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Logic {
        op: LogicOp,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Not {
        x: Box<Expr>,
    },
    Neg {
        x: Box<Expr>,
    },
    Call {
        f: Func,
        args: Vec<Expr>,
        #[serde(default)]
        decimals: i32,
        lower: Option<Box<Expr>>,
        upper: Option<Box<Expr>>,
    },
    Cut {
        x: Box<Expr>,
        edges: Vec<PlanF64>,
        labels: Option<Vec<String>>,
        right: bool,
        include_lowest: bool,
    },
    Agg {
        source: String,
        func: AggFunc,
        column: Option<String>,
        target_key: String,
        source_key: String,
        fill: Box<Expr>,
    },
    BinLookup {
        x: Box<Expr>,
        edges: Vec<PlanF64>,
        right: bool,
        include_lowest: bool,
        values: Vec<PlanF64>,
        missing: PlanF64,
    },
    StrLookup {
        x: Box<Expr>,
        mapping: IndexMap<String, PlanF64>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "t", content = "v", rename_all = "snake_case")]
pub enum ConstValue {
    Int(i64),
    Float(PlanF64),
    Bool(bool),
    Str(String),
    Null,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Func {
    Where,
    Maximum,
    Minimum,
    Abs,
    Log,
    Exp,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Isna,
    Notna,
    Fillna,
    AstypeInt,
    AstypeFloat,
    Clip,
    ToNumeric,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AggFunc {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Median,
    Std,
    Nunique,
    First,
    Last,
}
