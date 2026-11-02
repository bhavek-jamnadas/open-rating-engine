//! Evaluates plan expressions with numpy/pandas semantics.
//!
//! Dtypes follow numpy 2 promotion: a Python scalar in an expression is "weak", so
//! `float32_column * 2.0` stays float32, while two columns promote (`int64` with
//! `float32` gives `float64`). Missing values follow pandas: NaN compares unequal to
//! everything, `None != "x"` is true, and `isna` sees both.

use indexmap::IndexMap;

use crate::aggregate::{aggregate_groups, map_groups};
use crate::column::{group_rows, py_float_repr, row_keys, Column, Data, Table};
use crate::plan::{ArithOp, CmpOp, ConstValue, Expr, Func, LogicOp, PlanF64};

/// What an expression can read: the target table, every table (for `aggregate()`)
/// and, inside a rating-table formula, the looked-up values.
pub struct Ctx<'a> {
    pub table: &'a Table,
    pub data: &'a Data,
    pub rating: Option<&'a IndexMap<String, Column>>,
}

/// A weak scalar: a Python constant in the expression.
#[derive(Clone, Debug)]
pub enum Scalar {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Null,
}

#[derive(Clone, Debug)]
pub enum Val {
    Col(Column),
    Scalar(Scalar),
}

/// Numeric dtypes in promotion order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Bool,
    I64,
    F32,
    F64,
}

/// How an operand takes part in promotion.
#[derive(Clone, Copy, Debug)]
enum Operand {
    Col(Kind),
    WeakBool,
    WeakInt,
    WeakFloat,
}

fn promote_cols(a: Kind, b: Kind) -> Kind {
    use Kind::*;
    match (a.min(b), a.max(b)) {
        (x, y) if x == y => x,
        (I64, F32) => F64,
        (_, y) => y,
    }
}

fn promote(a: Operand, b: Operand) -> Kind {
    use Operand::*;
    match (a, b) {
        (Col(x), Col(y)) => promote_cols(x, y),
        (Col(k), WeakBool) | (WeakBool, Col(k)) => k,
        (Col(k), WeakInt) | (WeakInt, Col(k)) => {
            if k == Kind::Bool {
                Kind::I64
            } else {
                k
            }
        }
        (Col(k), WeakFloat) | (WeakFloat, Col(k)) => {
            if k >= Kind::F32 {
                k
            } else {
                Kind::F64
            }
        }
        (WeakFloat, _) | (_, WeakFloat) => Kind::F64,
        (WeakBool, WeakBool) => Kind::Bool,
        _ => Kind::I64,
    }
}

impl Val {
    fn operand(&self) -> Result<Operand, String> {
        Ok(match self {
            Val::Col(Column::Bool(_)) => Operand::Col(Kind::Bool),
            Val::Col(Column::I64(_)) => Operand::Col(Kind::I64),
            Val::Col(Column::F32(_)) => Operand::Col(Kind::F32),
            Val::Col(Column::F64(_)) => Operand::Col(Kind::F64),
            Val::Scalar(Scalar::Bool(_)) => Operand::WeakBool,
            Val::Scalar(Scalar::Int(_)) => Operand::WeakInt,
            Val::Scalar(Scalar::Float(_)) | Val::Scalar(Scalar::Null) => Operand::WeakFloat,
            Val::Col(Column::Str(_)) | Val::Scalar(Scalar::Str(_)) => {
                return Err("expected a number, got text".into())
            }
        })
    }

    fn is_text(&self) -> bool {
        matches!(self, Val::Col(Column::Str(_)) | Val::Scalar(Scalar::Str(_)))
    }

    fn is_null_scalar(&self) -> bool {
        matches!(self, Val::Scalar(Scalar::Null))
    }

    fn f64s(&self, n: usize) -> Vec<f64> {
        match self {
            Val::Col(Column::F64(v)) => v.clone(),
            Val::Col(Column::F32(v)) => v.iter().map(|&x| x as f64).collect(),
            Val::Col(Column::I64(v)) => v.iter().map(|&x| x as f64).collect(),
            Val::Col(Column::Bool(v)) => v.iter().map(|&x| x as i64 as f64).collect(),
            Val::Scalar(s) => vec![scalar_f64(s); n],
            Val::Col(Column::Str(_)) => unreachable!("text checked by operand()"),
        }
    }

    fn f32s(&self, n: usize) -> Vec<f32> {
        match self {
            Val::Col(Column::F32(v)) => v.clone(),
            Val::Col(Column::Bool(v)) => v.iter().map(|&x| x as i64 as f32).collect(),
            Val::Scalar(s) => vec![scalar_f64(s) as f32; n],
            // Promotion never pairs float32 with int64 or float64 columns.
            _ => unreachable!("float32 result from a non-float32 column"),
        }
    }

    fn i64s(&self, n: usize) -> Vec<i64> {
        match self {
            Val::Col(Column::I64(v)) => v.clone(),
            Val::Col(Column::Bool(v)) => v.iter().map(|&x| x as i64).collect(),
            Val::Scalar(Scalar::Int(i)) => vec![*i; n],
            Val::Scalar(Scalar::Bool(b)) => vec![*b as i64; n],
            _ => unreachable!("int64 result from a float operand"),
        }
    }

    fn bools(&self, n: usize) -> Vec<bool> {
        match self {
            Val::Col(Column::Bool(v)) => v.clone(),
            Val::Scalar(Scalar::Bool(b)) => vec![*b; n],
            _ => unreachable!("bool result from a non-bool operand"),
        }
    }

    fn strs(&self, n: usize) -> Vec<Option<String>> {
        match self {
            Val::Col(Column::Str(v)) => v.clone(),
            Val::Scalar(Scalar::Str(s)) => vec![Some(s.clone()); n],
            Val::Scalar(Scalar::Null) => vec![None; n],
            _ => unreachable!("text result from a numeric operand"),
        }
    }

    fn as_kind(&self, kind: Kind, n: usize) -> Column {
        match kind {
            Kind::F64 => Column::F64(self.f64s(n)),
            Kind::F32 => Column::F32(self.f32s(n)),
            Kind::I64 => Column::I64(self.i64s(n)),
            Kind::Bool => Column::Bool(self.bools(n)),
        }
    }

    /// The value as a column of `n` rows (a scalar is repeated).
    pub fn into_column(self, n: usize) -> Column {
        match self {
            Val::Col(c) => c,
            Val::Scalar(Scalar::Int(i)) => Column::I64(vec![i; n]),
            Val::Scalar(Scalar::Float(x)) => Column::F64(vec![x; n]),
            Val::Scalar(Scalar::Bool(b)) => Column::Bool(vec![b; n]),
            Val::Scalar(Scalar::Str(s)) => Column::Str(vec![Some(s); n]),
            Val::Scalar(Scalar::Null) => Column::F64(vec![f64::NAN; n]),
        }
    }
}

fn scalar_f64(s: &Scalar) -> f64 {
    match s {
        Scalar::Int(i) => *i as f64,
        Scalar::Float(x) => *x,
        Scalar::Bool(b) => *b as i64 as f64,
        Scalar::Null => f64::NAN,
        Scalar::Str(_) => unreachable!("text checked by operand()"),
    }
}

fn const_scalar(value: &ConstValue) -> Scalar {
    match value {
        ConstValue::Int(i) => Scalar::Int(*i),
        ConstValue::Float(PlanF64(x)) => Scalar::Float(*x),
        ConstValue::Bool(b) => Scalar::Bool(*b),
        ConstValue::Str(s) => Scalar::Str(s.clone()),
        ConstValue::Null => Scalar::Null,
    }
}

pub fn eval(expr: &Expr, ctx: &Ctx) -> Result<Val, String> {
    let n = ctx.table.nrows;
    match expr {
        Expr::Col { name } => Ok(Val::Col(ctx.table.column(name)?.clone())),
        Expr::Const { value, strong } => {
            let scalar = const_scalar(value);
            if *strong {
                Ok(Val::Col(Column::F64(vec![scalar_f64(&scalar); n])))
            } else {
                Ok(Val::Scalar(scalar))
            }
        }
        Expr::Rt { table } => {
            let rating = ctx.rating.ok_or("rating table reference outside a rating-table formula")?;
            let col = rating.get(table).ok_or_else(|| format!("rating table '{table}' not found"))?;
            Ok(Val::Col(col.clone()))
        }
        Expr::Arith { op, l, r } => arith(*op, eval(l, ctx)?, eval(r, ctx)?, n),
        Expr::Cmp { op, l, r } => compare(*op, &eval(l, ctx)?, &eval(r, ctx)?, n),
        Expr::Logic { op, l, r } => logic(*op, eval(l, ctx)?, eval(r, ctx)?, n),
        Expr::Not { x } => match eval(x, ctx)? {
            Val::Col(Column::Bool(v)) => Ok(Val::Col(Column::Bool(v.into_iter().map(|b| !b).collect()))),
            Val::Col(Column::I64(v)) => Ok(Val::Col(Column::I64(v.into_iter().map(|i| !i).collect()))),
            Val::Scalar(Scalar::Bool(b)) => Ok(Val::Scalar(Scalar::Bool(!b))),
            _ => Err("~ needs a boolean or integer value".into()),
        },
        Expr::Neg { x } => match eval(x, ctx)? {
            Val::Col(Column::F64(v)) => Ok(Val::Col(Column::F64(v.into_iter().map(|x| -x).collect()))),
            Val::Col(Column::F32(v)) => Ok(Val::Col(Column::F32(v.into_iter().map(|x| -x).collect()))),
            Val::Col(Column::I64(v)) => Ok(Val::Col(Column::I64(v.into_iter().map(|x| x.wrapping_neg()).collect()))),
            Val::Scalar(Scalar::Int(i)) => Ok(Val::Scalar(Scalar::Int(i.wrapping_neg()))),
            Val::Scalar(Scalar::Float(x)) => Ok(Val::Scalar(Scalar::Float(-x))),
            _ => Err("unary minus needs a number".into()),
        },
        Expr::Call { f, args, decimals, lower, upper } => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval(a, ctx)?);
            }
            let lower = lower.as_ref().map(|e| eval(e, ctx)).transpose()?;
            let upper = upper.as_ref().map(|e| eval(e, ctx)).transpose()?;
            call(*f, vals, *decimals, lower, upper, n)
        }
        Expr::Cut { x, edges, labels, right, include_lowest } => {
            let x = eval(x, ctx)?.into_column(n);
            let edges: Vec<f64> = edges.iter().map(|e| e.0).collect();
            let ids = cut_ids(&x, &edges, *right, *include_lowest)?;
            Ok(Val::Col(match labels {
                Some(labels) => Column::Str(ids.iter().map(|id| id.map(|i| labels[i].clone())).collect()),
                None if ids.iter().all(|i| i.is_some()) => Column::I64(ids.iter().map(|i| i.unwrap() as i64).collect()),
                None => Column::F64(ids.iter().map(|i| i.map_or(f64::NAN, |i| i as f64)).collect()),
            }))
        }
        Expr::Agg { source, func, column, target_key, source_key, fill } => {
            let src = ctx.data.get(source).ok_or_else(|| format!("aggregate(): source table '{source}' not in data"))?;
            let target = ctx.table.column(target_key).map_err(|_| {
                format!("aggregate(): join key '{target_key}' missing on the target table")
            })?;
            let key_col = src
                .columns
                .get(source_key)
                .ok_or_else(|| format!("aggregate(): join key '{source_key}' missing on source table '{source}'"))?;
            let (groups, index) = group_rows(&row_keys(&[key_col], src.nrows));
            let per_group = match column {
                Some(c) => {
                    let values = src
                        .columns
                        .get(c)
                        .ok_or_else(|| format!("aggregate(): source_column '{c}' missing on source table '{source}'"))?;
                    aggregate_groups(values, &groups, *func)?
                }
                None => Column::I64(groups.iter().map(|g| g.len() as i64).collect()),
            };
            let positions: Vec<Option<usize>> = (0..n).map(|i| index.get(&vec![target.key(i)]).copied()).collect();
            let mapped = map_groups(&per_group, &positions)?;
            let fill = eval(fill, ctx)?;
            call(Func::Fillna, vec![Val::Col(mapped), fill], 0, None, None, n)
        }
        Expr::BinLookup { x, edges, right, include_lowest, values, missing } => {
            let x = eval(x, ctx)?.into_column(n);
            let edges: Vec<f64> = edges.iter().map(|e| e.0).collect();
            let ids = cut_ids(&x, &edges, *right, *include_lowest)?;
            Ok(Val::Col(Column::F64(
                ids.iter().map(|id| id.map_or(missing.0, |i| values[i].0)).collect(),
            )))
        }
        Expr::StrLookup { x, mapping } => {
            let x = eval(x, ctx)?.into_column(n);
            let out = (0..n)
                .map(|i| string_value(&x, i).and_then(|s| mapping.get(&s)).map_or(f64::NAN, |v| v.0))
                .collect();
            Ok(Val::Col(Column::F64(out)))
        }
    }
}

/// pandas `astype("string")` of one value: `None` for missing values.
fn string_value(col: &Column, i: usize) -> Option<String> {
    match col {
        Column::Str(v) => v[i].clone(),
        Column::I64(v) => Some(v[i].to_string()),
        Column::Bool(v) => Some(if v[i] { "True".into() } else { "False".into() }),
        Column::F64(v) => (!v[i].is_nan()).then(|| py_float_repr(v[i], false)),
        Column::F32(v) => (!v[i].is_nan()).then(|| py_float_repr(v[i] as f64, true)),
    }
}

/// pandas `pd.cut` bin index for each value; `None` outside the bins or missing.
pub fn cut_ids(x: &Column, edges: &[f64], right: bool, include_lowest: bool) -> Result<Vec<Option<usize>>, String> {
    let values = x.to_f64().map_err(|_| "pd.cut needs a numeric column".to_string())?;
    let nbins = edges.len();
    Ok(values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                return None;
            }
            // numpy searchsorted: side="left" when right=True, else side="right".
            let mut id = if right { edges.partition_point(|&e| e < v) } else { edges.partition_point(|&e| e <= v) };
            if include_lowest && v == edges[0] {
                id = 1;
            }
            if id == 0 || id == nbins {
                None
            } else {
                Some(id - 1)
            }
        })
        .collect())
}

fn arith(op: ArithOp, a: Val, b: Val, n: usize) -> Result<Val, String> {
    if a.is_text() || b.is_text() {
        return Err("arithmetic on text values is not supported".into());
    }
    let (oa, ob) = (a.operand()?, b.operand()?);
    if let (Val::Scalar(x), Val::Scalar(y)) = (&a, &b) {
        return scalar_arith(op, x, y);
    }
    let mut kind = promote(oa, ob);
    if kind == Kind::Bool {
        return Err("arithmetic between two boolean values is not supported".into());
    }
    if op == ArithOp::Div && kind == Kind::I64 {
        kind = Kind::F64;
    }
    Ok(Val::Col(match kind {
        Kind::F64 => {
            let (x, y) = (a.f64s(n), b.f64s(n));
            Column::F64(x.iter().zip(&y).map(|(&p, &q)| f64_op(op, p, q)).collect())
        }
        Kind::F32 => {
            let (x, y) = (a.f32s(n), b.f32s(n));
            Column::F32(x.iter().zip(&y).map(|(&p, &q)| f32_op(op, p, q)).collect())
        }
        Kind::I64 => {
            let (x, y) = (a.i64s(n), b.i64s(n));
            let mut out = Vec::with_capacity(n);
            for (&p, &q) in x.iter().zip(&y) {
                out.push(match op {
                    ArithOp::Add => p.wrapping_add(q),
                    ArithOp::Sub => p.wrapping_sub(q),
                    ArithOp::Mul => p.wrapping_mul(q),
                    ArithOp::Pow => {
                        if q < 0 {
                            return Err("Integers to negative integer powers are not allowed.".into());
                        }
                        p.wrapping_pow(q.min(u32::MAX as i64) as u32)
                    }
                    ArithOp::Div => unreachable!("integer division is promoted to float64"),
                });
            }
            Column::I64(out)
        }
        Kind::Bool => unreachable!(),
    }))
}

fn f64_op(op: ArithOp, p: f64, q: f64) -> f64 {
    match op {
        ArithOp::Add => p + q,
        ArithOp::Sub => p - q,
        ArithOp::Mul => p * q,
        ArithOp::Div => p / q,
        ArithOp::Pow => p.powf(q),
    }
}

fn f32_op(op: ArithOp, p: f32, q: f32) -> f32 {
    match op {
        ArithOp::Add => p + q,
        ArithOp::Sub => p - q,
        ArithOp::Mul => p * q,
        ArithOp::Div => p / q,
        ArithOp::Pow => p.powf(q),
    }
}

fn scalar_arith(op: ArithOp, a: &Scalar, b: &Scalar) -> Result<Val, String> {
    let int = |s: &Scalar| match s {
        Scalar::Int(i) => Some(*i),
        Scalar::Bool(b) => Some(*b as i64),
        _ => None,
    };
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        match op {
            ArithOp::Add => return Ok(Val::Scalar(Scalar::Int(x.wrapping_add(y)))),
            ArithOp::Sub => return Ok(Val::Scalar(Scalar::Int(x.wrapping_sub(y)))),
            ArithOp::Mul => return Ok(Val::Scalar(Scalar::Int(x.wrapping_mul(y)))),
            ArithOp::Pow if y >= 0 => return Ok(Val::Scalar(Scalar::Int(x.wrapping_pow(y as u32)))),
            _ => {}
        }
    }
    Ok(Val::Scalar(Scalar::Float(f64_op(op, scalar_f64(a), scalar_f64(b)))))
}

fn compare(op: CmpOp, a: &Val, b: &Val, n: usize) -> Result<Val, String> {
    let ordering = !matches!(op, CmpOp::Eq | CmpOp::Ne);
    let out: Vec<bool> = if a.is_text() && b.is_text() {
        let (x, y) = (a.strs(n), b.strs(n));
        let mut out = Vec::with_capacity(n);
        for (p, q) in x.iter().zip(&y) {
            out.push(match (p, q) {
                (Some(p), Some(q)) => cmp_result(op, p.cmp(q)),
                _ if ordering => return Err("ordering comparison with a missing text value".into()),
                _ => op == CmpOp::Ne,
            });
        }
        out
    } else if a.is_text() || b.is_text() {
        // Text never equals a number in pandas; ordering them is a TypeError.
        if ordering {
            return Err("cannot order text against numbers".into());
        }
        vec![op == CmpOp::Ne; n]
    } else {
        let integral = |v: &Val| {
            matches!(
                v,
                Val::Col(Column::I64(_)) | Val::Col(Column::Bool(_)) | Val::Scalar(Scalar::Int(_)) | Val::Scalar(Scalar::Bool(_))
            )
        };
        if integral(a) && integral(b) {
            let (x, y) = (a.i64s(n), b.i64s(n));
            x.iter().zip(&y).map(|(p, q)| cmp_result(op, p.cmp(q))).collect()
        } else {
            let (x, y) = (a.f64s(n), b.f64s(n));
            x.iter()
                .zip(&y)
                .map(|(p, q)| match p.partial_cmp(q) {
                    Some(ord) => cmp_result(op, ord),
                    None => op == CmpOp::Ne,
                })
                .collect()
        }
    };
    if let (Val::Scalar(_), Val::Scalar(_)) = (a, b) {
        return Ok(Val::Scalar(Scalar::Bool(out.first().copied().unwrap_or(false))));
    }
    Ok(Val::Col(Column::Bool(out)))
}

fn cmp_result(op: CmpOp, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match op {
        CmpOp::Eq => ord == Equal,
        CmpOp::Ne => ord != Equal,
        CmpOp::Lt => ord == Less,
        CmpOp::Le => ord != Greater,
        CmpOp::Gt => ord == Greater,
        CmpOp::Ge => ord != Less,
    }
}

fn logic(op: LogicOp, a: Val, b: Val, n: usize) -> Result<Val, String> {
    let is_bool = |v: &Val| matches!(v, Val::Col(Column::Bool(_)) | Val::Scalar(Scalar::Bool(_)));
    if !is_bool(&a) || !is_bool(&b) {
        return Err("& and | need boolean values".into());
    }
    let (x, y) = (a.bools(n), b.bools(n));
    Ok(Val::Col(Column::Bool(
        x.iter()
            .zip(&y)
            .map(|(&p, &q)| if op == LogicOp::And { p && q } else { p || q })
            .collect(),
    )))
}

/// numpy truthiness of a condition: non-zero, and NaN counts as true.
fn mask(v: &Val, n: usize) -> Result<Vec<bool>, String> {
    Ok(match v {
        Val::Col(Column::Bool(b)) => b.clone(),
        Val::Col(Column::I64(x)) => x.iter().map(|&i| i != 0).collect(),
        Val::Col(Column::F64(x)) => x.iter().map(|&f| f != 0.0).collect(),
        Val::Col(Column::F32(x)) => x.iter().map(|&f| f != 0.0).collect(),
        Val::Scalar(Scalar::Bool(b)) => vec![*b; n],
        Val::Scalar(Scalar::Int(i)) => vec![*i != 0; n],
        Val::Scalar(Scalar::Float(f)) => vec![*f != 0.0; n],
        _ => return Err("np.where condition must be boolean or numeric".into()),
    })
}

fn unary_float(v: Val, f64_fn: fn(f64) -> f64, f32_fn: fn(f32) -> f32, name: &str) -> Result<Val, String> {
    Ok(match v {
        Val::Col(Column::F32(x)) => Val::Col(Column::F32(x.into_iter().map(f32_fn).collect())),
        Val::Col(c @ (Column::F64(_) | Column::I64(_) | Column::Bool(_))) => {
            Val::Col(Column::F64(c.to_f64()?.into_iter().map(f64_fn).collect()))
        }
        Val::Scalar(s @ (Scalar::Int(_) | Scalar::Float(_) | Scalar::Bool(_) | Scalar::Null)) => {
            Val::Scalar(Scalar::Float(f64_fn(scalar_f64(&s))))
        }
        _ => return Err(format!("{name} needs a number")),
    })
}

fn call(f: Func, mut args: Vec<Val>, decimals: i32, lower: Option<Val>, upper: Option<Val>, n: usize) -> Result<Val, String> {
    let arity = match f {
        Func::Where => 3,
        Func::Maximum | Func::Minimum | Func::Fillna => 2,
        _ => 1,
    };
    if args.len() != arity {
        return Err(format!("{f:?} takes {arity} arguments, got {}", args.len()));
    }
    match f {
        Func::Where => {
            let c = mask(&args[0], n)?;
            let y = args.pop().unwrap();
            let x = args.pop().unwrap();
            if x.is_text() || y.is_text() {
                let text_or_null = |v: &Val| v.is_text() || v.is_null_scalar();
                if !text_or_null(&x) || !text_or_null(&y) {
                    return Err("np.where cannot mix text and numbers".into());
                }
                let (xs, ys) = (x.strs(n), y.strs(n));
                return Ok(Val::Col(Column::Str(
                    c.iter().zip(xs).zip(ys).map(|((&c, p), q)| if c { p } else { q }).collect(),
                )));
            }
            let kind = promote(x.operand()?, y.operand()?);
            let (xs, ys) = (x.as_kind(kind, n), y.as_kind(kind, n));
            Ok(Val::Col(select(&c, xs, ys)))
        }
        Func::Maximum | Func::Minimum => {
            let y = args.pop().unwrap();
            let x = args.pop().unwrap();
            let kind = promote(x.operand()?, y.operand()?);
            let is_max = f == Func::Maximum;
            Ok(Val::Col(match (x.as_kind(kind, n), y.as_kind(kind, n)) {
                (Column::F64(p), Column::F64(q)) => Column::F64(p.iter().zip(&q).map(|(&a, &b)| nan_minmax(a, b, is_max)).collect()),
                (Column::F32(p), Column::F32(q)) => Column::F32(
                    p.iter().zip(&q).map(|(&a, &b)| nan_minmax(a as f64, b as f64, is_max) as f32).collect(),
                ),
                (Column::I64(p), Column::I64(q)) => {
                    Column::I64(p.iter().zip(&q).map(|(&a, &b)| if is_max { a.max(b) } else { a.min(b) }).collect())
                }
                (Column::Bool(p), Column::Bool(q)) => {
                    Column::Bool(p.iter().zip(&q).map(|(&a, &b)| if is_max { a || b } else { a && b }).collect())
                }
                _ => unreachable!(),
            }))
        }
        Func::Abs => match args.pop().unwrap() {
            Val::Col(Column::I64(v)) => Ok(Val::Col(Column::I64(v.into_iter().map(|i| i.wrapping_abs()).collect()))),
            other => unary_float(other, f64::abs, f32::abs, "abs"),
        },
        Func::Log => unary_float(args.pop().unwrap(), f64::ln, f32::ln, "np.log"),
        Func::Exp => unary_float(args.pop().unwrap(), f64::exp, f32::exp, "np.exp"),
        Func::Sqrt => unary_float(args.pop().unwrap(), f64::sqrt, f32::sqrt, "np.sqrt"),
        Func::Floor | Func::Ceil => match args.pop().unwrap() {
            // numpy 2 keeps integer dtypes for floor/ceil.
            v @ Val::Col(Column::I64(_)) => Ok(v),
            other if f == Func::Floor => unary_float(other, f64::floor, f32::floor, "np.floor"),
            other => unary_float(other, f64::ceil, f32::ceil, "np.ceil"),
        },
        Func::Round => round(args.pop().unwrap(), decimals),
        Func::Isna | Func::Notna => {
            let v = args.pop().unwrap();
            let want = f == Func::Isna;
            Ok(match v {
                Val::Col(c) => Val::Col(Column::Bool((0..c.len()).map(|i| c.is_null(i) == want).collect())),
                Val::Scalar(s) => {
                    let null = matches!(s, Scalar::Null) || matches!(s, Scalar::Float(x) if x.is_nan());
                    Val::Scalar(Scalar::Bool(null == want))
                }
            })
        }
        Func::Fillna => {
            let fill = args.pop().unwrap();
            let x = args.pop().unwrap();
            fillna(x, fill)
        }
        Func::AstypeInt => {
            let col = args.pop().unwrap().into_column(n);
            Ok(Val::Col(match col {
                Column::I64(v) => Column::I64(v),
                Column::Bool(v) => Column::I64(v.into_iter().map(|b| b as i64).collect()),
                Column::F64(_) | Column::F32(_) => {
                    let v = col.to_f64()?;
                    if v.iter().any(|x| !x.is_finite()) {
                        return Err("Cannot convert non-finite values (NA or inf) to integer".into());
                    }
                    Column::I64(v.into_iter().map(|x| x.trunc() as i64).collect())
                }
                Column::Str(_) => return Err(".astype(int) on text is not supported".into()),
            }))
        }
        Func::AstypeFloat => {
            let col = args.pop().unwrap().into_column(n);
            Ok(Val::Col(Column::F64(col.to_f64().map_err(|_| ".astype(float) on text is not supported".to_string())?)))
        }
        Func::Clip => clip(args.pop().unwrap(), lower, upper, n),
        Func::ToNumeric => to_numeric(args.pop().unwrap()),
    }
}

fn nan_minmax(a: f64, b: f64, is_max: bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if is_max {
        a.max(b)
    } else {
        a.min(b)
    }
}

fn select(mask: &[bool], x: Column, y: Column) -> Column {
    match (x, y) {
        (Column::F64(p), Column::F64(q)) => Column::F64(mask.iter().zip(p).zip(q).map(|((&m, a), b)| if m { a } else { b }).collect()),
        (Column::F32(p), Column::F32(q)) => Column::F32(mask.iter().zip(p).zip(q).map(|((&m, a), b)| if m { a } else { b }).collect()),
        (Column::I64(p), Column::I64(q)) => Column::I64(mask.iter().zip(p).zip(q).map(|((&m, a), b)| if m { a } else { b }).collect()),
        (Column::Bool(p), Column::Bool(q)) => Column::Bool(mask.iter().zip(p).zip(q).map(|((&m, a), b)| if m { a } else { b }).collect()),
        _ => unreachable!("both branches share a dtype"),
    }
}

fn round(v: Val, decimals: i32) -> Result<Val, String> {
    // numpy: multiply by 10**decimals, round half to even, divide back.
    let factor = 10f64.powi(decimals.abs());
    let f64_round = |x: f64| {
        if decimals >= 0 {
            (x * factor).round_ties_even() / factor
        } else {
            (x / factor).round_ties_even() * factor
        }
    };
    Ok(match v {
        Val::Col(Column::F64(x)) => Val::Col(Column::F64(x.into_iter().map(f64_round).collect())),
        Val::Col(Column::F32(x)) => {
            let factor = factor as f32;
            Val::Col(Column::F32(
                x.into_iter()
                    .map(|v| if decimals >= 0 { (v * factor).round_ties_even() / factor } else { (v / factor).round_ties_even() * factor })
                    .collect(),
            ))
        }
        v @ Val::Col(Column::I64(_)) if decimals >= 0 => v,
        Val::Scalar(Scalar::Float(x)) => Val::Scalar(Scalar::Float(f64_round(x))),
        _ => return Err("round() needs a float column, or an int column with decimals >= 0".into()),
    })
}

fn fillna(x: Val, fill: Val) -> Result<Val, String> {
    if fill.is_null_scalar() || matches!(fill, Val::Scalar(Scalar::Float(f)) if f.is_nan()) {
        return Ok(x);
    }
    Ok(match (x, fill) {
        (Val::Col(Column::F64(v)), Val::Scalar(s @ (Scalar::Int(_) | Scalar::Float(_) | Scalar::Bool(_)))) => {
            let f = scalar_f64(&s);
            Val::Col(Column::F64(v.into_iter().map(|x| if x.is_nan() { f } else { x }).collect()))
        }
        (Val::Col(Column::F32(v)), Val::Scalar(s @ (Scalar::Int(_) | Scalar::Float(_) | Scalar::Bool(_)))) => {
            let f = scalar_f64(&s) as f32;
            Val::Col(Column::F32(v.into_iter().map(|x| if x.is_nan() { f } else { x }).collect()))
        }
        (Val::Col(Column::Str(v)), Val::Scalar(Scalar::Str(s))) => {
            Val::Col(Column::Str(v.into_iter().map(|x| x.or_else(|| Some(s.clone()))).collect()))
        }
        (v @ Val::Col(Column::I64(_) | Column::Bool(_)), _) => v,
        (Val::Scalar(Scalar::Null), s) => s,
        (Val::Scalar(Scalar::Float(x)), s) if x.is_nan() => s,
        (v @ Val::Scalar(_), _) => v,
        _ => return Err("fillna value would mix text and numbers".into()),
    })
}

fn clip(x: Val, lower: Option<Val>, upper: Option<Val>, n: usize) -> Result<Val, String> {
    let mut operand = x.operand()?;
    for bound in [&lower, &upper].into_iter().flatten() {
        operand = Operand::Col(promote(operand, bound.operand()?));
    }
    let kind = match operand {
        Operand::Col(Kind::Bool) | Operand::WeakBool | Operand::WeakInt => Kind::I64,
        Operand::Col(k) => k,
        Operand::WeakFloat => Kind::F64,
    };
    let bound = |b: &Option<Val>| b.as_ref().map(|v| match v {
        Val::Scalar(s) => scalar_f64(s),
        _ => f64::NAN,
    });
    let (lo, hi) = (bound(&lower), bound(&upper));
    let apply = |v: f64| {
        if v.is_nan() {
            return v;
        }
        let v = lo.map_or(v, |l| v.max(l));
        hi.map_or(v, |h| v.min(h))
    };
    Ok(Val::Col(match x.as_kind(kind, n) {
        Column::F64(v) => Column::F64(v.into_iter().map(apply).collect()),
        Column::F32(v) => Column::F32(v.into_iter().map(|x| apply(x as f64) as f32).collect()),
        Column::I64(v) => Column::I64(v.into_iter().map(|x| apply(x as f64) as i64).collect()),
        _ => unreachable!(),
    }))
}

fn to_numeric(v: Val) -> Result<Val, String> {
    match v {
        Val::Col(Column::Str(values)) => {
            let parsed: Vec<Option<f64>> = values.iter().map(|s| s.as_deref().and_then(parse_number)).collect();
            let ints: Option<Vec<i64>> = values.iter().map(|s| s.as_deref().and_then(|s| s.trim().parse::<i64>().ok())).collect();
            Ok(Val::Col(match ints {
                Some(ints) => Column::I64(ints),
                None => Column::F64(parsed.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect()),
            }))
        }
        Val::Scalar(Scalar::Str(s)) => Ok(Val::Scalar(parse_number(&s).map_or(Scalar::Null, Scalar::Float))),
        other => Ok(other),
    }
}

fn parse_number(s: &str) -> Option<f64> {
    let t = s.trim();
    match t.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => t.parse::<f64>().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(v: Vec<f64>) -> Val {
        Val::Col(Column::F64(v))
    }

    #[test]
    fn float32_stays_float32_with_weak_scalars() {
        let a = Val::Col(Column::F32(vec![0.1, 0.2]));
        let out = arith(ArithOp::Mul, a, Val::Scalar(Scalar::Float(2.0)), 2).unwrap();
        assert!(matches!(out, Val::Col(Column::F32(_))));
    }

    #[test]
    fn int_and_float32_columns_promote_to_float64() {
        let a = Val::Col(Column::F32(vec![0.5]));
        let b = Val::Col(Column::I64(vec![2]));
        assert!(matches!(arith(ArithOp::Mul, a, b, 1).unwrap(), Val::Col(Column::F64(_))));
    }

    #[test]
    fn int_division_is_true_division() {
        let a = Val::Col(Column::I64(vec![1, 0]));
        let b = Val::Col(Column::I64(vec![0, 0]));
        let Val::Col(Column::F64(v)) = arith(ArithOp::Div, a, b, 2).unwrap() else { panic!() };
        assert!(v[0].is_infinite() && v[1].is_nan());
    }

    #[test]
    fn nan_comparisons_follow_pandas() {
        let Val::Col(Column::Bool(v)) = compare(CmpOp::Ne, &col(vec![f64::NAN]), &Val::Scalar(Scalar::Int(0)), 1).unwrap() else {
            panic!()
        };
        assert_eq!(v, vec![true]);
    }

    #[test]
    fn cut_matches_pandas() {
        let x = Column::F64(vec![15.0, 25.0, 26.0, 70.0, f64::NAN, 14.0]);
        let ids = cut_ids(&x, &[15.0, 25.0, 35.0, f64::INFINITY], true, false).unwrap();
        assert_eq!(ids, vec![None, Some(0), Some(1), Some(2), None, None]);
        let ids = cut_ids(&Column::F64(vec![15.0]), &[15.0, 25.0], true, true).unwrap();
        assert_eq!(ids, vec![Some(0)]);
    }

    #[test]
    fn round_is_half_to_even_after_scaling() {
        let Val::Col(Column::F64(v)) = round(col(vec![2.675, 0.5, 1.5, 2.5]), 2).unwrap() else { panic!() };
        assert_eq!(v, vec![2.68, 0.5, 1.5, 2.5]);
        let Val::Col(Column::F64(v)) = round(col(vec![0.5, 1.5, 2.5]), 0).unwrap() else { panic!() };
        assert_eq!(v, vec![0.0, 2.0, 2.0]);
    }
}
