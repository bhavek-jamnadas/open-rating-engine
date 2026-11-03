//! Runs plan steps: the pricing layers.

use std::cmp::Ordering;
use std::collections::HashMap;

use indexmap::IndexMap;

use crate::aggregate::aggregate_groups;
use crate::column::{group_rows, row_keys, Column, Data, Key, Table};
use crate::expr::{eval, Ctx};
use crate::onnx::Models;
use crate::plan::{AggFunc, Derived, Expr, Func, JoinRight, Link, Plan, RatingTable, Step};

/// A failure while scoring, with the layer it came from.
#[derive(Debug)]
pub struct LayerError {
    pub layer: String,
    pub message: String,
}

impl std::fmt::Display for LayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "layer '{}': {}", self.layer, self.message)
    }
}

/// Run every step of `plan` over `data`, as `PyfuncPricingPipeline.predict` does.
pub fn run_plan(plan: &Plan, models: &Models, mut data: Data) -> Result<Data, LayerError> {
    // Raw source values stay reachable as `<column>{}` while layers run.
    for (table, column) in &plan.source_accessors {
        if let Some(t) = data.get_mut(table) {
            let accessor = format!("{column}{{}}");
            if let (Some(col), false) = (t.columns.get(column).cloned(), t.columns.contains_key(&accessor)) {
                t.columns.insert(accessor, col);
            }
        }
    }
    for step in &plan.steps {
        run_step(step, models, &mut data).map_err(|message| LayerError { layer: step.layer().to_string(), message })?;
    }
    for table in data.values_mut() {
        table.columns.retain(|name, _| !name.ends_with("{}"));
    }
    Ok(data)
}

fn table<'a>(data: &'a Data, name: &str) -> Result<&'a Table, String> {
    data.get(name).ok_or_else(|| format!("table '{name}' not found in data"))
}

fn run_step(step: &Step, models: &Models, data: &mut Data) -> Result<(), String> {
    match step {
        Step::Column { table: t, column, expr, .. } => {
            let target = table(data, t)?;
            let ctx = Ctx { table: target, data, rating: None };
            let result = eval(expr, &ctx)?.into_column(target.nrows);
            data.get_mut(t).unwrap().set(column, result)
        }
        Step::RatingTables { table: t, column, tables, formula, .. } => {
            let result = rating_tables(table(data, t)?, data, tables, formula.as_ref())?;
            data.get_mut(t).unwrap().set(column, result)
        }
        Step::Onnx { table: t, column, model, inputs, .. } => {
            let result = models.score(model, table(data, t)?, inputs)?;
            data.get_mut(t).unwrap().set(column, result)
        }
        Step::Glm { table: t, column, inputs, derived, eta, offset, link, .. } => {
            let result = glm(table(data, t)?, data, inputs.as_deref(), derived, eta, offset.as_deref(), *link)?;
            data.get_mut(t).unwrap().set(column, result)
        }
        Step::Join { left, right, left_on, right_on, output, columns, suffixes, .. } => {
            let right_table = match right {
                JoinRight::Table(name) => table(data, name)?.clone(),
                JoinRight::Catalog(t) => t.to_table()?,
            };
            let merged = join(table(data, left)?, &right_table, left_on, right_on, columns.as_deref(), suffixes)?;
            data.insert(output.clone(), merged);
            Ok(())
        }
        Step::AggregateTable { source, output, keys, aggs, .. } => {
            let result = aggregate_table(table(data, source)?, keys, aggs)?;
            data.insert(output.clone(), result);
            Ok(())
        }
        Step::Rename { table: t, columns, .. } => {
            let map: HashMap<&str, &str> = columns.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let old = data.get_mut(t).ok_or_else(|| format!("table '{t}' not found in data"))?;
            let mut renamed = IndexMap::new();
            for (name, col) in std::mem::take(&mut old.columns) {
                let new = map.get(name.as_str()).map_or(name.clone(), |s| s.to_string());
                if renamed.insert(new.clone(), col).is_some() {
                    return Err(format!("renaming gives two columns named '{new}'"));
                }
            }
            old.columns = renamed;
            Ok(())
        }
        Step::DropColumns { table: t, columns, .. } => {
            let target = data.get_mut(t).ok_or_else(|| format!("table '{t}' not found in data"))?;
            for c in columns {
                if target.columns.shift_remove(c).is_none() {
                    return Err(format!("['{c}'] not found in axis"));
                }
            }
            Ok(())
        }
        Step::DropTables { tables, .. } => {
            for t in tables {
                if data.shift_remove(t).is_none() {
                    return Err(format!("table '{t}' not found in data."));
                }
            }
            Ok(())
        }
    }
}

fn rating_tables(target: &Table, data: &Data, tables: &[RatingTable], formula: Option<&Expr>) -> Result<Column, String> {
    let mut lookups: IndexMap<String, Column> = IndexMap::new();
    for rt in tables {
        let missing: Vec<&String> = rt.keys.iter().filter(|k| !target.columns.contains_key(*k)).collect();
        if !missing.is_empty() {
            return Err(format!("columns {missing:?} from rating table '{}' are missing", rt.name));
        }
        let key_cols: Vec<&Column> = rt.keys.iter().map(|k| target.column(k)).collect::<Result<_, _>>()?;
        let mut index: HashMap<Vec<Key>, usize> = HashMap::new();
        let mut text_keys = vec![false; rt.keys.len()];
        for (r, row) in rt.key_rows.iter().enumerate() {
            let key: Vec<Key> = row.iter().map(Key::from_json).collect::<Result<_, _>>()?;
            for (j, k) in key.iter().enumerate() {
                text_keys[j] |= k.is_str();
            }
            index.entry(key).or_insert(r);
        }
        // pandas refuses to merge a numeric column with text keys.
        for (j, col) in key_cols.iter().enumerate() {
            if col.is_numeric() && text_keys[j] {
                return Err(format!(
                    "You are trying to merge on {} and object columns for key '{}'",
                    col.dtype(),
                    rt.keys[j]
                ));
            }
        }
        let rows: Vec<Option<usize>> = row_keys(&key_cols, target.nrows).into_iter().map(|k| index.get(&k).copied()).collect();
        let default = rt.default.0;
        let column = if rt.value_dtype == "i64" && rows.iter().all(|r| r.is_some()) {
            let values: Vec<i64> = rt.values.iter().map(|v| v.0 as i64).collect();
            Column::I64(rows.iter().map(|r| values[r.unwrap()]).collect())
        } else {
            Column::F64(
                rows.iter()
                    .map(|r| {
                        let v = r.map_or(f64::NAN, |r| rt.values[r].0);
                        if v.is_nan() {
                            default
                        } else {
                            v
                        }
                    })
                    .collect(),
            )
        };
        lookups.insert(rt.name.clone(), column);
    }
    if let Some(formula) = formula {
        let ctx = Ctx { table: target, data, rating: Some(&lookups) };
        return Ok(eval(formula, &ctx)?.into_column(target.nrows));
    }
    // DataFrame.prod(axis=1): integer when every factor is, NaN skipped.
    if lookups.values().all(|c| matches!(c, Column::I64(_))) {
        let mut out = vec![1i64; target.nrows];
        for c in lookups.values() {
            if let Column::I64(v) = c {
                for (o, x) in out.iter_mut().zip(v) {
                    *o = o.wrapping_mul(*x);
                }
            }
        }
        return Ok(Column::I64(out));
    }
    let mut out = vec![1f64; target.nrows];
    for c in lookups.values() {
        for (o, x) in out.iter_mut().zip(c.to_f64()?) {
            if !x.is_nan() {
                *o *= x;
            }
        }
    }
    Ok(Column::F64(out))
}

fn glm(
    target: &Table,
    data: &Data,
    inputs: Option<&[(String, String)]>,
    derived: &[Derived],
    eta: &Expr,
    offset: Option<&str>,
    link: Link,
) -> Result<Column, String> {
    let mut working = match inputs {
        Some(map) => {
            let missing: Vec<String> = map
                .iter()
                .filter(|(_, src)| !target.columns.contains_key(src))
                .map(|(leaf, src)| format!("{src} (for '{leaf}')"))
                .collect();
            if !missing.is_empty() {
                return Err(format!("source columns missing: {missing:?}"));
            }
            let mut w = Table { nrows: target.nrows, columns: IndexMap::new() };
            for (leaf, src) in map {
                w.columns.insert(leaf.clone(), target.columns[src].clone());
            }
            w
        }
        None => target.clone(),
    };
    for d in derived {
        let ctx = Ctx { table: &working, data, rating: None };
        let value = eval(&d.expr, &ctx).map_err(|e| format!("formula column '{}': {e}", d.name))?;
        let column = value.into_column(working.nrows);
        working.set(&d.name, column)?;
    }
    let ctx = Ctx { table: &working, data, rating: None };
    let mut linear = eval(eta, &ctx)?.into_column(working.nrows).to_f64()?;
    if let Some(offset) = offset {
        target.column(offset).map_err(|_| format!("offset column '{offset}' missing"))?;
        let to_numeric = Expr::Call {
            f: Func::ToNumeric,
            args: vec![Expr::Col { name: offset.to_string() }],
            decimals: 0,
            lower: None,
            upper: None,
        };
        let numeric = eval(&to_numeric, &Ctx { table: target, data, rating: None })?;
        for (e, o) in linear.iter_mut().zip(numeric.into_column(target.nrows).to_f64()?) {
            *e += if o.is_nan() { 0.0 } else { o };
        }
    }
    let mu = linear
        .into_iter()
        .map(|e| match link {
            Link::Identity => e,
            Link::Log => e.exp(),
            Link::Logit => 1.0 / (1.0 + (-e).exp()),
            Link::Inverse => 1.0 / e,
        })
        .collect();
    Ok(Column::F64(mu))
}

/// `left.merge(right, how="left", left_on=..., right_on=..., suffixes=...)`.
fn join(
    left: &Table,
    right: &Table,
    left_on: &[String],
    right_on: &[String],
    columns: Option<&[String]>,
    suffixes: &(String, String),
) -> Result<Table, String> {
    let right_cols: Vec<String> = match columns {
        Some(cols) => right_on.iter().chain(cols.iter()).cloned().collect(),
        None => right.columns.keys().cloned().collect(),
    };
    for c in &right_cols {
        right.column(c)?;
    }
    let left_keys: Vec<&Column> = left_on.iter().map(|k| left.column(k)).collect::<Result<_, _>>()?;
    let right_keys: Vec<&Column> = right_on.iter().map(|k| right.column(k)).collect::<Result<_, _>>()?;
    for (l, r) in left_keys.iter().zip(&right_keys) {
        let has_text = |c: &Column, n: usize| matches!(c, Column::Str(_)) && (0..n).any(|i| !c.is_null(i));
        if (l.is_numeric() && has_text(r, right.nrows)) || (r.is_numeric() && has_text(l, left.nrows)) {
            return Err(format!("You are trying to merge on {} and {} columns", l.dtype(), r.dtype()));
        }
    }
    // pandas matches missing keys to each other in a merge.
    let mut index: HashMap<Vec<Key>, Vec<usize>> = HashMap::new();
    for (j, key) in row_keys(&right_keys, right.nrows).into_iter().enumerate() {
        index.entry(key).or_default().push(j);
    }
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for (i, key) in row_keys(&left_keys, left.nrows).into_iter().enumerate() {
        match index.get(&key) {
            Some(matches) => {
                for &j in matches {
                    left_rows.push(i);
                    right_rows.push(Some(j));
                }
            }
            None => {
                left_rows.push(i);
                right_rows.push(None);
            }
        }
    }
    // A key with the same name on both sides appears once, from the left.
    let shared_keys: Vec<&String> = left_on.iter().zip(right_on).filter(|(l, r)| l == r).map(|(l, _)| l).collect();
    let right_out: Vec<&String> = right_cols.iter().filter(|c| !shared_keys.contains(c)).collect();
    let overlap = |c: &String| left.columns.contains_key(c) && right_out.contains(&c);
    let mut out = Table { nrows: left_rows.len(), columns: IndexMap::new() };
    let add = |name: String, col: Column, out: &mut Table| -> Result<(), String> {
        if out.columns.insert(name.clone(), col).is_some() {
            return Err(format!("the merge produces two columns named '{name}'"));
        }
        Ok(())
    };
    for (name, col) in &left.columns {
        let new = if overlap(name) { format!("{name}{}", suffixes.0) } else { name.clone() };
        add(new, col.take(&left_rows), &mut out)?;
    }
    for name in right_out {
        let new = if left.columns.contains_key(name) { format!("{name}{}", suffixes.1) } else { name.clone() };
        add(new, right.columns[name].take_opt(&right_rows)?, &mut out)?;
    }
    Ok(out)
}

/// `df.groupby(keys, as_index=False).agg(**{out: (column, func)})`: one row per key,
/// sorted by key, missing keys dropped.
fn aggregate_table(source: &Table, keys: &[String], aggs: &[(String, String, AggFunc)]) -> Result<Table, String> {
    let key_cols: Vec<&Column> = keys.iter().map(|k| source.column(k)).collect::<Result<_, _>>()?;
    let (mut groups, _) = group_rows(&row_keys(&key_cols, source.nrows));
    groups.sort_by(|a, b| {
        for col in &key_cols {
            let ord = compare_values(col, a[0], b[0]);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
    let firsts: Vec<usize> = groups.iter().map(|g| g[0]).collect();
    let mut out = Table { nrows: groups.len(), columns: IndexMap::new() };
    for (name, col) in keys.iter().zip(&key_cols) {
        out.columns.insert(name.clone(), col.take(&firsts));
    }
    for (name, column, func) in aggs {
        let values = source.column(column)?;
        out.columns.insert(name.clone(), aggregate_groups(values, &groups, *func)?);
    }
    Ok(out)
}

fn compare_values(col: &Column, a: usize, b: usize) -> Ordering {
    match col {
        Column::Str(v) => v[a].cmp(&v[b]),
        Column::I64(v) => v[a].cmp(&v[b]),
        Column::Bool(v) => v[a].cmp(&v[b]),
        _ => {
            let v = col.to_f64().unwrap();
            v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal)
        }
    }
}
