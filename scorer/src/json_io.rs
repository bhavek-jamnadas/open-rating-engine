//! JSON records in and out, typed the way `pd.DataFrame(records)` types them.

use indexmap::IndexMap;
use serde_json::{Map, Number, Value};

use crate::column::{Column, Data, Table};

/// `{table: [record, ...]}` to tables. Column order is the order keys first appear;
/// a key missing from a record is a missing value.
pub fn records_to_data(inputs: &Map<String, Value>) -> Result<Data, String> {
    let mut data = Data::new();
    for (name, records) in inputs {
        let records = records.as_array().ok_or_else(|| format!("inputs['{name}'] must be a list of records"))?;
        data.insert(name.clone(), records_to_table(name, records)?);
    }
    Ok(data)
}

fn records_to_table(table: &str, records: &[Value]) -> Result<Table, String> {
    let mut cells: IndexMap<String, Vec<Option<&Value>>> = IndexMap::new();
    for (i, record) in records.iter().enumerate() {
        let record = record.as_object().ok_or_else(|| format!("inputs['{table}'][{i}] must be an object"))?;
        for (key, value) in record {
            let column = cells.entry(key.clone()).or_insert_with(|| vec![None; i]);
            column.push(Some(value));
        }
        for column in cells.values_mut() {
            column.resize(i + 1, None);
        }
    }
    let mut out = Table { nrows: records.len(), columns: IndexMap::new() };
    for (name, values) in cells {
        let column = infer_column(&values).map_err(|e| format!("inputs['{table}'] column '{name}': {e}"))?;
        out.columns.insert(name, column);
    }
    Ok(out)
}

/// pandas' dtype for a list of JSON values (`None` is a missing key).
fn infer_column(values: &[Option<&Value>]) -> Result<Column, String> {
    let present: Vec<&Value> = values.iter().flatten().copied().filter(|v| !v.is_null()).collect();
    let complete = present.len() == values.len();
    if present.is_empty() {
        return Ok(Column::F64(vec![f64::NAN; values.len()]));
    }
    if present.iter().all(|v| v.is_boolean()) {
        if !complete {
            return Err("booleans with missing values become object dtype, which is not supported".into());
        }
        return Ok(Column::Bool(present.iter().map(|v| v.as_bool().unwrap()).collect()));
    }
    if present.iter().all(|v| v.is_number()) {
        let ints: Option<Vec<i64>> = values.iter().map(|v| v.and_then(Value::as_i64)).collect();
        let all_int_literals = present.iter().all(|v| v.as_number().is_some_and(|n| n.is_i64()));
        if let (Some(ints), true) = (ints, all_int_literals) {
            return Ok(Column::I64(ints));
        }
        let mut out = Vec::with_capacity(values.len());
        for v in values {
            out.push(match v {
                Some(Value::Number(n)) => n.as_f64().ok_or("number out of range")?,
                _ => f64::NAN,
            });
        }
        return Ok(Column::F64(out));
    }
    if present.iter().all(|v| v.is_string()) {
        return Ok(Column::Str(values.iter().map(|v| v.and_then(Value::as_str).map(str::to_string)).collect()));
    }
    Err("mixes value types (object dtype), which is not supported".into())
}

/// Tables to `{table: [record, ...]}`, as `to_dict(orient="records")` plus the
/// serving container's rule that non-finite floats become null.
pub fn data_to_records(data: &Data) -> Value {
    let mut out = Map::new();
    for (name, table) in data {
        let mut rows = Vec::with_capacity(table.nrows);
        for i in 0..table.nrows {
            let mut row = Map::new();
            for (col, values) in &table.columns {
                row.insert(col.clone(), cell(values, i));
            }
            rows.push(Value::Object(row));
        }
        out.insert(name.clone(), Value::Array(rows));
    }
    Value::Object(out)
}

fn cell(column: &Column, i: usize) -> Value {
    match column {
        Column::F64(v) => Number::from_f64(v[i]).map_or(Value::Null, Value::Number),
        Column::F32(v) => Number::from_f64(v[i] as f64).map_or(Value::Null, Value::Number),
        Column::I64(v) => Value::Number(v[i].into()),
        Column::Bool(v) => Value::Bool(v[i]),
        Column::Str(v) => v[i].as_ref().map_or(Value::Null, |s| Value::String(s.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn table(records: Value) -> Table {
        records_to_table("t", records.as_array().unwrap()).unwrap()
    }

    #[test]
    fn dtypes_follow_pandas() {
        let t = table(json!([{"a": 1, "b": null, "c": "x"}, {"a": 2, "d": 1.5}]));
        assert_eq!(t.columns["a"], Column::I64(vec![1, 2]));
        assert!(matches!(&t.columns["b"], Column::F64(v) if v.iter().all(|x| x.is_nan())));
        assert_eq!(t.columns["c"], Column::Str(vec![Some("x".into()), None]));
        assert!(matches!(&t.columns["d"], Column::F64(v) if v[0].is_nan() && v[1] == 1.5));
        assert_eq!(table(json!([{"a": 1}, {"a": 2.0}])).columns["a"], Column::F64(vec![1.0, 2.0]));
    }
}
