//! Typed columns and tables, with the pandas dtypes the plan needs.
//!
//! A pandas column is float64, float32 (ONNX outputs), int64, bool or object holding
//! strings. Missing values are NaN in float columns and `None` in text columns; int64
//! and bool columns cannot hold a missing value, so an operation that introduces one
//! (an unmatched join row, say) turns int64 into float64, as pandas does.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<Option<String>>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::F32(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Column::F64(_) => "float64",
            Column::F32(_) => "float32",
            Column::I64(_) => "int64",
            Column::Bool(_) => "bool",
            Column::Str(_) => "object",
        }
    }

    pub fn is_numeric(&self) -> bool {
        !matches!(self, Column::Str(_))
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            Column::F64(v) => v[i].is_nan(),
            Column::F32(v) => v[i].is_nan(),
            Column::I64(_) | Column::Bool(_) => false,
            Column::Str(v) => v[i].is_none(),
        }
    }

    /// The value at `i` as a hashable key, matching Python equality (`1 == 1.0 == True`).
    pub fn key(&self, i: usize) -> Key {
        match self {
            Column::F64(v) => Key::from_f64(v[i]),
            Column::F32(v) => Key::from_f64(v[i] as f64),
            Column::I64(v) => Key::Int(v[i]),
            Column::Bool(v) => Key::Int(v[i] as i64),
            Column::Str(v) => match &v[i] {
                Some(s) => Key::Str(s.clone()),
                None => Key::Null,
            },
        }
    }

    /// Rows `indices` of this column, in order.
    pub fn take(&self, indices: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::F64(indices.iter().map(|&i| v[i]).collect()),
            Column::F32(v) => Column::F32(indices.iter().map(|&i| v[i]).collect()),
            Column::I64(v) => Column::I64(indices.iter().map(|&i| v[i]).collect()),
            Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
        }
    }

    /// Rows `indices` of this column, where `None` is a missing value. int64 becomes
    /// float64 when anything is missing; bool cannot hold a missing value.
    pub fn take_opt(&self, indices: &[Option<usize>]) -> Result<Column, String> {
        if indices.iter().all(|i| i.is_some()) {
            let flat: Vec<usize> = indices.iter().map(|i| i.unwrap()).collect();
            return Ok(self.take(&flat));
        }
        Ok(match self {
            Column::F64(v) => Column::F64(indices.iter().map(|i| i.map_or(f64::NAN, |i| v[i])).collect()),
            Column::F32(v) => Column::F32(indices.iter().map(|i| i.map_or(f32::NAN, |i| v[i])).collect()),
            Column::I64(v) => Column::F64(indices.iter().map(|i| i.map_or(f64::NAN, |i| v[i] as f64)).collect()),
            Column::Str(v) => Column::Str(indices.iter().map(|i| i.and_then(|i| v[i].clone())).collect()),
            Column::Bool(_) => {
                return Err("a boolean column would gain missing values, which pandas stores as \
                            object dtype; this is not supported"
                    .into())
            }
        })
    }

    /// The column as float64 values (`true` is 1). Text columns are an error.
    pub fn to_f64(&self) -> Result<Vec<f64>, String> {
        Ok(match self {
            Column::F64(v) => v.clone(),
            Column::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Column::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Bool(v) => v.iter().map(|&x| x as i64 as f64).collect(),
            Column::Str(_) => return Err("expected a numeric column, got text".into()),
        })
    }
}

/// A hashable value with Python's equality: ints, integral floats and bools compare
/// equal by value; NaN and `None` are both `Null`.
#[derive(Clone, Debug)]
pub enum Key {
    Null,
    Int(i64),
    Float(f64),
    Str(String),
}

impl Key {
    pub fn from_f64(x: f64) -> Key {
        if x.is_nan() {
            Key::Null
        } else if x.fract() == 0.0 && x.abs() < 9.2e18 {
            Key::Int(x as i64)
        } else {
            Key::Float(x)
        }
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Key, String> {
        use serde_json::Value;
        Ok(match value {
            Value::Null => Key::Null,
            Value::Bool(b) => Key::Int(*b as i64),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Key::Int(i),
                None => Key::from_f64(n.as_f64().ok_or("key number out of range")?),
            },
            Value::String(s) => Key::Str(s.clone()),
            other => return Err(format!("unsupported key value {other}")),
        })
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Key::Str(_))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        match (self, other) {
            (Key::Null, Key::Null) => true,
            (Key::Int(a), Key::Int(b)) => a == b,
            (Key::Float(a), Key::Float(b)) => a == b,
            (Key::Str(a), Key::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Key::Null => 0u8.hash(state),
            Key::Int(i) => {
                1u8.hash(state);
                i.hash(state)
            }
            Key::Float(f) => {
                2u8.hash(state);
                (f + 0.0).to_bits().hash(state)
            }
            Key::Str(s) => {
                3u8.hash(state);
                s.hash(state)
            }
        }
    }
}

/// Row keys for `columns`, one `Vec<Key>` per row.
pub fn row_keys(columns: &[&Column], nrows: usize) -> Vec<Vec<Key>> {
    (0..nrows).map(|i| columns.iter().map(|c| c.key(i)).collect()).collect()
}

/// Group rows by key, in order of first appearance, dropping rows with a missing key
/// (pandas `groupby(dropna=True)`).
pub fn group_rows(keys: &[Vec<Key>]) -> (Vec<Vec<usize>>, HashMap<Vec<Key>, usize>) {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index: HashMap<Vec<Key>, usize> = HashMap::new();
    for (row, key) in keys.iter().enumerate() {
        if key.iter().any(|k| matches!(k, Key::Null)) {
            continue;
        }
        let g = *index.entry(key.clone()).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(row);
    }
    (groups, index)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub nrows: usize,
    pub columns: IndexMap<String, Column>,
}

impl Table {
    pub fn column(&self, name: &str) -> Result<&Column, String> {
        self.columns.get(name).ok_or_else(|| format!("column '{name}' not found"))
    }

    /// Add or replace a column. A replaced column keeps its position.
    pub fn set(&mut self, name: &str, column: Column) -> Result<(), String> {
        if column.len() != self.nrows {
            return Err(format!(
                "column '{name}' has {} values but the table has {} rows",
                column.len(),
                self.nrows
            ));
        }
        self.columns.insert(name.to_string(), column);
        Ok(())
    }
}

/// The tables of one scoring request, in pandas dict order: inputs first, then
/// tables created by layers in the order they were created.
pub type Data = IndexMap<String, Table>;

/// `repr()` of a Python float (also numpy's `str()` of a float64 or float32), from
/// the shortest decimal digits that round-trip.
pub fn py_float_repr(x: f64, single: bool) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = if single { format!("{:e}", x as f32) } else { format!("{:e}", x) };
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            let mut int_part = digits.chars().take(int_len).collect::<String>();
            while int_part.len() < int_len {
                int_part.push('0');
            }
            let frac: String = digits.chars().skip(int_len).collect();
            let frac = if frac.is_empty() { "0".to_string() } else { frac };
            format!("{sign}{int_part}.{frac}")
        } else {
            format!("{sign}0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let body = if rest.is_empty() { first.to_string() } else { format!("{first}.{rest}") };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{body}e{esign}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e20, "1e+20"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (1e16, "1e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (-2.5, "-2.5"),
            (12345.678, "12345.678"),
        ];
        for (x, expected) in cases {
            assert_eq!(py_float_repr(x, false), expected, "{x}");
        }
        assert_eq!(py_float_repr(0.1f32 as f64, true), "0.1");
    }

    #[test]
    fn keys_follow_python_equality() {
        assert_eq!(Key::from_f64(1.0), Key::Int(1));
        assert_eq!(Column::Bool(vec![true]).key(0), Key::Int(1));
        assert_ne!(Key::Str("1".into()), Key::Int(1));
        assert_eq!(Key::from_f64(f64::NAN), Key::Null);
    }
}
