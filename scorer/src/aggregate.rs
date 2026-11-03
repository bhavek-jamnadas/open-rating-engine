//! Group aggregations with pandas `groupby` semantics.
//!
//! Missing values are skipped. Float sums and means use the same Kahan summation as
//! pandas' cython `group_sum` / `group_mean`, in the column's own precision, so
//! results match to the bit. A group with no values gives NaN, except `sum` (0) and
//! `count` / `nunique` (0).

use std::collections::HashSet;

use crate::column::{Column, Key};
use crate::plan::AggFunc;

/// One aggregated value per group. `groups` holds the row indices of each group.
pub fn aggregate_groups(values: &Column, groups: &[Vec<usize>], func: AggFunc) -> Result<Column, String> {
    match func {
        AggFunc::Count => {
            return Ok(Column::I64(groups.iter().map(|g| g.iter().filter(|&&i| !values.is_null(i)).count() as i64).collect()))
        }
        AggFunc::Nunique => {
            return Ok(Column::I64(
                groups
                    .iter()
                    .map(|g| g.iter().filter(|&&i| !values.is_null(i)).map(|&i| values.key(i)).collect::<HashSet<Key>>().len() as i64)
                    .collect(),
            ))
        }
        AggFunc::First | AggFunc::Last => return Ok(first_last(values, groups, func == AggFunc::First)),
        _ => {}
    }
    match values {
        Column::F64(v) => Ok(Column::F64(groups.iter().map(|g| float_agg(g.iter().map(|&i| v[i]), func)).collect())),
        Column::F32(v) => Ok(Column::F32(groups.iter().map(|g| float32_agg(g.iter().map(|&i| v[i]), func)).collect())),
        Column::I64(v) => int_agg(groups, |i| v[i], func),
        Column::Bool(v) => match func {
            AggFunc::Min | AggFunc::Max => {
                let is_max = func == AggFunc::Max;
                Ok(Column::Bool(groups.iter().map(|g| if is_max { g.iter().any(|&i| v[i]) } else { g.iter().all(|&i| v[i]) }).collect()))
            }
            _ => int_agg(groups, |i| v[i] as i64, func),
        },
        Column::Str(v) => match func {
            AggFunc::Min | AggFunc::Max => {
                let is_max = func == AggFunc::Max;
                Ok(Column::Str(
                    groups
                        .iter()
                        .map(|g| {
                            let it = g.iter().filter_map(|&i| v[i].as_ref());
                            if is_max { it.max().cloned() } else { it.min().cloned() }
                        })
                        .collect(),
                ))
            }
            _ => Err(format!("{func:?} cannot aggregate a text column")),
        },
    }
}

/// Per-group values placed onto rows; `None` positions are rows with no group.
pub fn map_groups(per_group: &Column, positions: &[Option<usize>]) -> Result<Column, String> {
    per_group.take_opt(positions)
}

fn first_last(values: &Column, groups: &[Vec<usize>], first: bool) -> Column {
    let pick = |g: &Vec<usize>| -> Option<usize> {
        if first {
            g.iter().copied().find(|&i| !values.is_null(i))
        } else {
            g.iter().rev().copied().find(|&i| !values.is_null(i))
        }
    };
    let rows: Vec<Option<usize>> = groups.iter().map(pick).collect();
    // A group with no value is NaN; int64 and bool columns never lack one.
    values.take_opt(&rows).expect("int and bool groups always have a value")
}

fn int_agg(groups: &[Vec<usize>], get: impl Fn(usize) -> i64, func: AggFunc) -> Result<Column, String> {
    Ok(match func {
        AggFunc::Sum => Column::I64(groups.iter().map(|g| g.iter().fold(0i64, |s, &i| s.wrapping_add(get(i)))).collect()),
        AggFunc::Min => Column::I64(groups.iter().map(|g| g.iter().map(|&i| get(i)).min().unwrap_or(0)).collect()),
        AggFunc::Max => Column::I64(groups.iter().map(|g| g.iter().map(|&i| get(i)).max().unwrap_or(0)).collect()),
        // pandas casts integers to float64 for these.
        _ => Column::F64(groups.iter().map(|g| float_agg(g.iter().map(|&i| get(i) as f64), func)).collect()),
    })
}

macro_rules! kahan {
    ($values:expr, $t:ty) => {{
        let mut sum: $t = 0.0;
        let mut comp: $t = 0.0;
        let mut count = 0usize;
        for v in $values {
            if v.is_nan() {
                continue;
            }
            count += 1;
            let y = v - comp;
            let t = sum + y;
            comp = t - sum - y;
            if comp.is_nan() {
                // pandas GH#53606: an infinite value leaves NaN compensation.
                comp = 0.0;
            }
            sum = t;
        }
        (sum, count)
    }};
}

fn float_agg(values: impl Iterator<Item = f64> + Clone, func: AggFunc) -> f64 {
    match func {
        AggFunc::Sum => kahan!(values, f64).0,
        AggFunc::Mean => {
            let (sum, count) = kahan!(values, f64);
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        }
        AggFunc::Min => values.filter(|v| !v.is_nan()).fold(f64::NAN, |m, v| if m.is_nan() || v < m { v } else { m }),
        AggFunc::Max => values.filter(|v| !v.is_nan()).fold(f64::NAN, |m, v| if m.is_nan() || v > m { v } else { m }),
        AggFunc::Median => median(values.filter(|v| !v.is_nan()).collect()),
        AggFunc::Std => welford_std(values.filter(|v| !v.is_nan())),
        _ => unreachable!("handled by aggregate_groups"),
    }
}

fn float32_agg(values: impl Iterator<Item = f32> + Clone, func: AggFunc) -> f32 {
    match func {
        AggFunc::Sum => kahan!(values, f32).0,
        AggFunc::Mean => {
            let (sum, count) = kahan!(values, f32);
            if count == 0 {
                f32::NAN
            } else {
                sum / count as f32
            }
        }
        // pandas computes the median in float64 and casts back.
        AggFunc::Median => median(values.filter(|v| !v.is_nan()).map(|v| v as f64).collect()) as f32,
        AggFunc::Std => {
            let mut n = 0f32;
            let (mut mean, mut m2) = (0f32, 0f32);
            for v in values.filter(|v| !v.is_nan()) {
                n += 1.0;
                let old = mean;
                mean += (v - old) / n;
                m2 += (v - mean) * (v - old);
            }
            if n <= 1.0 {
                f32::NAN
            } else {
                (m2 / (n - 1.0)).sqrt()
            }
        }
        _ => float_agg(values.map(|v| v as f64), func) as f32,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Sample standard deviation (ddof=1) with Welford's update, as pandas `group_var`.
fn welford_std(values: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0f64;
    let (mut mean, mut m2) = (0f64, 0f64);
    for v in values {
        n += 1.0;
        let old = mean;
        mean += (v - old) / n;
        m2 += (v - mean) * (v - old);
    }
    if n <= 1.0 {
        f64::NAN
    } else {
        (m2 / (n - 1.0)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kahan_sum_matches_pandas() {
        // pandas groupby sum gives 1.0 here; a naive or Neumaier sum differs.
        let col = Column::F64(vec![1e16, 1.0, -1e16, 1.0]);
        let out = aggregate_groups(&col, &[vec![0, 1, 2, 3]], AggFunc::Sum).unwrap();
        assert_eq!(out, Column::F64(vec![1.0]));
    }

    #[test]
    fn first_skips_missing_values() {
        let col = Column::F64(vec![f64::NAN, 2.0]);
        assert_eq!(aggregate_groups(&col, &[vec![0, 1]], AggFunc::First).unwrap(), Column::F64(vec![2.0]));
        let col = Column::Str(vec![None, Some("x".into())]);
        assert_eq!(aggregate_groups(&col, &[vec![0, 1]], AggFunc::First).unwrap(), Column::Str(vec![Some("x".into())]));
    }

    #[test]
    fn sum_of_all_missing_is_zero() {
        let col = Column::F64(vec![f64::NAN]);
        assert_eq!(aggregate_groups(&col, &[vec![0]], AggFunc::Sum).unwrap(), Column::F64(vec![0.0]));
    }
}
