//! Conformance: the scorer must reproduce the pandas pipeline's output exactly.
//!
//! Each directory under `tests/fixtures` is a plan bundle with `input.json` and the
//! pandas result in `expected.json`, written by `tests/make_fixtures.py`.

use std::path::Path;

use ors_scorer::Bundle;
use serde_json::Value;

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every difference between two `{table: [records]}` values, as readable lines.
fn differences(expected: &Value, actual: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let (Some(exp), Some(act)) = (expected.as_object(), actual.as_object()) else {
        return vec!["predictions are not objects".into()];
    };
    let exp_tables: Vec<&String> = exp.keys().collect();
    let act_tables: Vec<&String> = act.keys().collect();
    if exp_tables != act_tables {
        out.push(format!("tables: expected {exp_tables:?}, got {act_tables:?}"));
    }
    for (table, rows) in exp {
        let (Some(rows), Some(got)) = (rows.as_array(), act.get(table).and_then(Value::as_array)) else { continue };
        if rows.len() != got.len() {
            out.push(format!("{table}: expected {} rows, got {}", rows.len(), got.len()));
            continue;
        }
        for (i, (row, got_row)) in rows.iter().zip(got).enumerate() {
            let (row, got_row) = (row.as_object().unwrap(), got_row.as_object().unwrap());
            let cols: Vec<&String> = row.keys().collect();
            let got_cols: Vec<&String> = got_row.keys().collect();
            if cols != got_cols {
                out.push(format!("{table}[{i}] columns: expected {cols:?}, got {got_cols:?}"));
                continue;
            }
            for (col, value) in row {
                let other = &got_row[col];
                let same = match (value, other) {
                    (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
                    (a, b) => a == b,
                };
                if !same {
                    out.push(format!("{table}[{i}].{col}: expected {value}, got {other}"));
                }
            }
        }
    }
    out
}

#[test]
fn fixtures_match_pandas() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let mut dirs: Vec<_> = std::fs::read_dir(&root).expect("tests/fixtures").flatten().map(|e| e.path()).collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no fixtures in {}", root.display());
    let mut failures = Vec::new();
    for dir in dirs.iter().filter(|d| d.is_dir()) {
        let bundle = Bundle::load(dir, 1).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let input = read_json(&dir.join("input.json"));
        let expected = read_json(&dir.join("expected.json"));
        let actual = bundle.score(input["inputs"].as_object().unwrap()).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let diffs = differences(&expected["predictions"], &actual);
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        if !diffs.is_empty() {
            failures.push(format!("{name}: {} differences\n  {}", diffs.len(), diffs.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ")));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn layer_errors_name_the_layer() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("kitchen_sink");
    let bundle = Bundle::load(&dir, 1).unwrap();
    let inputs = serde_json::json!({"policy": [{"policy_number": "P1"}], "risk": [{"age": 30}], "cover": []});
    let err = bundle.score(inputs.as_object().unwrap()).unwrap_err().to_string();
    assert!(err.starts_with("layer '"), "{err}");
}
