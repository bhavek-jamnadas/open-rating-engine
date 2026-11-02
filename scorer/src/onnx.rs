//! ONNX models, scored with ONNX Runtime as `OnnxScoring` does.

use std::collections::HashMap;
use std::path::Path;

use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};

use crate::column::{py_float_repr, Column, Table};

pub struct Models {
    sessions: HashMap<String, Session>,
}

impl Models {
    pub fn empty() -> Models {
        Models { sessions: HashMap::new() }
    }

    /// Load each `{key: path}` model. `intra_threads` of 0 lets ONNX Runtime choose.
    pub fn load<'a>(models: impl IntoIterator<Item = (&'a str, &'a Path)>, intra_threads: usize) -> Result<Models, String> {
        let mut sessions = HashMap::new();
        for (key, path) in models {
            let mut builder = Session::builder().map_err(|e| e.to_string())?;
            if intra_threads > 0 {
                builder = builder.with_intra_threads(intra_threads).map_err(|e| e.to_string())?;
            }
            let session = builder
                .commit_from_file(path)
                .map_err(|e| format!("loading ONNX model '{key}' from {}: {e}", path.display()))?;
            sessions.insert(key.to_string(), session);
        }
        Ok(Models { sessions })
    }

    /// Score `table` with model `key`. `inputs` maps each model input name to the
    /// column that feeds it. Returns the model's first output, one value per row.
    pub fn score(&self, key: &str, table: &Table, inputs: &[(String, String)]) -> Result<Column, String> {
        let session = self.sessions.get(key).ok_or_else(|| format!("ONNX model '{key}' is not loaded"))?;
        let sources: HashMap<&str, &str> = inputs.iter().map(|(m, s)| (m.as_str(), s.as_str())).collect();
        let missing: Vec<String> = inputs
            .iter()
            .filter(|(_, src)| !table.columns.contains_key(src))
            .map(|(m, src)| format!("{src} (for input '{m}')"))
            .collect();
        if !missing.is_empty() {
            return Err(format!("source columns missing: {missing:?}"));
        }
        let output_type = session.outputs.first().and_then(|o| o.output_type.tensor_type()).ok_or("model has no tensor output")?;
        let n = table.nrows;
        if n == 0 {
            return empty_column(output_type);
        }
        let mut feed: Vec<(String, DynValue)> = Vec::with_capacity(session.inputs.len());
        for input in &session.inputs {
            let src = sources
                .get(input.name.as_str())
                .ok_or_else(|| format!("model input '{}' has no source column", input.name))?;
            let column = &table.columns[*src];
            let ty = match &input.input_type {
                ValueType::Tensor { ty, .. } => *ty,
                other => return Err(format!("model input '{}' is a {other:?}, not a tensor", input.name)),
            };
            feed.push((input.name.clone(), tensor(column, ty, &input.name)?));
        }
        let outputs = session.run(feed).map_err(|e| format!("ONNX scoring failed: {e}"))?;
        let out = &outputs[0];
        let column = match output_type {
            TensorElementType::Float32 => Column::F32(out.try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?.1.to_vec()),
            TensorElementType::Float64 => Column::F64(out.try_extract_raw_tensor::<f64>().map_err(|e| e.to_string())?.1.to_vec()),
            TensorElementType::Int64 => Column::I64(out.try_extract_raw_tensor::<i64>().map_err(|e| e.to_string())?.1.to_vec()),
            other => return Err(format!("model output type {other:?} is not supported")),
        };
        if column.len() != n {
            return Err(format!("model returned {} values for {n} rows", column.len()));
        }
        Ok(column)
    }
}

fn empty_column(ty: TensorElementType) -> Result<Column, String> {
    Ok(match ty {
        TensorElementType::Float32 => Column::F32(Vec::new()),
        TensorElementType::Float64 => Column::F64(Vec::new()),
        TensorElementType::Int64 => Column::I64(Vec::new()),
        other => return Err(format!("model output type {other:?} is not supported")),
    })
}

/// A `[rows, 1]` tensor of `column` cast to `ty`, as `.astype(numpy_dtype)` would.
fn tensor(column: &Column, ty: TensorElementType, name: &str) -> Result<DynValue, String> {
    let shape = [column.len(), 1];
    let err = |e: ort::Error| e.to_string();
    let bad = || format!("input '{name}': cannot convert a {} column to {ty:?}", column.dtype());
    Ok(match ty {
        TensorElementType::Float32 => {
            let v: Vec<f32> = match column {
                Column::F32(v) => v.clone(),
                Column::Str(_) => return Err(bad()),
                other => other.to_f64()?.into_iter().map(|x| x as f32).collect(),
            };
            Tensor::from_array((shape, v)).map_err(err)?.into_dyn()
        }
        TensorElementType::Float64 => {
            let v = column.to_f64().map_err(|_| bad())?;
            Tensor::from_array((shape, v)).map_err(err)?.into_dyn()
        }
        TensorElementType::Int64 | TensorElementType::Int32 => {
            let v: Vec<i64> = match column {
                Column::I64(v) => v.clone(),
                Column::Bool(v) => v.iter().map(|&b| b as i64).collect(),
                Column::F64(_) | Column::F32(_) => {
                    let f = column.to_f64()?;
                    if f.iter().any(|x| !x.is_finite()) {
                        return Err(format!("input '{name}': cannot convert missing values to integer"));
                    }
                    f.into_iter().map(|x| x.trunc() as i64).collect()
                }
                Column::Str(_) => return Err(bad()),
            };
            if ty == TensorElementType::Int32 {
                let v: Vec<i32> = v.into_iter().map(|x| x as i32).collect();
                Tensor::from_array((shape, v)).map_err(err)?.into_dyn()
            } else {
                Tensor::from_array((shape, v)).map_err(err)?.into_dyn()
            }
        }
        TensorElementType::Bool => {
            let v: Vec<bool> = match column {
                Column::Bool(v) => v.clone(),
                Column::Str(_) => return Err(bad()),
                other => other.to_f64()?.into_iter().map(|x| x != 0.0).collect(),
            };
            Tensor::from_array((shape, v)).map_err(err)?.into_dyn()
        }
        TensorElementType::String => {
            // numpy `astype(str)`: Python's str() of each value.
            let v: Vec<String> = match column {
                Column::Str(v) => v.iter().map(|s| s.clone().unwrap_or_else(|| "None".into())).collect(),
                Column::I64(v) => v.iter().map(|x| x.to_string()).collect(),
                Column::Bool(v) => v.iter().map(|&b| if b { "True".into() } else { "False".into() }).collect(),
                Column::F64(v) => v.iter().map(|&x| py_float_repr(x, false)).collect(),
                Column::F32(v) => v.iter().map(|&x| py_float_repr(x as f64, true)).collect(),
            };
            Tensor::from_string_array((shape, v.as_slice())).map_err(err)?.into_dyn()
        }
        other => return Err(format!("input '{name}': ONNX type {other:?} is not supported")),
    })
}
