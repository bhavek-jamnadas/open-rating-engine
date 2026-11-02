//! Scores open-rating-engine scoring plans.
//!
//! A plan bundle is a directory with `plan.json` (written by
//! `open_rating_engine.scoring_plan.write_plan_bundle`) and the ONNX models it uses.
//! [`Bundle::score`] takes the same `{table: [records]}` inputs as the Python
//! serving container and returns the same `{table: [records]}` predictions.

pub mod aggregate;
pub mod column;
pub mod expr;
pub mod json_io;
pub mod onnx;
pub mod ops;
pub mod plan;
pub mod server;

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::onnx::Models;
use crate::ops::{run_plan, LayerError};
use crate::plan::{Plan, PLAN_FORMAT, PLAN_VERSION};

pub const SCORER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Files that mean the bundle expects checks this scorer does not perform.
const UNSUPPORTED_FILES: &[(&str, &str)] = &[(
    "input_validation.json",
    "input validation is not implemented in the Rust scorer, so it refuses to serve this bundle",
)];

pub struct Bundle {
    pub plan: Plan,
    pub models: Models,
    /// sha256 of `plan.json`, logged with every request.
    pub plan_hash: String,
    pub dir: PathBuf,
}

#[derive(Debug)]
pub enum ScoreError {
    /// The request could not be read as tables.
    Input(String),
    /// A layer failed, as a Python layer raising `LayerCalculationError` would.
    Layer(LayerError),
}

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ScoreError::Input(m) => write!(f, "invalid input: {m}"),
            ScoreError::Layer(e) => write!(f, "{e}"),
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

impl Bundle {
    /// Read and check a bundle directory. `onnx_threads` of 0 lets ONNX Runtime choose.
    pub fn load(dir: &Path, onnx_threads: usize) -> Result<Bundle, String> {
        for (file, reason) in UNSUPPORTED_FILES {
            if dir.join(file).exists() {
                return Err(format!("{}: {reason}", dir.join(file).display()));
            }
        }
        let plan_path = dir.join("plan.json");
        let bytes = std::fs::read(&plan_path).map_err(|e| format!("reading {}: {e}", plan_path.display()))?;
        let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", plan_path.display()))?;
        if plan.format != PLAN_FORMAT || plan.version != PLAN_VERSION {
            return Err(format!(
                "{} is '{}' version {}; this scorer reads '{PLAN_FORMAT}' version {PLAN_VERSION}",
                plan_path.display(),
                plan.format,
                plan.version
            ));
        }
        let mut model_paths = Vec::new();
        for (key, model) in &plan.models {
            let path = dir.join(&model.path);
            let data = std::fs::read(&path).map_err(|e| format!("model '{key}' at {}: {e}", path.display()))?;
            let digest = sha256_hex(&data);
            if digest != model.sha256 {
                return Err(format!("model '{key}' at {} does not match the plan's sha256", path.display()));
            }
            model_paths.push((key.clone(), path));
        }
        let models = Models::load(model_paths.iter().map(|(k, p)| (k.as_str(), p.as_path())), onnx_threads)?;
        Ok(Bundle { plan, models, plan_hash: sha256_hex(&bytes), dir: dir.to_path_buf() })
    }

    /// Score `{table: [records]}` and return `{table: [records]}`.
    pub fn score(&self, inputs: &Map<String, Value>) -> Result<Value, ScoreError> {
        let data = json_io::records_to_data(inputs).map_err(ScoreError::Input)?;
        let data = run_plan(&self.plan, &self.models, data).map_err(ScoreError::Layer)?;
        Ok(json_io::data_to_records(&data))
    }
}
