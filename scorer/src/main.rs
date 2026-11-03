//! `ors-scorer serve` runs the HTTP endpoint; `ors-scorer score` scores one request
//! from a file or stdin, for parity checks and debugging.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use ors_scorer::{server, Bundle, SCORER_VERSION};
use serde_json::{json, Value};

const USAGE: &str = "\
usage:
  ors-scorer serve [--bundle DIR] [--port PORT]
  ors-scorer score [--bundle DIR] [--input FILE]
  ors-scorer --version

--bundle defaults to $MODEL_DIR, then /app/model. --port defaults to $PORT, then 5001.
score reads {\"inputs\": {table: [records]}} from FILE or stdin and prints
{\"predictions\": {table: [records]}}.
$ORS_ONNX_THREADS sets ONNX Runtime's intra-op threads. The default, 1, keeps results
identical on every machine: tree ensembles sum in a thread-dependent order, so float32
outputs change with the thread count. 0 lets ONNX Runtime choose.
$ORS_MAX_BODY_MB caps the request size for serve (default 256).";

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ors-scorer: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else { return Err(USAGE.into()) };
    if command == "--version" {
        println!("ors-scorer {SCORER_VERSION}");
        return Ok(());
    }
    let mut bundle = std::env::var("MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/app/model"));
    let mut port: u16 = env_or("PORT", 5001);
    let mut input: Option<PathBuf> = None;
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--bundle" => bundle = PathBuf::from(value()?),
            "--port" => port = value()?.parse().map_err(|_| "--port must be a number".to_string())?,
            "--input" => input = Some(PathBuf::from(value()?)),
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let bundle = Bundle::load(&bundle, env_or("ORS_ONNX_THREADS", 1))?;
    match command.as_str() {
        "serve" => {
            let max_body = env_or("ORS_MAX_BODY_MB", 256usize) * 1024 * 1024;
            let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            runtime
                .block_on(server::serve(Arc::new(bundle), port, max_body))
                .map_err(|e| e.to_string())
        }
        "score" => {
            let mut text = String::new();
            match input {
                Some(path) => text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
                None => {
                    std::io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
                }
            }
            let request: Value = serde_json::from_str(&text).map_err(|e| format!("input is not JSON: {e}"))?;
            let inputs = request
                .get("inputs")
                .and_then(Value::as_object)
                .ok_or("input must be {\"inputs\": {table: [records]}}")?;
            let predictions = bundle.score(inputs).map_err(|e| e.to_string())?;
            println!("{}", json!({ "predictions": predictions }));
            Ok(())
        }
        other => Err(format!("unknown command {other}\n{USAGE}")),
    }
}
