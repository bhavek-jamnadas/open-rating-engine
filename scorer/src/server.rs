//! The HTTP contract of the Python serving container: `GET /ping`, `POST /invocations`.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::{Bundle, ScoreError, SCORER_VERSION};

pub fn router(bundle: Arc<Bundle>, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/ping", get(|| async { "OK" }))
        .route("/invocations", post(invocations))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(bundle)
}

pub async fn serve(bundle: Arc<Bundle>, port: u16, max_body_bytes: usize) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    eprintln!(
        "ors-scorer {SCORER_VERSION} serving {} (plan {}) on port {port}",
        bundle.dir.display(),
        &bundle.plan_hash[..12]
    );
    axum::serve(listener, router(bundle, max_body_bytes))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

async fn invocations(State(bundle): State<Arc<Bundle>>, body: Bytes) -> Response {
    let t0 = Instant::now();
    let worker = bundle.clone();
    let result = tokio::task::spawn_blocking(move || handle(&worker, &body)).await;
    let (status, payload) = match result {
        Ok(outcome) => outcome,
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "internal", "detail": e.to_string()})),
    };
    // One JSON line per request, the shape the Python container logs, plus which
    // runtime and plan produced the price.
    let line = json!({
        "evt": "asterisk.invoke",
        "latency_ms": (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        "status": status.as_u16(),
        "runtime": "rust",
        "scorer_version": SCORER_VERSION,
        "plan_hash": bundle.plan_hash,
    });
    println!("{line}");
    (status, Json(payload)).into_response()
}

fn handle(bundle: &Bundle, body: &[u8]) -> (StatusCode, Value) {
    let request: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, json!({"error": "invalid_json", "detail": e.to_string()})),
    };
    let Some(inputs) = request.get("inputs").and_then(Value::as_object) else {
        return (StatusCode::BAD_REQUEST, json!({"error": "invalid_input", "detail": "body must be {\"inputs\": {table: [records]}}"}));
    };
    match bundle.score(inputs) {
        Ok(predictions) => (StatusCode::OK, json!({ "predictions": predictions })),
        Err(ScoreError::Input(m)) => (StatusCode::BAD_REQUEST, json!({"error": "invalid_input", "detail": m})),
        Err(ScoreError::Layer(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error": "layer_failed", "layer": e.layer, "detail": e.message}),
        ),
    }
}
