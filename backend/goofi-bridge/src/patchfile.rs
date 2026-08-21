//! `/patch.gfi` — the patch as a file the BROWSER carries, in both directions.
//!
//! A copy in and a copy out, not a second save semantics: `save_path` is untouched, so Ctrl-S keeps
//! meaning "overwrite the file this patch came from".

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::AppState;

/// The name the browser's Save dialog opens with: the open patch's own filename, or a default.
fn download_name(state: &AppState) -> String {
    state
        .save_path
        .lock()
        .unwrap()
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "patch.gfi".into())
}

/// `GET /patch.gfi` — pack the open patch and hand it over. Packed under the graph lock, so the
/// manifest and the workspace describe one moment.
pub(crate) async fn download(State(state): State<AppState>) -> Response {
    let mount = state.mount();
    let tmp = std::env::temp_dir().join(format!("goofi-export-{}.gfi", crate::nonce_hex()));
    // Scoped so the guard is gone before this function can yield — a std MutexGuard held across an
    // await makes the handler's future non-Send, and axum will not take it.
    let packed = {
        let g = state.graph.lock().unwrap();
        goofi_engine::archive::write_gfi(&tmp, &g.serialize(), &mount)
    }
    .and_then(|()| std::fs::read(&tmp).map_err(|e| format!("{}: {e}", tmp.display())));
    let _ = std::fs::remove_file(&tmp);

    match packed {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "application/octet-stream".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{}\"", download_name(&state)),
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

/// `POST /patch.gfi` — replace the open patch with the uploaded archive, through the real `load`
/// op. `adopt: false`, because the staged copy is deleted the moment the load returns.
pub(crate) async fn upload(State(state): State<AppState>, body: Bytes) -> Response {
    let tmp = std::env::temp_dir().join(format!("goofi-import-{}.gfi", crate::nonce_hex()));
    if let Err(e) = std::fs::write(&tmp, &body) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", tmp.display())).into_response();
    }
    let load = state.call("load", json!({ "path": tmp.to_string_lossy(), "adopt": false }), "upload");
    let _ = std::fs::remove_file(&tmp);

    match load {
        Ok(_) => (StatusCode::OK, "loaded\n").into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, format!("{e}\n")).into_response(),
    }
}
