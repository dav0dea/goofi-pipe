//! `/patch.gfi` — the patch as a file the BROWSER carries, in both directions.
//!
//! **Why this exists.** A container reaches only what was bind-mounted, and that boundary cannot be
//! lifted from inside — no `docker run` flag means "the whole host filesystem" on Linux, macOS and
//! Windows alike. The browser has no such limit: it runs ON the host, and its own Save/Open dialogs
//! reach anywhere the user can. So the bytes travel over HTTP and the browser does the filesystem
//! half. Nothing here is Docker-specific; it is simply the only door that is the same door
//! everywhere.
//!
//! **What it is NOT.** This is a copy in and a copy out, not a second save semantics. `save_path`
//! is untouched, so Ctrl-S keeps meaning "overwrite the file this patch came from" and never
//! silently retargets a download. That distinction is the whole reason an earlier browser-download
//! save was removed (see the `save` arm): it left the dirty flag standing and gave the save-path
//! design a second thing to carry.
//!
//! **Both directions reuse the ops.** Download packs through `archive::write_gfi` — the same
//! packer `save` uses — and upload runs the real `load` op against a staged copy. Neither route
//! holds its own idea of what a `.gfi` is, so neither can drift from the disk path.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::AppState;

/// The name the browser's Save dialog opens with. The open patch's own filename when it has one,
/// so re-exporting a patch offers to overwrite the file it came from rather than minting
/// `patch (3).gfi` — and a bare default when it has none.
fn download_name(state: &AppState) -> String {
    state
        .save_path
        .lock()
        .unwrap()
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "patch.gfi".into())
}

/// `GET /patch.gfi` — pack the open patch and hand it over.
///
/// Packed **under the graph lock**, exactly as `save` does: the manifest and the workspace must
/// describe one moment, and taking them separately would let a graph EDIT land between them. The pack
/// goes through a temp file rather than a `Vec` because `archive::write_gfi` writes to a path and
/// names its errors by that path — reusing it verbatim keeps one packer instead of two.
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
        // The patch could not be packed — this end's fault, not the caller's.
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

/// `POST /patch.gfi` — replace the open patch with the uploaded archive.
///
/// The bytes are staged to a temp file and handed to the **real `load` op**, so the upload inherits
/// every part of a load that is easy to get wrong: the fresh mount, the type registration that must
/// precede `load_doc`, the all-or-nothing swap, the history clear, the harness stop, the broadcast.
/// Reimplementing that here would be sixty lines of load-bearing ordering kept in step by hand.
///
/// `adopt: false` is the one difference, and it is the honest one: the file the user picked lives
/// on THEIR machine, at a path this process cannot name and must not invent. The staged copy is
/// deleted the moment the load returns, so adopting it would aim the next silent Ctrl-S at a file
/// that no longer exists.
pub(crate) async fn upload(State(state): State<AppState>, body: Bytes) -> Response {
    let tmp = std::env::temp_dir().join(format!("goofi-import-{}.gfi", crate::nonce_hex()));
    if let Err(e) = std::fs::write(&tmp, &body) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{}: {e}", tmp.display())).into_response();
    }
    let req = json!({
        "id": 0,
        "op": "load",
        "payload": { "path": tmp.to_string_lossy(), "adopt": false },
    });
    let reply = crate::dispatch(&state, &req.to_string());
    let _ = std::fs::remove_file(&tmp);

    // A refused load left the open patch untouched (the arm guarantees it), so the only thing to do
    // here is report why. 400, not 500: the overwhelmingly likely cause is the wrong file in a file
    // dialog, which is the caller's mistake and one they can act on.
    let error = reply
        .as_deref()
        .and_then(|r| serde_json::from_str::<Value>(r).ok())
        .and_then(|v| v.get("error").cloned());
    match error {
        None => (StatusCode::OK, "loaded\n").into_response(),
        Some(e) => {
            let msg = e.as_str().map(str::to_owned).unwrap_or_else(|| e.to_string());
            (StatusCode::BAD_REQUEST, format!("{msg}\n")).into_response()
        }
    }
}
