//! `/patch.gfi` — the patch as a FILE the browser carries, in both directions.
//!
//! This is the door onto locations no mount reaches. A container sees only what was bind-mounted,
//! and that boundary cannot be lifted from inside; the browser, however, runs on the host and its
//! own file dialogs reach anywhere. So the bytes travel over HTTP and the browser does the
//! filesystem part.
//!
//! Deliberately NOT a `/control` op: this is a byte stream, and the ops registry describes JSON
//! request/reply pairs. What the route must not do is grow its own idea of what a `.gfi` is — both
//! directions go through the very same `archive` functions a disk save and a disk load use, so the
//! two routes cannot drift from the two ops.
//!

use goofi_tests::{host, http, j, tool, Goofi};

/// The whole point, end to end: a patch leaves one goofi as bytes and arrives in ANOTHER one as
/// the same graph. Two servers rather than one, because a round trip through the same instance
/// would pass against a route that returned a cached manifest and never packed anything.
#[tokio::test]
async fn a_patch_exported_as_bytes_loads_into_a_different_instance() {
    let src = Goofi::new();
    let source = host(&src.serve().await).to_string();
    let source = source.as_str();
    tool(source, "add_node", j!({ "type": "Oscillator" })).await;
    tool(source, "add_node", j!({ "type": "Buffer" })).await;

    let (status, head, gfi) = http(source, "GET", "/patch.gfi", "", b"").await;
    assert_eq!(status, 200, "{head}");
    assert_eq!(&gfi[..2], b"PK", "a .gfi is a zip, so it starts with the zip magic");
    assert!(
        head.to_ascii_lowercase().contains("content-disposition"),
        "the reply names a filename, or the browser saves it as `patch.gfi` from the URL: {head}"
    );

    // A DIFFERENT instance, whose graph starts empty.
    let dst = Goofi::new();
    let dest = host(&dst.serve().await).to_string();
    let dest = dest.as_str();
    let before = tool(dest, "inspect_patch", j!({})).await;
    assert!(!before.contains("Oscillator"), "the destination starts empty: {before}");

    let (status, head, _) = http(dest, "POST", "/patch.gfi", "Content-Type: application/octet-stream\r\n", &gfi).await;
    assert_eq!(status, 200, "{head}");

    let after = tool(dest, "inspect_patch", j!({})).await;
    assert!(after.contains("Oscillator"), "the imported patch is live: {after}");
    assert!(after.contains("Buffer"), "…all of it: {after}");

    // …and it did NOT adopt the staging file as its home. The upload was written to a temp path
    // and deleted the moment the load returned, so a patch that adopted it would aim the next
    // silent Ctrl-S at a file that no longer exists — overwriting nothing, or worse, recreating a
    // stray `.gfi` in the system temp directory. The patch's real home is on the USER's machine,
    // which this process cannot name, so "never saved" is the only honest answer.
    assert!(
        after.contains("(never saved)"),
        "an uploaded patch has no server-side home: {after}"
    );
}

/// A POST of something that is not an archive is a USER error — the wrong file in a file dialog —
/// so it answers 400 with a readable reason and leaves the running patch alone. Returning 500, or
/// wiping the graph on the way to failing, are the two ways this goes wrong.
#[tokio::test]
async fn a_post_that_is_not_an_archive_is_refused_and_changes_nothing() {
    let g = Goofi::new();
    let addr = host(&g.serve().await).to_string();
    let addr = addr.as_str();
    tool(addr, "add_node", j!({ "type": "Oscillator" })).await;

    let (status, head, body) = http(addr, "POST", "/patch.gfi", "Content-Type: application/octet-stream\r\n", b"not a zip").await;
    assert_eq!(status, 400, "a bad upload is the caller's error, not the server's: {head}");
    assert!(!body.is_empty(), "the refusal says why");

    let after = tool(addr, "inspect_patch", j!({})).await;
    assert!(after.contains("Oscillator"), "the live patch survived a refused upload: {after}");
}
