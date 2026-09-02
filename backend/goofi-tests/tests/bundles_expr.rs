//! The eeg bundle's declared expression, on the real evaluator: `file` follows
//! `globals.goofi_home` and `me.params.playback.sample`, and editing the dropdown re-aims it.
//! Embed-gated for the evaluator; the node itself still runs on the subprocess tier.
#![cfg(feature = "embed")]

use std::path::Path;

use goofi_tests::{hex, install, j, require_python, shape, Goofi};

fn install_bundled(g: &Goofi, bundle: &str, file: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../node-bundles").join(bundle).join("nodes_signal").join(file);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the bundled node {}: {e}", path.display()));
    install(g, file, &source)
}

/// A tiny FIF written by mne itself, `channels` wide, into the samples folder.
fn write_recording(py: &str, path: &Path, channels: usize) {
    let names: Vec<String> = (0..channels).map(|c| format!("\"C{c}\"")).collect();
    let script = format!(
        r#"
import mne, numpy as np
sf = 128.0; t = np.arange(int(sf * 2)) / sf
x = np.stack([np.sin(2 * np.pi * 10 * t) for _ in range({channels})]) * 1e-6
mne.io.RawArray(x, mne.create_info([{names}], sf, "eeg"), verbose=False).save({path:?}, overwrite=True, verbose=False)
"#,
        names = names.join(", "),
        path = path.to_string_lossy()
    );
    let out = std::process::Command::new(py)
        .args(["-c", &script])
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .output()
        .expect("spawn python");
    assert!(out.status.success(), "mne could not write {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn the_playback_file_follows_goofi_home_and_the_sample_dropdown() {
    let py = require_python();
    let g = Goofi::new();
    g.state.graph.lock().unwrap().set_evaluator(std::sync::Arc::new(
        goofi_python::inproc::PyExprEvaluator::new().expect("the evaluator constructs"),
    ));
    // Where the expression will point: this machine's own samples folder, seeded with two
    // recordings of different widths so the frames say which one plays.
    let samples = goofi_core::home::dir().join("data").join("samples");
    std::fs::create_dir_all(&samples).unwrap();
    write_recording(&py.py, &samples.join("two_raw.fif"), 2);
    write_recording(&py.py, &samples.join("three_raw.fif"), 3);

    let ty = install_bundled(&g, "eeg", "eeg_playback.py");
    // The dropdown is aimed at a LOCAL file at birth: the default sample would download.
    let born = g.call("node add", j!({ "type": ty,
        "param": [{ "name": "playback/sample", "value": "two_raw.fif" }] }));
    let play = goofi_tests::Uid::from_hex(born["uid"].as_str().unwrap()).unwrap();
    let probe = g.probe(play, "out");

    // The DECLARED binding is live from birth, and its source is the authored spelling.
    let file = g.doc()["nodes"][hex(play)]["params"]["playback"]["file"].clone();
    assert_eq!(file["mode"], j!("expression"), "{file}");
    let source = file["expr"].as_str().unwrap_or_default();
    assert!(source.contains("globals.goofi_home") && source.contains("me.params.playback.sample"),
            "{source}");

    let d = g.until("the recording the expression aimed at to play", |g| {
        if let Some(e) = g.error(play) {
            panic!("{ty} failed instead of answering: {e}");
        }
        probe.latest().filter(|d| shape(d)[0] == 2)
    });
    assert_eq!(d.meta().sfreq(), Some(128.0), "the resolved path was loaded for real");

    // The dropdown edit re-binds the reader: the file follows `me.params.playback.sample`.
    g.set_param(play, "playback", "sample", "three_raw.fif");
    g.until("the file to follow the dropdown", |_| probe.latest().filter(|d| shape(d)[0] == 3));
}
