//! Probe-based discovery over the real `goofi` extension. REQUIRES a python with
//! `goofi` importable — set GOOFI_PYMOD_TEST_PYTHON to that interpreter (e.g. the
//! maturin-develop venv). The tests are `#[ignore]` so a plain `cargo test` stays
//! green without the venv; run them explicitly with `-- --ignored`. When run, they
//! error (never silent-skip) if the env var is unset/unusable, so a green run is a
//! real cross-language check (mirrors the goofi-subproc convention).

use std::path::Path;

use goofi_node::discover::{discover, discover_one, probe_introspect};
use goofi_node::Isolation;

fn test_python() -> String {
    let py = std::env::var("GOOFI_PYMOD_TEST_PYTHON").unwrap_or_default();
    assert!(
        !py.is_empty(),
        "set GOOFI_PYMOD_TEST_PYTHON to a python with `goofi` installed \
         (e.g. `maturin develop` venv); refusing to silent-skip"
    );
    // Sanity: `import goofi` must work in that interpreter.
    let out = std::process::Command::new(&py)
        .args(["-c", "import goofi"])
        .output()
        .expect("spawn probe python");
    assert!(
        out.status.success(),
        "`import goofi` failed in {py}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    py
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
#[ignore = "needs GOOFI_PYMOD_TEST_PYTHON"]
fn discovers_a_valid_node_with_declarations() {
    let py = test_python();
    let d = discover_one(&fixtures().join("negate.py"), &py, "python", Isolation::Subprocess)
        .expect("negate.py discovers");
    assert_eq!(d.manifest.type_name, "Negate");
    assert_eq!(d.manifest.doc, "Negate the input.");
    assert_eq!(d.manifest.inputs[0].name, "data");
    assert_eq!(d.manifest.outputs[0].name, "out");
}

#[test]
#[ignore = "needs GOOFI_PYMOD_TEST_PYTHON"]
fn missing_dep_greys_out_instead_of_crashing() {
    let py = test_python();
    // The probe import fails -> None, never a panic/crash.
    assert!(probe_introspect(&fixtures().join("missing_dep.py"), &py).is_none());
    assert!(discover_one(&fixtures().join("missing_dep.py"), &py, "python", Isolation::Subprocess).is_none());
}

#[test]
#[ignore = "needs GOOFI_PYMOD_TEST_PYTHON"]
fn discover_dir_skips_hidden_and_broken() {
    let py = test_python();
    let found = discover(&fixtures(), &py, "python", Isolation::Subprocess).expect("scan");
    let names: Vec<_> = found.iter().map(|d| d.manifest.type_name).collect();
    assert!(names.contains(&"Negate"), "negate discovered: {names:?}");
    assert!(!names.iter().any(|n| *n == "Hidden"), "_hidden.py is skipped");
    assert!(!names.iter().any(|n| *n == "Broken"), "missing-dep node greyed out");
}

#[test]
#[ignore = "needs GOOFI_PYMOD_TEST_PYTHON"]
fn probe_ignores_a_host_pythonpath() {
    // The probe interpreter must import ITS OWN installed goofi, not one leaked in via a host
    // `PYTHONPATH` (as `.cargo/config.toml` sets for the embedded FT interpreter). A poison
    // `goofi/` on PYTHONPATH — the shape of a cross-version site-packages — must NOT shadow the
    // probe's goofi. Without the env-strip the probe imports the poison and discovery silently
    // finds nothing; with it, discovery works regardless of the host PYTHONPATH.
    let py = test_python();
    let poison = std::env::temp_dir().join(format!("goofi_poison_{}", std::process::id()));
    std::fs::create_dir_all(poison.join("goofi")).unwrap();
    std::fs::write(
        poison.join("goofi").join("__init__.py"),
        "raise RuntimeError('a host-PYTHONPATH goofi must be ignored by the probe')\n",
    )
    .unwrap();

    // This test is the only one in its process that touches PYTHONPATH; post-fix every probe
    // Command strips it, so setting it here cannot leak into sibling tests.
    std::env::set_var("PYTHONPATH", &poison);
    let d = discover_one(&fixtures().join("negate.py"), &py, "python", Isolation::Subprocess);
    std::env::remove_var("PYTHONPATH");
    let _ = std::fs::remove_dir_all(&poison);

    assert!(d.is_some(), "discovery must ignore a poison goofi on the host PYTHONPATH");
}
