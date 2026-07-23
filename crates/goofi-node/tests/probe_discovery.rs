//! Probe-based discovery over the real `goofi` extension. REQUIRES a python with
//! `goofi` importable; it finds one itself (the repo venvs), so these run on a plain
//! `cargo test`. Override with GOOFI_PYMOD_TEST_PYTHON. When no usable interpreter
//! exists they FAIL with an actionable message rather than skipping — a green run has
//! to mean the cross-language path actually ran (the goofi-subproc convention).

use std::path::Path;

use goofi_node::discover::{discover, discover_one, probe_introspect};
use goofi_node::Isolation;

/// The first interpreter that can `import goofi`: an explicit override, then the repo's two
/// provisioned venvs (either works — the probe only imports `goofi`), then the system python.
fn test_python() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cands: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("GOOFI_PYMOD_TEST_PYTHON") {
        if !p.is_empty() {
            cands.push(p);
        }
    }
    for venv in [".ftvenv", ".venv"] {
        cands.push(repo.join(venv).join("bin/python").to_string_lossy().into_owned());
    }
    cands.push("python3".to_string());
    for cand in &cands {
        let ok = std::process::Command::new(cand)
            .args(["-c", "import goofi"])
            // A host PYTHONPATH would shadow the candidate's own goofi and make this
            // probe disagree with the one under test, which strips it.
            .env_remove("PYTHONPATH")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return cand.clone();
        }
    }
    panic!(
        "no python with `goofi` importable (tried {cands:?}). Provision one with \
         ./scripts/provision-goofi-py.sh, or set GOOFI_PYMOD_TEST_PYTHON. Refusing to \
         silent-skip a cross-language test."
    )
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
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
fn missing_dep_greys_out_instead_of_crashing() {
    let py = test_python();
    // The probe import fails -> None, never a panic/crash.
    assert!(probe_introspect(&fixtures().join("missing_dep.py"), &py).is_none());
    assert!(discover_one(&fixtures().join("missing_dep.py"), &py, "python", Isolation::Subprocess).is_none());
}

#[test]
fn discover_dir_skips_hidden_and_broken() {
    let py = test_python();
    let found = discover(&fixtures(), &py, "python", Isolation::Subprocess).expect("scan");
    let names: Vec<_> = found.iter().map(|d| d.manifest.type_name).collect();
    assert!(names.contains(&"Negate"), "negate discovered: {names:?}");
    assert!(!names.iter().any(|n| *n == "Hidden"), "_hidden.py is skipped");
    assert!(!names.iter().any(|n| *n == "Broken"), "missing-dep node greyed out");
}

#[test]
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
