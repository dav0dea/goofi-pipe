//! Probe-based discovery over the real `goofi` extension. REQUIRES a python with
//! `goofi` importable; it finds one itself (the repo venvs), so these run on a plain
//! `cargo test`. Override with GOOFI_PYMOD_TEST_PYTHON. When no usable interpreter
//! exists they FAIL with an actionable message rather than skipping — a green run has
//! to mean the cross-language path actually ran (the `goofi-python` convention).

use std::path::Path;

use goofi_node::discover::{discover_one, probe_introspect, Discovery};
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
    for venv in [".gfivenv-ft", ".gfivenv"] {
        // Both layouts: a venv keeps its interpreter under `bin/` on unix and `Scripts/` on
        // Windows. Naming only one is not a near miss — it drops through to the `python3` below,
        // which on Windows is an App Execution Alias that answers with a Microsoft Store advert.
        for tail in ["bin/python", "Scripts/python.exe"] {
            cands.push(repo.join(venv).join(tail).to_string_lossy().into_owned());
        }
    }
    cands.push("python3".to_string());
    for cand in &cands {
        let ok = std::process::Command::new(cand)
            .args(["-c", "import goofi"])
            // A host PYTHONPATH would shadow the candidate's own goofi and make this
            // probe disagree with the one under test, which strips it.
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
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
         goofi once (it provisions both venvs at startup), or set GOOFI_PYMOD_TEST_PYTHON. Refusing to \
         silent-skip a cross-language test."
    )
}

/// The FREE-THREADED interpreter (the in-process tier's host), for the routing gate's SAFE half:
/// an explicit override, else the repo's `.gfivenv-ft`. Panics rather than skipping — this gate decides
/// which tier every Python node lands on, so leaving a branch uncovered is not acceptable.
fn ft_python() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cands: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("GOOFI_FT_PYTHON") {
        if !p.is_empty() {
            cands.push(p);
        }
    }
    for tail in ["bin/python", "Scripts/python.exe"] {
        cands.push(repo.join(".gfivenv-ft").join(tail).to_string_lossy().into_owned());
    }
    for cand in &cands {
        // Free-threaded builds are exactly the ones where sys._is_gil_enabled() is False.
        let ft = std::process::Command::new(cand)
            .args(["-c", "import sys; sys.exit(0 if not sys._is_gil_enabled() else 1)"])
            .env_remove("PYTHONPATH")
            .env_remove("PYTHONHOME")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ft {
            return cand.clone();
        }
    }
    panic!(
        "no free-threaded python found (tried {cands:?}). Provision one with \
         goofi once (it provisions both venvs at startup), or set GOOFI_FT_PYTHON."
    )
}

/// The GIL interpreter (the subprocess tier's host) — the repo's `.gfivenv`.
fn gil_python() -> String {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv = repo.join(".gfivenv");
    // `bin/` on unix, `Scripts/` on Windows — whichever this venv actually has; the conventional
    // name stands in when neither does, so the assertion below still names a path.
    let p = [venv.join("bin/python"), venv.join("Scripts/python.exe")]
        .into_iter()
        .find(|c| c.is_file())
        .unwrap_or_else(|| venv.join("bin/python"));
    let ok = std::process::Command::new(&p)
        .args(["-c", "import goofi"])
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "no `.gfivenv` python with goofi importable ({}). Run goofi once (it provisions both venvs at startup)", p.display());
    p.to_string_lossy().into_owned()
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn discovers_a_valid_node_with_declarations() {
    let py = test_python();
    let Discovery::Found(d) = discover_one(&fixtures().join("negate.py"), &py, "python", Isolation::Subprocess)
    else {
        panic!("negate.py discovers")
    };
    assert_eq!(d.manifest.type_name, "Negate");
    assert_eq!(d.manifest.doc, "Negate the input.");
    assert_eq!(d.manifest.inputs[0].name, "data");
    assert_eq!(d.manifest.outputs[0].name, "out");
}

#[test]
fn missing_dep_greys_out_instead_of_crashing() {
    let py = test_python();
    // The probe import fails -> the REASON, never a panic/crash — and the node is reported as
    // unavailable rather than silently skipped, so the palette can explain itself.
    let err = probe_introspect(&fixtures().join("missing_dep.py"), &py).unwrap_err();
    assert_eq!(err, "definitely_not_installed_pkg", "a missing import names the module");
    match discover_one(&fixtures().join("missing_dep.py"), &py, "python", Isolation::Subprocess) {
        Discovery::Unavailable { type_name, reason } => {
            assert_eq!(type_name, "MissingDep");
            assert_eq!(reason, "definitely_not_installed_pkg");
        }
        _ => panic!("a node whose dep is missing is Unavailable, not skipped"),
    }
    // A file that is not a node at all is a different outcome entirely.
    assert!(matches!(
        discover_one(&fixtures().join("_hidden.py"), &py, "python", Isolation::Subprocess),
        Discovery::Skip
    ));
}

#[test]
fn a_node_whose_import_prints_still_discovers() {
    // The probe child runs the node module AND emits the JSON payload, so a dependency that
    // greets stdout on import (the pygame banner is the canonical one) would prepend itself to
    // the payload: the child exits 0, the parse fails, and a perfectly good node is greyed out
    // with "malformed introspection". The child loop already solved this for itself
    // (goofi-pymod/src/serve.rs dup2's fd 1 to stderr before compiling the user module); the
    // probe must too, or every node with a chatty dep silently becomes unavailable.
    let py = test_python();
    let Discovery::Found(d) =
        discover_one(&fixtures().join("chatty.py"), &py, "python", Isolation::Subprocess)
    else {
        panic!("a node whose import prints to stdout still discovers")
    };
    assert_eq!(d.manifest.type_name, "Chatty");
    assert_eq!(d.manifest.doc, "A node whose dependency prints on import.");
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

    assert!(
        matches!(d, Discovery::Found(_)),
        "discovery must ignore a poison goofi on the host PYTHONPATH"
    );
}

#[test]
fn every_param_kind_carries_its_doc_across_the_probe() {
    // `doc=` is declared in Python, serialized by the wheel's introspect, and parsed back into
    // ParamDecl — one arm per param kind on each side. Only the Int arm was covered by any test
    // that a documented command actually runs, so all four are pinned here.
    let py = test_python();
    let Discovery::Found(d) =
        discover_one(&fixtures().join("documented.py"), &py, "python", Isolation::Subprocess)
    else {
        panic!("documented.py discovers")
    };
    let doc = |name: &str| {
        d.manifest
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("param `{name}` discovered"))
            .doc
    };
    assert_eq!(doc("count"), Some("how many"));
    assert_eq!(doc("gain"), Some("how loud"));
    assert_eq!(doc("enabled"), Some("whether to run"));
    assert_eq!(doc("mode"), Some("which mode"));
}

#[test]
fn the_probe_itself_is_the_gil_routing_gate() {
    // `Discovered.gil_safe` is the ONE oracle deciding which tier a Python node runs on: the probe
    // interpreter imports the module, constructs the class, and reports whether the GIL is still
    // disabled afterwards. Both directions are pinned, since a wrong answer either quarantines a
    // fast node forever or re-enables the GIL process-wide for every other in-process node.
    let node = fixtures().join("negate.py");

    let Discovery::Found(ft) = discover_one(&node, &ft_python(), "python", Isolation::InProcess) else {
        panic!("negate.py discovers on the free-threaded interpreter")
    };
    assert!(ft.gil_safe, "a free-threaded probe whose imports left the GIL disabled → in-process tier");

    let Discovery::Found(gil) = discover_one(&node, &gil_python(), "python", Isolation::Subprocess) else {
        panic!("negate.py discovers on the GIL interpreter")
    };
    assert!(!gil.gil_safe, "a GIL interpreter can never host in-process → subprocess tier");
}

#[test]
fn the_gil_sample_covers_the_config_hooks() {
    // The gate must read the GIL state AFTER the `config_*` hooks have run. A hook is exactly
    // where a node does its declaration-time imports (goofi-pymod's own `device_options.py`
    // fixture documents that as intended usage), and importing a C extension built without
    // free-threading support re-enables the GIL process-wide. Sampled before the hooks, such a
    // node routes to the in-process tier and only fails on its first tick.
    let Discovery::Found(d) =
        discover_one(&fixtures().join("gil_flip.py"), &ft_python(), "python", Isolation::InProcess)
    else {
        panic!("gil_flip.py discovers on the free-threaded interpreter")
    };
    assert!(!d.gil_safe, "a hook that re-enabled the GIL must route to the subprocess tier");
}
