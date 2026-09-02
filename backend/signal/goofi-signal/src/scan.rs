//! The signal engine's scan of one `nodes_signal/` folder: a `.py` file is probed in the
//! interpreter that will run it and registered on the tier its imports allow; an `.rs` file is
//! built through goofi-build — or found built — and loaded behind its version symbol.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use goofi_node::{Isolation, Scanned, ScannedType, Stamp};
use goofi_python::{Discovered, Discovery};
use goofi_signal_sdk::host::Loaded;

use crate::SignalEngine;

/// The interpreters the scan probes and runs with. The subprocess one is the caller's; the
/// free-threaded one is the one this build links, if any, and it is what routes a file in-process.
#[derive(Clone, Debug)]
pub struct Python {
    pub subproc: String,
    free_threaded: Option<String>,
}

impl Python {
    pub fn new(subproc: String) -> Python {
        Python { subproc, free_threaded: free_threaded() }
    }
}

/// What one file's probes decided, before anything is registered.
#[derive(Clone)]
pub(crate) enum Probed {
    InProcess(Discovered),
    Subprocess(Discovered),
    Unavailable(String),
}

impl SignalEngine {
    pub fn set_python(&mut self, python: Python) {
        self.python = Some(python);
    }
}

pub(crate) fn scan(engine: &mut SignalEngine, dir: &Path) -> Vec<ScannedType> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(e) => {
            eprintln!("failed to read {}: {e}", dir.display());
            return Vec::new();
        }
    };
    paths.sort();
    let rust: Vec<(PathBuf, String, Option<Stamp>)> = paths
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter_map(|p| goofi_node::type_name_of(p).map(|name| (p.clone(), name, stamp(p))))
        .collect();
    let paths: Vec<(PathBuf, String, Option<Stamp>)> = paths
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "py"))
        .filter_map(|p| goofi_node::type_name_of(&p).map(|name| (p.clone(), name, stamp(&p))))
        .collect();
    // Probes spawn an interpreter each, so a folder is probed a few files at a time; a file whose
    // stamp has not moved since its last probe is not probed again.
    let width = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
    let mut probes: Vec<Option<Probed>> = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(width) {
        let python = engine.python.clone();
        let cached: Vec<Option<Probed>> = chunk
            .iter()
            .map(|(p, _, s)| engine.probed.get(p).filter(|(seen, _)| Some(*seen) == *s).map(|(_, r)| r.clone()))
            .collect();
        std::thread::scope(|s| {
            let handles: Vec<_> = chunk
                .iter()
                .zip(&cached)
                .map(|((p, _, _), hit)| {
                    let python = python.as_ref();
                    s.spawn(move || hit.clone().unwrap_or_else(|| probe(p, python)))
                })
                .collect();
            for h in handles {
                probes.push(h.join().ok());
            }
        });
    }
    let mut out = Vec::new();
    for ((path, type_name, stamp), probed) in paths.into_iter().zip(probes) {
        let Some(probed) = probed else { continue };
        if let Some(s) = stamp {
            engine.probed.insert(path, (s, probed.clone()));
        }
        let outcome = engine.register(&type_name, probed);
        out.push(ScannedType { type_name, stamp, outcome });
    }
    for (path, type_name, stamp) in rust {
        let outcome = engine.register_rust(&path, &type_name);
        out.push(ScannedType { type_name, stamp, outcome });
    }
    out
}

impl SignalEngine {
    /// An `.rs` file: the artifact the prebuild left for these bytes, loaded — a library that will
    /// not load displaces a stale registration and greys the type out with the reason.
    fn register_rust(&mut self, path: &Path, type_name: &str) -> Scanned {
        let base = goofi_build::base_dir(&goofi_core::home::dir());
        let loaded = goofi_build::built(&goofi_build::SIGNAL, path, &base)
            .and_then(|artifact| self.load_rust(&artifact, type_name));
        match loaded {
            Ok(replaced) => Scanned::Registered { isolation: Isolation::Native, replaced },
            Err(reason) => {
                self.remove_dyn_type(type_name);
                Scanned::Unavailable(reason)
            }
        }
    }

    fn load_rust(&mut self, artifact: &Path, type_name: &str) -> Result<bool, String> {
        if !self.rust_loaded.contains_key(artifact) {
            let opened = goofi_build::open(artifact)?;
            let intro = goofi_node::parse_introspection(&opened.describe)?;
            if let Some(reason) = goofi_node::illegal_slot(&intro) {
                return Err(reason);
            }
            let manifest = goofi_node::leak_manifest(type_name.to_string(), &intro, "signal");
            let loaded = unsafe { Loaded::open(opened.library, manifest) }?;
            self.rust_loaded.insert(artifact.to_path_buf(), Arc::new(loaded));
        }
        let loaded = self.rust_loaded[artifact].clone();
        let manifest = loaded.manifest();
        let factory: goofi_signal_sdk::NodeFactory = Box::new(move |_| loaded.instantiate());
        Ok(self.register_dyn_type(manifest, factory, &goofi_node::NATIVE))
    }
}

impl SignalEngine {
    fn register(&mut self, type_name: &str, probed: Probed) -> Scanned {
        let subproc = self.python.as_ref().map(|p| p.subproc.clone()).unwrap_or_default();
        match probed {
            Probed::InProcess(d) => {
                let (manifest, factory, tier) = routed(d, &subproc);
                let isolation = tier.get();
                Scanned::Registered { isolation, replaced: self.register_dyn_type(manifest, factory, tier) }
            }
            Probed::Subprocess(d) => {
                let t = goofi_python::subproc::node_type_from(&subproc, d);
                let isolation = t.isolation.get();
                Scanned::Registered {
                    isolation,
                    replaced: self.register_dyn_type(t.manifest, t.factory, t.isolation),
                }
            }
            // The latest scan is the answer: a stale runtime type is displaced first.
            Probed::Unavailable(reason) => {
                self.remove_dyn_type(type_name);
                Scanned::Unavailable(reason)
            }
        }
    }
}

/// The GIL-gate router: a file whose imports keep the GIL disabled runs in-process, any other in
/// a subprocess. One interpreter per probe, because re-enabling the GIL is one-way.
fn probe(path: &Path, python: Option<&Python>) -> Probed {
    let Some(python) = python else {
        return Probed::Unavailable("no Python interpreter provisioned — run `cargo run -p goofi-init`".into());
    };
    if let Some(ft) = python.free_threaded.as_deref() {
        if let Discovery::Found(d) = in_process(path, ft) {
            if d.gil_safe {
                return Probed::InProcess(d);
            }
            // A re-enabled GIL falls through to the subprocess tier, as a failed probe does.
        }
    }
    match goofi_python::subproc::probe(path, &python.subproc) {
        Discovery::Found(d) => Probed::Subprocess(d),
        Discovery::Unavailable { reason, .. } => Probed::Unavailable(reason),
        Discovery::Skip => Probed::Unavailable("not a node file".into()),
    }
}

#[cfg(feature = "embed")]
fn in_process(path: &Path, ft: &str) -> Discovery {
    goofi_python::inproc::probe(path, ft)
}

#[cfg(feature = "embed")]
fn free_threaded() -> Option<String> {
    goofi_python::inproc::interpreter_path()
}

#[cfg(not(feature = "embed"))]
fn in_process(_path: &Path, _ft: &str) -> Discovery {
    Discovery::Skip
}

#[cfg(not(feature = "embed"))]
fn free_threaded() -> Option<String> {
    None
}

/// A discovered in-process type, registered ROUTED: its tier cell decides the tier at every
/// build, so the runtime GIL tripwire demoting it is all a re-route takes.
#[cfg(feature = "embed")]
fn routed(
    d: Discovered,
    subproc: &str,
) -> (&'static goofi_node::NodeManifest, goofi_signal_sdk::NodeFactory, &'static goofi_node::IsolationCell) {
    let t = goofi_python::routed_node_type(d, subproc);
    (t.manifest, t.factory, t.isolation)
}

#[cfg(not(feature = "embed"))]
fn routed(
    d: Discovered,
    subproc: &str,
) -> (&'static goofi_node::NodeManifest, goofi_signal_sdk::NodeFactory, &'static goofi_node::IsolationCell) {
    let t = goofi_python::subproc::node_type_from(subproc, d);
    (t.manifest, t.factory, t.isolation)
}

fn stamp(path: &Path) -> Option<Stamp> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.len(), m.modified().ok()?))
}
