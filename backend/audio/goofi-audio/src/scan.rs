//! The audio engine's scan of one `nodes_audio/` folder: an `.rs` file is found built through
//! goofi-build — never built here — and loaded behind its version symbol; anything else in the
//! folder is named as not an audio node. A file whose stem is a built-in node's name is not a
//! node file, as a stem outside the name rule is not.

use std::path::Path;
use std::sync::Arc;

use goofi_audio_sdk::host::Loaded;
use goofi_node::{Isolation, Scanned, ScannedType};

use crate::nodes::Class;
use crate::AudioEngine;

pub(crate) fn scan(engine: &mut AudioEngine, dir: &Path) -> Vec<ScannedType> {
    let mut out = Vec::new();
    for (path, type_name, stamp) in goofi_node::node_files(dir) {
        if crate::nodes::built_in(&type_name) {
            continue;
        }
        let outcome = if path.extension().is_some_and(|e| e == "rs") {
            engine.register_rust(&path, &type_name)
        } else {
            Scanned::Unavailable("an audio node is an `.rs` file".into())
        };
        out.push(ScannedType { type_name, stamp, outcome });
    }
    out
}

impl AudioEngine {
    /// An `.rs` file: the artifact the prebuild left for these bytes, loaded — a library that will
    /// not load displaces a stale registration and greys the type out with the reason.
    fn register_rust(&mut self, path: &Path, type_name: &str) -> Scanned {
        let base = goofi_build::base_dir(&goofi_core::home::dir());
        let loaded = goofi_build::built(&goofi_build::AUDIO, path, &base)
            .and_then(|artifact| self.load_rust(&artifact, type_name));
        match loaded {
            Ok(replaced) => Scanned::Registered { isolation: Isolation::Native, replaced },
            Err(reason) => {
                self.classes.remove(type_name);
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
            let manifest = goofi_node::leak_manifest(type_name.to_string(), &intro, "audio");
            let loaded = unsafe { Loaded::open(opened.library, manifest) }?;
            self.rust_loaded.insert(artifact.to_path_buf(), Arc::new(loaded));
        }
        let loaded = self.rust_loaded[artifact].clone();
        let manifest = loaded.manifest();
        let class = Class { manifest, make: Arc::new(move |_| loaded.instantiate()) };
        Ok(self.classes.insert(manifest.type_name, class).is_some())
    }
}
