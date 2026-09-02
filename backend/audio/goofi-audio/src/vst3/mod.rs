//! A `.vst3` bundle as a source of audio node types. A child `goofi` scans it — a plugin that
//! crashes at load takes the scanner down, never the server — and its classes become manifests
//! here, each hosted behind [`AudioNode`] by `node`.
// The bindings mirror the C++ headers, so a host object's trait methods carry their names.
#![allow(non_snake_case)]

mod host;
mod module;
mod node;

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use goofi_audio_sdk::AudioNode;
use goofi_core::probe;
use goofi_node::{Isolation, Scanned, ScannedType, Stamp};
use serde::{Deserialize, Serialize};
use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{ComPtr, ComWrapper};

use crate::nodes::Class;
use crate::AudioEngine;
pub(crate) use node::Derived;
use node::{Kind, Plugin};

/// A stepped parameter with this many steps or fewer is a `Str` of the plugin's own strings.
const STR_STEPS: i32 = 64;

/// What the scanner prints: the factory's vendor and every "Audio Module Class".
#[derive(Serialize, Deserialize)]
pub struct Bundle {
    pub vendor: String,
    pub classes: Vec<ClassInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ClassInfo {
    pub cid: String,
    pub name: String,
    /// Channel counts per audio bus, main first.
    pub inputs: Vec<u16>,
    pub outputs: Vec<u16>,
    pub events: bool,
    pub params: Vec<ParamInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamInfo {
    pub id: u32,
    pub title: String,
    pub units: String,
    pub steps: i32,
    pub default: f64,
    pub flags: i32,
    /// One display string per step when there are `STR_STEPS` or fewer; the default's alone
    /// otherwise.
    pub strings: Vec<String>,
}

/// Where this platform keeps its plugins.
pub fn platform_dirs() -> Vec<PathBuf> {
    let home = std::env::home_dir();
    #[cfg(target_os = "linux")]
    let dirs = [home.map(|h| h.join(".vst3")), Some("/usr/lib/vst3".into()), Some("/usr/local/lib/vst3".into())];
    #[cfg(target_os = "macos")]
    let dirs = [home.map(|h| h.join("Library/Audio/Plug-Ins/VST3")), Some("/Library/Audio/Plug-Ins/VST3".into()), None];
    #[cfg(windows)]
    let dirs = [
        std::env::var_os("COMMONPROGRAMFILES").map(|p| PathBuf::from(p).join("VST3")),
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Programs").join("Common").join("VST3")),
        home.and(None),
    ];
    dirs.into_iter().flatten().collect()
}

/// `goofi vst3-scan <bundle>`: the child half. The bundle's classes as JSON on stdout, or the
/// reason on stderr and a non-zero exit.
pub fn scan_main(args: &[String]) -> i32 {
    let Some(bundle) = args.first() else {
        eprintln!("usage: goofi vst3-scan <bundle>");
        return 2;
    };
    match describe(Path::new(bundle)) {
        Ok(found) => {
            println!("{}", serde_json::to_string(&found).expect("plain data serializes"));
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// The binary inside a bundle, laid out as the platform lays it — or the bundle itself where it
/// is one file.
fn binary_of(bundle: &Path) -> Result<PathBuf, String> {
    if bundle.is_file() {
        return Ok(bundle.to_path_buf());
    }
    let stem = bundle.file_stem().and_then(|s| s.to_str()).ok_or("a bundle has a name")?;
    let arch = std::env::consts::ARCH;
    let (folder, file) = if cfg!(target_os = "macos") {
        ("MacOS".to_string(), stem.to_string())
    } else if cfg!(windows) {
        (format!("{arch}-win"), format!("{stem}.vst3"))
    } else {
        (format!("{arch}-linux"), format!("{stem}.so"))
    };
    let path = bundle.join("Contents").join(folder).join(file);
    path.is_file().then_some(path).ok_or_else(|| format!("no binary at {}", bundle.display()))
}

fn stamp_of(binary: &Path) -> Result<Stamp, String> {
    let meta = std::fs::metadata(binary).map_err(|e| format!("{}: {e}", binary.display()))?;
    Ok((meta.len(), meta.modified().map_err(|e| e.to_string())?))
}

fn describe(bundle: &Path) -> Result<Bundle, String> {
    let factory = module::Module::open(&binary_of(bundle)?)?.factory()?;
    let mut classes = Vec::new();
    for (cid, name) in factory.audio_classes() {
        classes.push(describe_class(&factory, cid, name)?);
    }
    Ok(Bundle { vendor: factory.vendor(), classes })
}

fn describe_class(factory: &module::Factory, cid: TUID, name: String) -> Result<ClassInfo, String> {
    let context = ComWrapper::new(host::Host).to_com_ptr::<FUnknown>().expect("a host is an FUnknown");
    let component: ComPtr<IComponent> = factory.create(&cid)?;
    unsafe {
        ok(component.initialize(context.as_ptr()), "initialize")?;
        // One object or two: a controller of its own is created and initialized alongside.
        let own: Option<ComPtr<IEditController>> = component.cast();
        let separate = match &own {
            Some(_) => None,
            None => {
                let mut ccid: TUID = [0; 16];
                (component.getControllerClassId(&mut ccid) == kResultOk)
                    .then(|| factory.create::<IEditController>(&ccid))
                    .transpose()?
            }
        };
        if let Some(c) = &separate {
            ok(c.initialize(context.as_ptr()), "initialize the controller")?;
        }
        let audio = MediaTypes_::kAudio as MediaType;
        let buses = |dir: BusDirection| -> Vec<u16> {
            (0..component.getBusCount(audio, dir))
                .map(|i| {
                    let mut info: BusInfo = std::mem::zeroed();
                    component.getBusInfo(audio, dir, i, &mut info);
                    info.channelCount.clamp(0, u16::MAX as i32) as u16
                })
                .collect()
        };
        let inputs = buses(BusDirections_::kInput as BusDirection);
        let outputs = buses(BusDirections_::kOutput as BusDirection);
        let events = component.getBusCount(MediaTypes_::kEvent as MediaType, BusDirections_::kInput as BusDirection) > 0;
        let params = own.as_ref().or(separate.as_ref()).map(|c| params_of(c)).unwrap_or_default();
        if let Some(c) = &separate {
            c.terminate();
        }
        component.terminate();
        Ok(ClassInfo { cid: hex_of(&cid), name, inputs, outputs, events, params })
    }
}

unsafe fn params_of(c: &ComPtr<IEditController>) -> Vec<ParamInfo> {
    (0..c.getParameterCount())
        .filter_map(|i| {
            let mut info: ParameterInfo = std::mem::zeroed();
            (c.getParameterInfo(i, &mut info) == kResultOk).then(|| {
                let string_at = |v: f64| {
                    let mut s: String128 = [0; 128];
                    c.getParamStringByValue(info.id, v, &mut s);
                    host::utf16(&s)
                };
                let strings = if (1..=STR_STEPS).contains(&info.stepCount) {
                    (0..=info.stepCount).map(|k| string_at(k as f64 / info.stepCount as f64)).collect()
                } else {
                    vec![string_at(info.defaultNormalizedValue)]
                };
                ParamInfo {
                    id: info.id,
                    title: host::utf16(&info.title),
                    units: host::utf16(&info.units),
                    steps: info.stepCount,
                    default: info.defaultNormalizedValue,
                    flags: info.flags,
                    strings,
                }
            })
        })
        .collect()
}

pub(super) fn ok(result: tresult, what: &str) -> Result<(), String> {
    (result == kResultOk).then_some(()).ok_or_else(|| format!("{what} answered {result}"))
}

fn hex_of(cid: &TUID) -> String {
    cid.iter().map(|b| format!("{:02x}", *b as u8)).collect()
}

fn cid_of(hex: &str) -> Option<TUID> {
    let bytes: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok()).collect::<Option<_>>()?;
    let mut cid: TUID = [0; 16];
    for (dst, b) in cid.iter_mut().zip(bytes.iter().take(16)) {
        *dst = *b as std::ffi::c_char;
    }
    (bytes.len() == 16).then_some(cid)
}

/// Every bundle in `dir`: a row per class, or one greyed row naming the bundle.
pub(crate) fn scan_dir(engine: &mut AudioEngine, dir: &Path) -> Vec<ScannedType> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut bundles: Vec<PathBuf> =
        entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "vst3")).collect();
    bundles.sort();
    bundles.iter().flat_map(|b| scan_bundle(engine, b)).collect()
}

fn scan_bundle(engine: &mut AudioEngine, bundle: &Path) -> Vec<ScannedType> {
    let stem = bundle.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let greyed = |reason: String| match camel(stem) {
        Some(type_name) => vec![ScannedType { type_name, stamp: None, outcome: Scanned::Unavailable(reason) }],
        None => Vec::new(),
    };
    let Some((scanner, _)) = &engine.vst3 else {
        return greyed("no VST3 scanner was handed to the engine".into());
    };
    let found = binary_of(bundle).and_then(|binary| {
        let stamp = stamp_of(&binary)?;
        described(scanner, bundle, &binary, stamp).map(|b| (binary, stamp, b))
    });
    match found {
        Err(reason) => greyed(reason),
        Ok((binary, stamp, found)) => found
            .classes
            .into_iter()
            .filter_map(|class| {
                let (type_name, replaced) = engine.register_plugin(&found.vendor, &binary, stamp, class)?;
                let outcome = Scanned::Registered { isolation: Isolation::Native, replaced };
                Some(ScannedType { type_name, stamp: Some(stamp), outcome })
            })
            .collect(),
    }
}

/// The scanner's answer for this binary at this stamp, from the cache or from a child.
fn described(scanner: &Path, bundle: &Path, binary: &Path, stamp: Stamp) -> Result<Bundle, String> {
    let dir = goofi_build::base_dir(&goofi_core::home::dir()).join("vst3");
    let mut hasher = std::hash::DefaultHasher::new();
    binary.hash(&mut hasher);
    let mtime = stamp.1.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let file = dir.join(format!("{:016x}-{}-{mtime}.json", hasher.finish(), stamp.0));
    if let Some(cached) = std::fs::read_to_string(&file).ok().and_then(|t| serde_json::from_str(&t).ok()) {
        return Ok(cached);
    }
    let output = std::process::Command::new(scanner)
        .arg("vst3-scan")
        .arg(bundle)
        .output()
        .map_err(|e| format!("could not run the scanner {}: {e}", scanner.display()))?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(match output.status.code() {
            Some(_) if !said.is_empty() => said,
            Some(code) => format!("the scanner exited with {code}"),
            None => "the scanner crashed".into(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let found: Bundle = serde_json::from_str(&text).map_err(|e| format!("the scanner's answer did not parse: {e}"))?;
    let _ = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file, text));
    Ok(found)
}

impl AudioEngine {
    /// One class as one type: named by the plugin, or by vendor and plugin where goofi's own
    /// library holds the name. A registration from the same binary at the same stamp is kept.
    fn register_plugin(&mut self, vendor: &str, binary: &Path, stamp: Stamp, class: ClassInfo) -> Option<(String, bool)> {
        let cid = cid_of(&class.cid)?;
        let name = camel(&class.name)?;
        let type_name = match self.classes.get(name.as_str()) {
            Some(held) if held.plugin.is_none() => format!("{}{name}", camel(vendor)?),
            _ => name,
        };
        let same = |d: &Arc<Derived>| d.binary == binary && d.stamp == stamp && d.cid == cid;
        if self.classes.get(type_name.as_str()).and_then(|c| c.plugin.as_ref()).is_some_and(same) {
            return Some((type_name, false));
        }
        let (intro, params) = introspection(vendor, &class);
        let derived = Arc::new(Derived { binary: binary.to_path_buf(), stamp, cid, inputs: class.inputs, outputs: class.outputs, voice: class.events, params });
        let manifest = goofi_node::leak_manifest(type_name.clone(), &intro, "audio");
        let plugin = derived.clone();
        let make = Arc::new(move |_| Box::new(Plugin::new(plugin.clone())) as Box<dyn AudioNode>);
        let replaced = self.classes.insert(manifest.type_name, Class { manifest, make, plugin: Some(derived) }).is_some();
        Some((type_name, replaced))
    }
}

const OMITTED: i32 = ParameterInfo_::ParameterFlags_::kIsHidden
    | ParameterInfo_::ParameterFlags_::kIsReadOnly
    | ParameterInfo_::ParameterFlags_::kIsBypass
    | ParameterInfo_::ParameterFlags_::kIsProgramChange;

/// The manifest a class derives to, and — in the same pass, so they cannot drift — how each of
/// its params reaches the plugin.
fn introspection(vendor: &str, class: &ClassInfo) -> (probe::Introspection, Vec<(ParamID, Kind)>) {
    let audio = goofi_core::SlotType::Audio.name().to_string();
    let float = |default: f64, min: f64, max: f64| probe::ParamSpec::Float { default, min, max };
    let voice = |name: &str, doc: &str, spec: probe::ParamSpec| probe::Param {
        group: "voice".into(),
        name: name.into(),
        doc: Some(doc.into()),
        expression: None,
        spec,
    };
    let mut params = if class.events {
        vec![
            voice("gate", "A note on at the rise, off at the fall — per channel, so a polyphonic gate is that many voices.", probe::ParamSpec::Bool { default: false }),
            voice("pitch", "Volts per octave, zero at C4.", float(0.0, -5.0, 5.0)),
            voice("velocity", "The note's velocity, 0 to 1.", float(1.0, 0.0, 1.0)),
        ]
    } else {
        Vec::new()
    };
    let mut names: Vec<String> = Vec::new();
    let mut kinds = Vec::new();
    for p in class.params.iter().filter(|p| p.flags & OMITTED == 0) {
        let name = unique(lower_camel(&p.title).unwrap_or_else(|| "param".into()), &mut names);
        let (spec, kind, doc) = if p.steps <= 0 {
            let shown = format!("{} {}", p.strings.first().map(String::as_str).unwrap_or_default(), p.units);
            (float(p.default.clamp(0.0, 1.0), 0.0, 1.0), Kind::Float, format!("{}, normalized; {} by default.", p.title, shown.trim()))
        } else if p.steps <= STR_STEPS {
            let options = distinct(&p.strings, p.steps as usize + 1);
            let at = ((p.default * p.steps as f64).round() as usize).min(options.len() - 1);
            let default = options[at].clone();
            (probe::ParamSpec::Str { default, options, refresh: false }, Kind::Stepped(p.steps as f64), p.title.clone())
        } else {
            let default = (p.default * p.steps as f64).round() as i64;
            (probe::ParamSpec::Int { default, min: 0, max: p.steps as i64 }, Kind::Stepped(p.steps as f64), p.title.clone())
        };
        params.push(probe::Param { group: "plugin".into(), name, doc: Some(doc), expression: None, spec });
        kinds.push((p.id, kind));
    }
    let intro = probe::Introspection {
        gil_safe: true,
        doc: format!("{} by {vendor}", class.name),
        category: Some(vendor.to_string()),
        producer: false,
        inputs: (0..class.inputs.len())
            .map(|i| probe::Slot { name: numbered("input", i), kind: audio.clone(), trigger: false, multi: true, required: false })
            .collect(),
        outputs: (0..class.outputs.len()).map(|i| probe::OutSlot { name: numbered("out", i), kind: audio.clone() }).collect(),
        params,
    };
    (intro, kinds)
}

/// The alnum words of `s`, capitalized and joined: a legal name, or none.
fn camel(s: &str) -> Option<String> {
    let name: String = s
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w[..1].to_ascii_uppercase() + &w[1..])
        .collect();
    goofi_core::globals::is_valid_name(&name).then_some(name)
}

fn lower_camel(s: &str) -> Option<String> {
    camel(s).map(|n| n[..1].to_ascii_lowercase() + &n[1..]).filter(|n| goofi_core::globals::is_valid_name(n))
}

fn numbered(base: &str, i: usize) -> String {
    match i {
        0 => base.to_string(),
        _ => format!("{base}{}", i + 1),
    }
}

fn unique(name: String, used: &mut Vec<String>) -> String {
    let mut candidate = name.clone();
    let mut n = 1;
    while used.contains(&candidate) {
        n += 1;
        candidate = format!("{name}{n}");
    }
    used.push(candidate.clone());
    candidate
}

/// Exactly `count` options, each distinct and none empty: a `Str` param's scalar is its index.
fn distinct(strings: &[String], count: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(count);
    for k in 0..count {
        let s = strings.get(k).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| k.to_string());
        out.push(if out.contains(&s) { format!("{s} ({k})") } else { s });
    }
    out
}
