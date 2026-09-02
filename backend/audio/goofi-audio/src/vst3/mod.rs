//! A `.vst3` bundle as a source of audio node types. A child `goofi` scans it — a plugin that
//! crashes at load takes the scanner down, never the server — and its classes become manifests
//! here, each hosted behind [`AudioNode`] by `node`.
// The bindings mirror the C++ headers, so a host object's trait methods carry their names.
#![allow(non_snake_case)]

mod host;
mod module;
mod node;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_audio_sdk::{AudioNode, MAX_PORTS};
use goofi_core::probe;
use goofi_node::{Isolation, Scanned, ScannedType, Stamp};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{ComPtr, ComWrapper};

use crate::nodes::Class;
use crate::AudioEngine;
pub(crate) use node::Derived;
use node::{Kind, Plugin};

/// A stepped parameter with this many steps or fewer is a `Str` of the plugin's own strings.
const STR_STEPS: i32 = 64;

/// A CEILING on the child, as `OPEN_WAIT` is on a device: the scan runs under the graph lock, and
/// a plugin that blocks at load must not wedge every op.
const SCAN_WAIT: Duration = Duration::from_secs(20);

/// What the scanner writes: the factory's vendor and every "Audio Module Class".
#[derive(Serialize, Deserialize)]
pub(crate) struct Bundle {
    vendor: String,
    classes: Vec<ClassInfo>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ClassInfo {
    cid: [u8; 16],
    name: String,
    /// Channel counts per audio bus, main first.
    inputs: Vec<u16>,
    outputs: Vec<u16>,
    events: bool,
    params: Vec<ParamInfo>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ParamInfo {
    id: u32,
    title: String,
    units: String,
    steps: i32,
    default: f64,
    flags: i32,
    /// The plugin's own rendering of its default, for a continuous param's doc.
    shown: String,
    /// One display string per step, empty for a param that is not stepped within `STR_STEPS`.
    steps_shown: Vec<String>,
}

/// Where this platform keeps its plugins.
pub fn platform_dirs() -> Vec<PathBuf> {
    #[cfg(target_os = "linux")]
    let dirs = [std::env::home_dir().map(|h| h.join(".vst3")), Some("/usr/lib/vst3".into()), Some("/usr/local/lib/vst3".into())];
    #[cfg(target_os = "macos")]
    let dirs = [std::env::home_dir().map(|h| h.join("Library/Audio/Plug-Ins/VST3")), Some("/Library/Audio/Plug-Ins/VST3".into())];
    #[cfg(windows)]
    let dirs = [
        std::env::var_os("COMMONPROGRAMFILES").map(|p| PathBuf::from(p).join("VST3")),
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Programs").join("Common").join("VST3")),
    ];
    dirs.into_iter().flatten().collect()
}

/// `goofi vst3-scan <bundle> <answer>`: the child half, which writes the bundle's classes to
/// `answer` as JSON — never to stdout, which the plugin it just loaded also owns.
pub fn scan_main(args: &[String]) -> i32 {
    let (Some(bundle), Some(answer)) = (args.first(), args.get(1)) else {
        eprintln!("usage: goofi vst3-scan <bundle> <answer>");
        return 2;
    };
    match describe(Path::new(bundle)).and_then(|found| {
        serde_json::to_vec(&found).map_err(|e| e.to_string()).and_then(|b| std::fs::write(answer, b).map_err(|e| e.to_string()))
    }) {
        Ok(()) => 0,
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
        // The bundle layout spells this one `arm64`, where Rust spells it `aarch64`.
        let arch = if arch == "aarch64" { "arm64" } else { arch };
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
    let factory = module::factory(&binary_of(bundle)?)?;
    // One class the host cannot describe never costs the bundle its others.
    let classes = factory.audio_classes().into_iter().filter_map(|(cid, name)| describe_class(&factory, cid, name).ok()).collect();
    Ok(Bundle { vendor: factory.vendor(), classes })
}

fn describe_class(factory: &module::Factory, cid: TUID, name: String) -> Result<ClassInfo, String> {
    let context = ComWrapper::new(host::Host).to_com_ptr::<FUnknown>().expect("a host is an FUnknown");
    let component: ComPtr<IComponent> = factory.create(&cid)?;
    unsafe {
        ok(component.initialize(context.as_ptr()), "initialize")?;
        let described = describe_initialized(factory, &component, &context, cid, name);
        component.terminate();
        described
    }
}

/// The body between `initialize` and `terminate`, so no `?` can leave a component initialized.
unsafe fn describe_initialized(
    factory: &module::Factory,
    component: &ComPtr<IComponent>,
    context: &ComPtr<FUnknown>,
    cid: TUID,
    name: String,
) -> Result<ClassInfo, String> {
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
    Ok(ClassInfo { cid: cid.map(|b| b as u8), name, inputs, outputs, events, params })
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
                let steps_shown = match (1..=STR_STEPS).contains(&info.stepCount) {
                    true => (0..=info.stepCount).map(|k| string_at(k as f64 / info.stepCount as f64)).collect(),
                    false => Vec::new(),
                };
                ParamInfo {
                    id: info.id,
                    title: host::utf16(&info.title),
                    units: host::utf16(&info.units),
                    steps: info.stepCount,
                    default: info.defaultNormalizedValue,
                    flags: info.flags,
                    shown: string_at(info.defaultNormalizedValue),
                    steps_shown,
                }
            })
        })
        .collect()
}

pub(super) fn ok(result: tresult, what: &str) -> Result<(), String> {
    (result == kResultOk).then_some(()).ok_or_else(|| format!("{what} answered {result}"))
}

/// Every bundle under `dir`, at any depth — an installer may group them by vendor — as a row per
/// class, or one greyed row naming the bundle.
pub(crate) fn scan_dir(engine: &mut AudioEngine, dir: &Path) -> Vec<ScannedType> {
    bundles_under(dir).iter().flat_map(|b| scan_bundle(engine, b)).collect()
}

fn bundles_under(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    let (mut bundles, folders): (Vec<PathBuf>, Vec<PathBuf>) =
        paths.into_iter().partition(|p| p.extension().is_some_and(|e| e == "vst3"));
    for folder in folders.iter().filter(|p| p.is_dir()) {
        bundles.extend(bundles_under(folder));
    }
    bundles
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
        described(scanner, bundle, &binary).map(|b| (binary, stamp, b))
    });
    match found {
        Err(reason) => greyed(reason),
        Ok((_, _, found)) if found.classes.is_empty() => greyed("no audio class in the bundle".into()),
        Ok((binary, stamp, found)) => {
            let vendor = found.vendor;
            found.classes.into_iter().map(|class| engine.register_plugin(&vendor, &binary, stamp, class)).collect()
        }
    }
}

/// The scanner's answer for this binary, from the cache or from a child. Keyed by the binary's own
/// BYTES, as an authored node's artifact is: a patch mount is a fresh directory every boot, and a
/// load restores no mtimes, so nothing about a bundle's path or stamp survives an archive.
fn described(scanner: &Path, bundle: &Path, binary: &Path) -> Result<Bundle, String> {
    let bytes = std::fs::read(binary).map_err(|e| format!("{}: {e}", binary.display()))?;
    let dir = goofi_build::base_dir(&goofi_core::home::dir()).join("vst3");
    let key = format!("{:x}", Sha256::digest(&bytes));
    let file = dir.join(format!("{key}.json"));
    if let Some(cached) = std::fs::read(&file).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
        return Ok(cached);
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // Written beside and renamed in, so two goofis scanning one bundle cannot tear the cache.
    let part = dir.join(format!("{key}.{}.part", std::process::id()));
    let said = run_scanner(scanner, bundle, &part);
    let answer = said.and_then(|()| std::fs::read(&part).map_err(|e| format!("the scanner wrote nothing: {e}")));
    let found = answer.and_then(|b| serde_json::from_slice(&b).map_err(|e| format!("the scanner's answer did not parse: {e}")));
    match found {
        Ok(found) => {
            let _ = std::fs::rename(&part, &file);
            Ok(found)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

/// One child, under a ceiling. Its own words on failure, from the file its errors go to — never a
/// pipe, which a plugin's chatter could fill while nobody is reading it.
fn run_scanner(scanner: &Path, bundle: &Path, part: &Path) -> Result<(), String> {
    let errors = part.with_extension("err");
    let sink = std::fs::File::create(&errors).map_err(|e| format!("{}: {e}", errors.display()))?;
    let mut child = std::process::Command::new(scanner)
        .arg("vst3-scan")
        .arg(bundle)
        .arg(part)
        .stdin(std::process::Stdio::null())
        .stdout(sink.try_clone().map_err(|e| e.to_string())?)
        .stderr(sink)
        .spawn()
        .map_err(|e| format!("could not run the scanner {}: {e}", scanner.display()))?;
    let deadline = Instant::now() + SCAN_WAIT;
    let status = loop {
        match child.try_wait() {
            Err(e) => break Err(format!("the scanner could not be waited for: {e}")),
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("the scanner did not answer in {}s", SCAN_WAIT.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let said = std::fs::read_to_string(&errors).unwrap_or_default().trim().to_string();
    let _ = std::fs::remove_file(&errors);
    match status? {
        s if s.success() => Ok(()),
        _ if !said.is_empty() => Err(said),
        s => Err(match s.code() {
            Some(code) => format!("the scanner exited with {code}"),
            None => "the scanner crashed".into(),
        }),
    }
}

impl AudioEngine {
    /// One class as one type: named by the plugin, or by vendor and plugin where the name is
    /// taken. A registration from the same binary at the same stamp is kept.
    fn register_plugin(&mut self, vendor: &str, binary: &Path, stamp: Stamp, class: ClassInfo) -> ScannedType {
        let Some(bare) = camel(&class.name) else {
            let reason = format!("`{}` is not a legal name: {}", class.name, goofi_core::globals::NAME_RULE);
            return ScannedType { type_name: class.name, stamp: None, outcome: Scanned::Unavailable(reason) };
        };
        let cid: TUID = class.cid.map(|b| b as std::ffi::c_char);
        let same = |d: &Arc<Derived>| d.binary == binary && d.stamp == stamp && d.cid == cid;
        // The vendor prefixes only where SOMEONE ELSE holds the name — a goofi node, or another
        // plugin. A rescan finds this very class there and keeps the name it already had.
        let mine = |name: &str| self.classes.get(name).is_some_and(|c| c.plugin.as_ref().is_some_and(&same));
        let type_name = match self.classes.contains_key(bare.as_str()) && !mine(&bare) {
            true => camel(vendor).map_or(bare.clone(), |v| format!("{v}{bare}")),
            false => bare,
        };
        if mine(&type_name) {
            let outcome = Scanned::Registered { isolation: Isolation::Native, replaced: false };
            return ScannedType { type_name, stamp: Some(stamp), outcome };
        }
        let (intro, params) = introspection(vendor, &class);
        // The refusal is HERE, where the palette can carry it: an insert-time one would offer a
        // type that every `node add` then answers with the same words, for ever.
        let widest = intro.params.len().max(intro.inputs.len()).max(intro.outputs.len());
        if widest > MAX_PORTS {
            let reason = format!("declares {widest} ports and params, and {MAX_PORTS} is the ceiling");
            return ScannedType { type_name, stamp: Some(stamp), outcome: Scanned::Unavailable(reason) };
        }
        let derived = Arc::new(Derived { binary: binary.to_path_buf(), stamp, cid, inputs: class.inputs, outputs: class.outputs, params });
        let manifest = goofi_node::leak_manifest(type_name.clone(), &intro, "audio");
        let plugin = derived.clone();
        let make = Arc::new(move |_| Box::new(Plugin::new(plugin.clone())) as Box<dyn AudioNode>);
        let replaced = self.classes.insert(manifest.type_name, Class { manifest, make, plugin: Some(derived) }).is_some();
        ScannedType { type_name, stamp: Some(stamp), outcome: Scanned::Registered { isolation: Isolation::Native, replaced } }
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
            let shown = format!("{} {}", p.shown, p.units);
            (float(p.default.clamp(0.0, 1.0), 0.0, 1.0), Kind::Float, format!("{}, normalized; {} by default.", p.title, shown.trim()))
        } else if p.steps <= STR_STEPS {
            let options = distinct(&p.steps_shown, p.steps as usize + 1);
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

/// Exactly `count` options, each distinct and none empty: a `Str` param's scalar is the index of
/// the FIRST option that matches, so a repeat would read back as the wrong step.
fn distinct(shown: &[String], count: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(count);
    for k in 0..count {
        let s = shown.get(k).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| k.to_string());
        let mut candidate = s.clone();
        let mut n = 1;
        while out.contains(&candidate) {
            n += 1;
            candidate = format!("{s} ({n})");
        }
        out.push(candidate);
    }
    out
}
