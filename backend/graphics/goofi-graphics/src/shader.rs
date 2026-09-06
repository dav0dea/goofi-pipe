//! A `.wgsl` node file: the header block that IS its manifest, the prelude the engine appends,
//! and naga's verdict on the two together.

use std::sync::atomic::{AtomicU64, Ordering};

use goofi_core::probe::Introspection;
use goofi_core::SlotType;
use goofi_node::{NodeManifest, ParamDecl, ParamSpec};

/// The names the prelude declares. A header that takes one is refused, rather than shadowing it.
const RESERVED: &[&str] = &["time", "resolution", "samp", "p", "Params", "Vs", "vs", "fs", "shade"];

/// The one output every graphics node has.
const OUT: &str = "out";

/// The header's manifest, with the one output added. The file's WHOLE text stays the source that
/// naga reads, so a line number it reports is the line an author sees.
pub fn header(source: &str) -> Result<Introspection, String> {
    let start = source.find("/*").ok_or("no header: a node file opens with `/* goofi { … } */`")?;
    let end = source[start..].find("*/").map(|i| start + i).ok_or("the header block is not closed")?;
    let inside = source[start + 2..end].trim_start();
    let json = inside.strip_prefix("goofi").ok_or("the header block does not open with `goofi`")?;
    let mut intro = goofi_node::parse_introspection(json.trim()).map_err(|e| format!("header: {e}"))?;
    if !intro.outputs.is_empty() {
        return Err("the header lists outputs; a graphics node has the one output `out`".into());
    }
    intro.outputs.push(goofi_core::probe::OutSlot { name: OUT.to_string(), kind: SlotType::Texture.name().to_string() });
    for s in &intro.inputs {
        match SlotType::from_name(&s.kind) {
            Some(SlotType::Texture | SlotType::Array) => {}
            _ => return Err(format!("input `{}` is `{}`; a graphics input is TEXTURE or ARRAY", s.name, s.kind)),
        }
    }
    let taken = intro
        .inputs
        .iter()
        .map(|s| s.name.as_str())
        .chain(intro.params.iter().map(|p| p.name.as_str()))
        .find(|n| RESERVED.contains(n));
    if let Some(name) = taken {
        return Err(format!("`{name}` is the prelude's; choose another name"));
    }
    Ok(intro)
}

fn wgsl_type(spec: &ParamSpec) -> &'static str {
    match spec {
        ParamSpec::Float { .. } => "f32",
        ParamSpec::Int { .. } => "i32",
        ParamSpec::Bool { .. } | ParamSpec::Str { .. } | ParamSpec::Pulse => "u32",
    }
}

/// What the engine appends after the file: the bindings a body reads, and the stages that call it.
/// `uv` is (0, 0) at the TOP-left, WGSL's own texture space.
pub fn prelude(manifest: &NodeManifest) -> String {
    let mut s = String::from(
        "\n@group(0) @binding(0) var<uniform> time: f32;\n\
         @group(0) @binding(1) var<uniform> resolution: vec2f;\n\
         @group(0) @binding(2) var samp: sampler;\n",
    );
    if !manifest.params.is_empty() {
        s.push_str("struct Params {\n");
        for d in manifest.params {
            s.push_str(&format!("    {}: {},\n", d.name, wgsl_type(&d.spec)));
        }
        s.push_str("}\n@group(0) @binding(3) var<uniform> p: Params;\n");
    }
    for (i, input) in manifest.inputs.iter().enumerate() {
        s.push_str(&format!("@group(1) @binding({i}) var {}: texture_2d<f32>;\n", input.name));
    }
    s.push_str(
        "struct Vs { @builtin(position) pos: vec4f, @location(0) uv: vec2f }\n\
         @vertex fn vs(@builtin(vertex_index) i: u32) -> Vs {\n\
         \x20   let x = f32(i32(i & 1u) * 4 - 1);\n\
         \x20   let y = f32(i32(i & 2u) * 2 - 1);\n\
         \x20   var o: Vs;\n\
         \x20   o.pos = vec4f(x, y, 0.0, 1.0);\n\
         \x20   o.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);\n\
         \x20   return o;\n\
         }\n\
         @fragment fn fs(v: Vs) -> @location(0) vec4f { return shade(v.uv); }\n",
    );
    s
}

/// naga's verdict on the file plus its prelude, with the line and column in it.
pub fn validate(full: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(full).map_err(|e| e.emit_to_string(full))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .map(|_| ())
        .map_err(|e| e.emit_to_string(full))
}

/// One stage's `Params` buffer: a 4-byte scalar per declared param, padded to 16. Every field is
/// a scalar, so the layout needs no layouter.
pub fn uniform_bytes(decls: &[ParamDecl], atomics: &[AtomicU64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(decls.len() * 4 + 16);
    for (d, a) in decls.iter().zip(atomics) {
        let v = f64::from_bits(a.load(Ordering::Relaxed));
        match d.spec {
            ParamSpec::Float { .. } => out.extend_from_slice(&(v as f32).to_le_bytes()),
            ParamSpec::Int { .. } => out.extend_from_slice(&(v.round() as i32).to_le_bytes()),
            _ => out.extend_from_slice(&(v.round().max(0.0) as u32).to_le_bytes()),
        }
    }
    out.resize(out.len().next_multiple_of(16), 0);
    out
}
