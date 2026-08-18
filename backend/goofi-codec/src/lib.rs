//! goofi-codec — the GOOF v2 binary frame **encoder** (browser data plane) and a
//! decoder used only at the subprocess boundary.
//!
//! Frame (little-endian throughout):
//! ```text
//!   0   4  magic "GOOF"
//!   4   1  version = 2
//!   5   1  dtype tag (0=ARRAY, 1=STRING, 2=TABLE)
//!   6   4  meta_len (u32)   — msgpack meta dict
//!   10  4  body_len (u32)
//!   14  *  meta bytes (msgpack)
//!   *   *  body bytes
//! ```
//! ARRAY body: `[u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]`.
//! STRING body: utf-8. TABLE body: `[u32 n]` then `[u16 key_len][key][u32 value_len][frame]`*.
//!
//! `decode.ts` in the frontend is the decode-only mirror, so this crate is
//! authoritative for what the browser receives. The meta dict is *projected* here
//! from the typed `Meta` + derived shape/dtype (which are not stored in `Meta`).

pub mod liveness;

use goofi_core::{ArrayStore, Coord, Data, MetaValue, Value};
use rmpv::Value as Mp;

/// The frame's identity, mirrored in `frontend/src/lib/codec/` — public because that mirror
/// and its golden are what stop the two drifting.
pub const MAGIC: &[u8; 4] = b"GOOF";
pub const VERSION: u8 = 2;
pub const HEADER_SIZE: usize = 14;

/// Encode a `Data` into a fresh GOOF v2 frame.
pub fn encode(d: &Data) -> Vec<u8> {
    let meta_bytes = pack_meta(d);
    let mut body = Vec::new();
    write_body(d, &mut body);

    let mut out = Vec::with_capacity(HEADER_SIZE + meta_bytes.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(d.dtype_tag());
    out.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta_bytes);
    out.extend_from_slice(&body);
    out
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

fn write_body(d: &Data, out: &mut Vec<u8>) {
    match d.value() {
        Value::Array(store) => encode_array_body(store, out),
        Value::Str(s) => out.extend_from_slice(s.as_bytes()),
        Value::Table(map) => {
            out.extend_from_slice(&(map.len() as u32).to_le_bytes());
            for (key, value) in map.iter() {
                let kb = key.as_bytes();
                out.extend_from_slice(&(kb.len() as u16).to_le_bytes());
                out.extend_from_slice(kb);
                let frame = encode(value);
                out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
                out.extend_from_slice(&frame);
            }
        }
    }
}

/// The array-body layout `[u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]`
/// inside a GOOF frame body. Inverse: [`decode_array_body`]. (Module-internal: the subprocess
/// tier now sends whole GOOF frames, so it no longer reuses this layout directly.)
fn encode_array_body(store: &ArrayStore, out: &mut Vec<u8>) {
    let dtype_str: &[u8] = b"<f4"; // arrays are always f32
    let shape = store.shape();
    out.push(shape.len() as u8);
    out.push(dtype_str.len() as u8);
    out.extend_from_slice(dtype_str);
    for &dim in shape {
        out.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    out.extend_from_slice(store.as_bytes());
}

// ---------------------------------------------------------------------------
// Meta projection (typed Meta + derived shape/dtype -> msgpack map)
// ---------------------------------------------------------------------------

/// Meta names the wire *derives* from the `Data` itself, so they are never taken from `Meta`.
const DERIVED_KEYS: [&str; 2] = ["shape", "dtype"];

/// Serialize a `Data`'s `Meta` (channels/sfreq/index/extra) to the msgpack map used in a GOOF
/// frame body. Inverse: [`parse_meta`]. (Module-internal.)
fn pack_meta(d: &Data) -> Vec<u8> {
    let meta = d.meta();
    let mut entries: Vec<(Mp, Mp)> = Vec::new();

    // Every meta entry (builtins + extras), skipping unset (`Null`) builtins — so an unset
    // ufreq/index stays off the wire. `channels` is projected per value-kind below (arrays
    // only, matching the wire contract), never here — and so are the derived names, which a
    // node's own meta dict is free to carry (`dict_to_meta` accepts any string key); dropping
    // them here is what keeps the map from carrying the same key twice.
    for (k, v) in meta.iter() {
        if k == goofi_core::META_CHANNELS || DERIVED_KEYS.contains(&k.as_str()) || matches!(v, MetaValue::Null) {
            continue;
        }
        entries.push((Mp::from(k.as_str()), mv_to_mp(v)));
    }

    // Derived shape/dtype (never stored in Meta), plus channels for arrays (always, even empty).
    match d.value() {
        Value::Array(store) => {
            let shape: Vec<Mp> = store.shape().iter().map(|&d| Mp::from(d as u64)).collect();
            entries.push((Mp::from("shape"), Mp::Array(shape)));
            entries.push((Mp::from("dtype"), Mp::from("float32"))); // arrays are always f32
            entries.push((Mp::from("channels"), channels_to_mp(meta.channels())));
        }
        Value::Str(_) => {
            entries.push((Mp::from("dtype"), Mp::from("str")));
        }
        Value::Table(_) => {
            entries.push((Mp::from("dtype"), Mp::from("table")));
        }
    }

    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &Mp::Map(entries)).expect("msgpack meta encode");
    buf
}

fn channels_to_mp(ch: &goofi_core::Axes) -> Mp {
    // Positional axes -> the dim-keyed wire dict (Python-compat). A dimension without
    // coords emits nothing (byte-identical to the old map for coord-only frames).
    // Axis names have no wire slot yet — they are internal-first.
    let mut entries: Vec<(Mp, Mp)> = Vec::new();
    for (dim, axis) in ch.0.iter().enumerate() {
        if let Some(coords) = &axis.coords {
            let list: Vec<Mp> = coords.iter().map(coord_to_mp).collect();
            entries.push((Mp::from(format!("dim{dim}")), Mp::Array(list)));
        }
    }
    Mp::Map(entries)
}

fn coord_to_mp(c: &Coord) -> Mp {
    match c {
        Coord::Num(n) => Mp::from(*n),
        Coord::Str(s) => Mp::from(s.as_ref()),
    }
}

fn mv_to_mp(v: &MetaValue) -> Mp {
    match v {
        MetaValue::Null => Mp::Nil,
        MetaValue::Bool(b) => Mp::from(*b),
        MetaValue::Int(i) => Mp::from(*i),
        MetaValue::Uint(u) => Mp::from(*u),
        MetaValue::Float(f) => Mp::from(*f),
        MetaValue::Str(s) => Mp::from(s.as_str()),
        MetaValue::Bytes(b) => Mp::Binary(b.clone()),
        MetaValue::List(l) => Mp::Array(l.iter().map(mv_to_mp).collect()),
        MetaValue::Map(m) => {
            Mp::Map(m.iter().map(|(k, v)| (Mp::from(k.as_str()), mv_to_mp(v))).collect())
        }
        // `channels` (the only Axes) is projected via channels_to_mp, never through here.
        MetaValue::Axes(_) => Mp::Nil,
    }
}

/// Split a frame into `(dtype_tag, meta_bytes, body_bytes)`, validating the
/// header. Shared by tests and (later) the subprocess decoder.
pub fn split_frame(frame: &[u8]) -> std::result::Result<(u8, &[u8], &[u8]), String> {
    if frame.len() < HEADER_SIZE {
        return Err(format!("frame too small: {} bytes", frame.len()));
    }
    if &frame[0..4] != MAGIC {
        return Err(format!("bad magic {:?}", &frame[0..4]));
    }
    if frame[4] != VERSION {
        return Err(format!("bad version {}", frame[4]));
    }
    let tag = frame[5];
    let meta_len = u32::from_le_bytes(frame[6..10].try_into().unwrap()) as usize;
    let body_len = u32::from_le_bytes(frame[10..14].try_into().unwrap()) as usize;
    let meta_end = HEADER_SIZE + meta_len;
    let body_end = meta_end + body_len;
    if frame.len() < body_end {
        return Err(format!(
            "frame truncated: need {body_end}, have {}",
            frame.len()
        ));
    }
    Ok((tag, &frame[HEADER_SIZE..meta_end], &frame[meta_end..body_end]))
}

// ---------------------------------------------------------------------------
// Decode (inverse of `encode`) — used at the subprocess boundary to receive a
// frame from a Python worker. Reconstructs `Data` (value + Meta), deriving
// shape/dtype from the body (never from the redundant meta keys).
// ---------------------------------------------------------------------------

/// Decode a GOOF v2 frame into a `Data`. The inverse of [`encode`].
pub fn decode(frame: &[u8]) -> std::result::Result<Data, String> {
    let (tag, meta_bytes, body) = split_frame(frame)?;
    let meta = parse_meta(meta_bytes)?;
    match tag {
        0 => decode_array_body(body, meta),
        1 => {
            let s = std::str::from_utf8(body).map_err(|e| e.to_string())?;
            Ok(Data::string(s, meta))
        }
        2 => decode_table(body, meta),
        other => Err(format!("unknown dtype tag {other}")),
    }
}

/// A forward-only reader over a body slice. Every read bounds-checks with `checked_add` + `.get()`,
/// so a truncated or hostile frame yields `Err` — never a panic or a wrapping over-read. This is the
/// codec's must-never-panic contract in ONE place, instead of a hand-written check per field.
struct Cursor<'a> {
    body: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(body: &'a [u8]) -> Cursor<'a> {
        Cursor { body, off: 0 }
    }
    /// The next `n` bytes, advancing past them; `Err(what truncated)` if the body is too short.
    fn take(&mut self, n: usize, what: &str) -> std::result::Result<&'a [u8], String> {
        let end = self.off.checked_add(n).ok_or_else(|| format!("{what} length overflow"))?;
        let s = self.body.get(self.off..end).ok_or_else(|| format!("{what} truncated"))?;
        self.off = end;
        Ok(s)
    }
    fn u8(&mut self, what: &str) -> std::result::Result<usize, String> {
        Ok(self.take(1, what)?[0] as usize)
    }
    fn u16(&mut self, what: &str) -> std::result::Result<usize, String> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()) as usize)
    }
    fn u32(&mut self, what: &str) -> std::result::Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()) as usize)
    }
    /// Everything from the cursor to the end (consumes the cursor).
    fn rest(self) -> &'a [u8] {
        &self.body[self.off..]
    }
}

/// Decode the array-body layout written by [`encode_array_body`] into a `Data` carrying `meta`.
/// This is the ingest boundary: a foreign source dtype is cast to f32 here (the only place a dtype
/// is parsed). The subprocess tier's cast-warning now lives at the pyo3 boundary (`goofi-pymod`),
/// so the source dtype is no longer surfaced to a caller. (Module-internal.)
fn decode_array_body(body: &[u8], meta: goofi_core::Meta) -> std::result::Result<Data, String> {
    // [u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]
    let mut cur = Cursor::new(body);
    let ndim = cur.u8("array ndim")?;
    let dslen = cur.u8("array dtype len")?;
    let dstr = std::str::from_utf8(cur.take(dslen, "array dtype string")?).map_err(|e| e.to_string())?;
    let src = goofi_core::SrcDtype::from_numpy_typestr(dstr)
        .ok_or_else(|| format!("unsupported dtype `{dstr}`"))?;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(cur.u32("array shape")?);
    }
    // Cast the foreign body to f32, then construct. The shape×4 overflow guard lives in
    // array_f32 — kept there deliberately.
    let (f32_bytes, _did_cast) = goofi_core::cast_to_f32(src, cur.rest()).map_err(|e| e.to_string())?;
    Data::array_f32(shape, f32_bytes, meta).map_err(|e| e.to_string())
}

fn decode_table(body: &[u8], meta: goofi_core::Meta) -> std::result::Result<Data, String> {
    let mut cur = Cursor::new(body);
    let n = cur.u32("table count")?;
    let mut map: indexmap::IndexMap<String, Data> = indexmap::IndexMap::new();
    for _ in 0..n {
        let klen = cur.u16("table key length")?;
        let key = std::str::from_utf8(cur.take(klen, "table key")?)
            .map_err(|e| e.to_string())?
            .to_string();
        let vlen = cur.u32("table value length")?;
        let child = decode(cur.take(vlen, "table value frame")?)?;
        map.insert(key, child);
    }
    Ok(Data::table(map, meta))
}

/// Parse the msgpack meta map written by [`pack_meta`] back into a typed `Meta` (shape/dtype are
/// re-derived from the body, never these keys). (Module-internal.)
fn parse_meta(bytes: &[u8]) -> std::result::Result<goofi_core::Meta, String> {
    let mut meta = goofi_core::Meta::empty();
    if bytes.is_empty() {
        return Ok(meta);
    }
    let mut cur = bytes;
    let v = rmpv::decode::read_value(&mut cur).map_err(|e| e.to_string())?;
    let Mp::Map(entries) = v else {
        return Ok(meta);
    };
    for (k, val) in entries {
        let Some(key) = k.as_str() else { continue };
        match key {
            // shape/dtype are derived from the body — ignore the redundant keys.
            "shape" | "dtype" => {}
            "channels" => meta.set_channels(parse_channels(&val)),
            // sfreq/ufreq/index/reduced and all extras are stored generically; the typed
            // accessors coerce them on read.
            other => meta.set(other, mp_to_mv(&val)),
        }
    }
    Ok(meta)
}

fn parse_channels(v: &Mp) -> goofi_core::Axes {
    // The dim-keyed wire dict -> positional axes, padding empty entries up to the max
    // labeled dim (entries may arrive out of order).
    let mut axes = goofi_core::Axes::new();
    if let Mp::Map(entries) = v {
        for (k, list) in entries {
            let Some(dim) = k
                .as_str()
                .and_then(|s| s.strip_prefix("dim"))
                .and_then(|d| d.parse::<usize>().ok())
            else {
                continue;
            };
            if let Mp::Array(items) = list {
                let coords: Vec<Coord> = items.iter().map(mp_to_coord).collect();
                axes = axes.with(dim, goofi_core::Axis::coords(coords));
            }
        }
    }
    axes
}

fn mp_to_coord(v: &Mp) -> Coord {
    match v {
        Mp::String(s) => Coord::Str(s.as_str().unwrap_or("").into()),
        other => Coord::Num(other.as_f64().unwrap_or(0.0)),
    }
}

fn mp_to_mv(v: &Mp) -> MetaValue {
    match v {
        Mp::Nil => MetaValue::Null,
        Mp::Boolean(b) => MetaValue::Bool(*b),
        Mp::Integer(i) => {
            if let Some(s) = i.as_i64() {
                MetaValue::Int(s)
            } else if let Some(u) = i.as_u64() {
                MetaValue::Uint(u)
            } else {
                MetaValue::Null
            }
        }
        Mp::F32(f) => MetaValue::Float(*f as f64),
        Mp::F64(f) => MetaValue::Float(*f),
        Mp::String(s) => MetaValue::Str(s.as_str().unwrap_or("").to_string()),
        Mp::Binary(b) => MetaValue::Bytes(b.clone()),
        Mp::Array(a) => MetaValue::List(a.iter().map(mp_to_mv).collect()),
        Mp::Map(m) => MetaValue::Map(
            m.iter()
                .filter_map(|(k, v)| k.as_str().map(|ks| (ks.to_string(), mp_to_mv(v))))
                .collect(),
        ),
        Mp::Ext(_, _) => MetaValue::Null,
    }
}

// ---------------------------------------------------------------------------
// Subprocess multi-slot request/response frames — the shared wire between the
// parent (`goofi-python`'s `subproc::RemoteNode`) and the child (`goofi.serve` in pymod).
// A request carries the live params + the present input slots; a response the
// output slots. Each slot's `Data` is a self-describing GOOF frame ([`encode`]/
// [`decode`]), so channels/sfreq/index/dtype cross with full fidelity — no
// opaque-echo or typed-sfreq-prefix hacks (those existed only because the old
// Python child couldn't parse meta; the Rust child now shares this codec). Params
// serialize via serde (`Param` derives it), so there is no per-variant juggling.
// The transport-level `seq` (re-publish dedup) is an OUTER prefix owned by the
// caller (`one_roundtrip` / `serve`), never part of these frames.
// ---------------------------------------------------------------------------

/// `group -> name -> Param` — structurally identical to `goofi_node::ParamGroups`
/// (the codec has no goofi-node dep, so it is spelled out here). The parent passes
/// its live `p.groups()` directly.
pub type ParamMap = indexmap::IndexMap<String, indexmap::IndexMap<String, goofi_core::Param>>;

/// Append a named-slot list: `[u16 n]` then n × `[u16 name_len][name][u32 frame_len][GOOF frame]`.
pub fn encode_slots(slots: &[(&str, &Data)], out: &mut Vec<u8>) {
    out.extend_from_slice(&(slots.len() as u16).to_le_bytes());
    for (name, d) in slots {
        let nb = name.as_bytes();
        out.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        out.extend_from_slice(nb);
        let frame = encode(d);
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        out.extend_from_slice(&frame);
    }
}

/// Decode the named-slot list written by [`encode_slots`]. Bounds-safe via [`Cursor`].
pub fn decode_slots(body: &[u8]) -> std::result::Result<Vec<(String, Data)>, String> {
    let mut cur = Cursor::new(body);
    let n = cur.u16("slot count")?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let nlen = cur.u16("slot name length")?;
        let name = std::str::from_utf8(cur.take(nlen, "slot name")?).map_err(|e| e.to_string())?.to_string();
        let flen = cur.u32("slot frame length")?;
        let data = decode(cur.take(flen, "slot frame")?)?;
        out.push((name, data));
    }
    Ok(out)
}

/// Encode a request frame: `[u32 params_len][params msgpack][slots]`.
pub fn encode_request(params: &ParamMap, slots: &[(&str, &Data)]) -> Vec<u8> {
    let pbytes = rmp_serde::to_vec(params).expect("param serialize (Param derives Serialize)");
    let mut out = Vec::new();
    out.extend_from_slice(&(pbytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&pbytes);
    encode_slots(slots, &mut out);
    out
}

/// Decode a request frame written by [`encode_request`].
pub fn decode_request(buf: &[u8]) -> std::result::Result<(ParamMap, Vec<(String, Data)>), String> {
    let mut cur = Cursor::new(buf);
    let plen = cur.u32("params length")?;
    let pbytes = cur.take(plen, "params blob")?;
    let params: ParamMap = rmp_serde::from_slice(pbytes).map_err(|e| e.to_string())?;
    let slots = decode_slots(cur.rest())?;
    Ok((params, slots))
}

/// A decoded subprocess response: the node's output slots, or a per-tick node error message.
/// A `process()`/`setup()` raise is reported as [`Response::NodeError`] so the parent surfaces
/// it like the in-process tier's `Ok(Err)` — WITHOUT killing + respawning the child (which
/// would lose node state + the real exception text). A malformed frame is the outer `Err` of
/// [`decode_response`] instead.
pub enum Response {
    Slots(Vec<(String, Data)>),
    NodeError(String),
}

/// Encode an OK response: `[0][slots]`.
pub fn encode_response(slots: &[(&str, &Data)]) -> Vec<u8> {
    let mut out = vec![0u8];
    encode_slots(slots, &mut out);
    out
}

/// Encode a node-error response: `[1][utf8 message]` — a per-tick `process()`/`setup()` raise
/// the child reports instead of dying, carrying the real Python exception text.
pub fn encode_error_response(msg: &str) -> Vec<u8> {
    let mut out = vec![1u8];
    out.extend_from_slice(msg.as_bytes());
    out
}

/// Decode a response frame ([`encode_response`] / [`encode_error_response`]). The outer `Err`
/// is a malformed/hostile frame; `Response::NodeError` is a node-reported per-tick error.
pub fn decode_response(buf: &[u8]) -> std::result::Result<Response, String> {
    let (&tag, rest) = buf.split_first().ok_or("empty response frame")?;
    match tag {
        0 => Ok(Response::Slots(decode_slots(rest)?)),
        1 => Ok(Response::NodeError(String::from_utf8_lossy(rest).into_owned())),
        other => Err(format!("unknown response tag {other}")),
    }
}
