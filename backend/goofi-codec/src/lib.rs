//! The GOOF v2 binary frame codec: the browser data plane, and the subprocess boundary.
//!
//! Frame: `magic "GOOF" | u8 version | u8 dtype tag | u32 meta_len | u32 body_len | meta | body`,
//! little-endian, with the meta dict projected from the typed `Meta` plus derived shape/dtype.

pub mod liveness;

use goofi_core::{ArrayStore, Coord, Data, MetaValue, Value};
use rmpv::Value as Mp;

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

/// `[u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]`.
fn encode_array_body(store: &ArrayStore, out: &mut Vec<u8>) {
    let dtype_str: &[u8] = b"<f4";
    let shape = store.shape();
    out.push(shape.len() as u8);
    out.push(dtype_str.len() as u8);
    out.extend_from_slice(dtype_str);
    for &dim in shape {
        out.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    out.extend_from_slice(store.as_bytes());
}

/// Meta names the wire derives from the `Data` itself, so they are never taken from `Meta`.
const DERIVED_KEYS: [&str; 2] = ["shape", "dtype"];

/// Serialize a `Data`'s `Meta` to the msgpack map used in a GOOF frame.
fn pack_meta(d: &Data) -> Vec<u8> {
    let meta = d.meta();
    let mut entries: Vec<(Mp, Mp)> = Vec::new();

    // `channels` and the derived names are projected below; dropping them here is what keeps
    // the map from carrying one key twice.
    for (k, v) in meta.iter() {
        if k == goofi_core::META_CHANNELS || DERIVED_KEYS.contains(&k.as_str()) || matches!(v, MetaValue::Null) {
            continue;
        }
        entries.push((Mp::from(k.as_str()), mv_to_mp(v)));
    }

    match d.value() {
        Value::Array(store) => {
            let shape: Vec<Mp> = store.shape().iter().map(|&d| Mp::from(d as u64)).collect();
            entries.push((Mp::from("shape"), Mp::Array(shape)));
            entries.push((Mp::from("dtype"), Mp::from("float32")));
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
    // Python-compat: positional axes cross as a dim-keyed dict, and axis names have no wire slot.
    let entries = ch
        .dims()
        .map(|(dim, coords)| (Mp::from(dim), Mp::Array(coords.iter().map(coord_to_mp).collect())))
        .collect();
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
        // `channels` is projected via channels_to_mp, never through here.
        MetaValue::Axes(_) => Mp::Nil,
    }
}

/// Split a frame into `(dtype_tag, meta_bytes, body_bytes)`, validating the header.
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

/// A forward-only reader over a body slice: every read is bounds-checked, so a truncated or
/// hostile frame yields `Err` rather than a panic.
struct Cursor<'a> {
    body: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn new(body: &'a [u8]) -> Cursor<'a> {
        Cursor { body, off: 0 }
    }
    /// The next `n` bytes, advancing past them; `what` names them in the error.
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
    fn rest(self) -> &'a [u8] {
        &self.body[self.off..]
    }
}

/// Decode an array body: the ingest boundary, where a foreign source dtype is cast to f32.
fn decode_array_body(body: &[u8], meta: goofi_core::Meta) -> std::result::Result<Data, String> {
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
    // The shape×4 overflow guard lives in `array_f32`, deliberately.
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

/// Parse the msgpack meta map written by [`pack_meta`] back into a typed `Meta`.
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
            other => meta.set(other, mp_to_mv(&val)),
        }
    }
    Ok(meta)
}

fn parse_channels(v: &Mp) -> goofi_core::Axes {
    // Entries may arrive out of order, so `with` pads up to the max labeled dim.
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

/// `group -> name -> Param`, spelled out because the codec has no goofi-node dep.
pub type ParamMap = indexmap::IndexMap<String, indexmap::IndexMap<String, goofi_core::Param>>;

/// The slot entries of a run request: `(slot, source, frame)`, the source empty on a single slot.
pub type SourcedSlots = Vec<(String, String, Data)>;

/// Append a named-slot list: `[u16 n]` then n × `[u16 name_len][name][u16 src_len][src]
/// [u32 frame_len][GOOF frame]`; `src` is the `node.slot` a multi-slot frame came from, else empty.
pub fn encode_slots(slots: &[(&str, &str, &Data)], out: &mut Vec<u8>) {
    out.extend_from_slice(&(slots.len() as u16).to_le_bytes());
    for (name, source, d) in slots {
        for text in [name, source] {
            let tb = text.as_bytes();
            out.extend_from_slice(&(tb.len() as u16).to_le_bytes());
            out.extend_from_slice(tb);
        }
        let frame = encode(d);
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        out.extend_from_slice(&frame);
    }
}

/// Decode the named-slot list written by [`encode_slots`].
pub fn decode_slots(body: &[u8]) -> std::result::Result<SourcedSlots, String> {
    let mut cur = Cursor::new(body);
    let n = cur.u16("slot count")?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let nlen = cur.u16("slot name length")?;
        let name = std::str::from_utf8(cur.take(nlen, "slot name")?).map_err(|e| e.to_string())?.to_string();
        let slen = cur.u16("slot source length")?;
        let source = std::str::from_utf8(cur.take(slen, "slot source")?).map_err(|e| e.to_string())?.to_string();
        let flen = cur.u32("slot frame length")?;
        let data = decode(cur.take(flen, "slot frame")?)?;
        out.push((name, source, data));
    }
    Ok(out)
}

/// A decoded subprocess request, always carrying the node's live params: one tick, the ⟳ on
/// one string param, or a pulse on one pulse param.
pub enum Request {
    Process { params: ParamMap, slots: SourcedSlots },
    Refresh { params: ParamMap, group: String, name: String },
    Pulse { params: ParamMap, group: String, name: String },
}

fn encode_params(params: &ParamMap, out: &mut Vec<u8>) {
    let pbytes = rmp_serde::to_vec(params).expect("param serialize (Param derives Serialize)");
    out.extend_from_slice(&(pbytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&pbytes);
}

/// Encode a tick request: `[0][u32 params_len][params msgpack][slots]`, each slot with its source.
pub fn encode_request(params: &ParamMap, slots: &[(&str, &str, &Data)]) -> Vec<u8> {
    let mut out = vec![0u8];
    encode_params(params, &mut out);
    encode_slots(slots, &mut out);
    out
}

/// Encode a refresh request: `[1][u32 params_len][params msgpack][(group, name) msgpack]`.
pub fn encode_refresh_request(params: &ParamMap, group: &str, name: &str) -> Vec<u8> {
    encode_keyed_request(1, params, group, name)
}

/// Encode a pulse request: `[2][u32 params_len][params msgpack][(group, name) msgpack]`.
pub fn encode_pulse_request(params: &ParamMap, group: &str, name: &str) -> Vec<u8> {
    encode_keyed_request(2, params, group, name)
}

fn encode_keyed_request(tag: u8, params: &ParamMap, group: &str, name: &str) -> Vec<u8> {
    let mut out = vec![tag];
    encode_params(params, &mut out);
    out.extend_from_slice(&rmp_serde::to_vec(&(group, name)).expect("two strings"));
    out
}

/// Decode a request frame written by [`encode_request`], [`encode_refresh_request`] or
/// [`encode_pulse_request`].
pub fn decode_request(buf: &[u8]) -> std::result::Result<Request, String> {
    let (&tag, rest) = buf.split_first().ok_or("empty request frame")?;
    let mut cur = Cursor::new(rest);
    let plen = cur.u32("params length")?;
    let pbytes = cur.take(plen, "params blob")?;
    let params: ParamMap = rmp_serde::from_slice(pbytes).map_err(|e| e.to_string())?;
    match tag {
        0 => Ok(Request::Process { params, slots: decode_slots(cur.rest())? }),
        1 | 2 => {
            let (group, name): (String, String) =
                rmp_serde::from_slice(cur.rest()).map_err(|e| e.to_string())?;
            Ok(if tag == 1 { Request::Refresh { params, group, name } } else { Request::Pulse { params, group, name } })
        }
        other => Err(format!("unknown request tag {other}")),
    }
}

/// A decoded subprocess response: the node's output slots, a per-tick node error message, or
/// a refreshed option list — `None` when the node had no answer and the param keeps its own.
pub enum Response {
    Slots(Vec<(String, Data)>),
    NodeError(String),
    Options(Option<Vec<String>>),
}

/// Encode an OK response: `[0][slots]`; an output has no source, so each crosses with none.
pub fn encode_response(slots: &[(&str, &Data)]) -> Vec<u8> {
    let mut out = vec![0u8];
    let unsourced: Vec<(&str, &str, &Data)> = slots.iter().map(|(name, d)| (*name, "", *d)).collect();
    encode_slots(&unsourced, &mut out);
    out
}

/// Encode a node-error response: `[1][utf8 message]`.
pub fn encode_error_response(msg: &str) -> Vec<u8> {
    let mut out = vec![1u8];
    out.extend_from_slice(msg.as_bytes());
    out
}

/// Encode a refresh response: `[2][Option<Vec<String>> msgpack]`.
pub fn encode_options_response(options: &Option<Vec<String>>) -> Vec<u8> {
    let mut out = vec![2u8];
    out.extend_from_slice(&rmp_serde::to_vec(options).expect("strings"));
    out
}

/// Decode a response frame; the outer `Err` is a malformed frame, never a node-reported one.
pub fn decode_response(buf: &[u8]) -> std::result::Result<Response, String> {
    let (&tag, rest) = buf.split_first().ok_or("empty response frame")?;
    match tag {
        0 => Ok(Response::Slots(decode_slots(rest)?.into_iter().map(|(name, _, d)| (name, d)).collect())),
        1 => Ok(Response::NodeError(String::from_utf8_lossy(rest).into_owned())),
        2 => Ok(Response::Options(rmp_serde::from_slice(rest).map_err(|e| e.to_string())?)),
        other => Err(format!("unknown response tag {other}")),
    }
}
