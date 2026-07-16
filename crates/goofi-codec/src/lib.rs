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

use goofi_core::{ArrayStore, Coord, Data, MetaValue, Value};
use rmpv::Value as Mp;

const MAGIC: &[u8; 4] = b"GOOF";
const VERSION: u8 = 2;
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
        Value::Array(store) => write_array_body(store, out),
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

fn write_array_body(store: &ArrayStore, out: &mut Vec<u8>) {
    let dtype_str = store.dtype().numpy_typestr().as_bytes();
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

fn pack_meta(d: &Data) -> Vec<u8> {
    let meta = d.meta();
    let mut entries: Vec<(Mp, Mp)> = Vec::new();

    // Arbitrary keys first (sorted, from BTreeMap iteration).
    for (k, v) in meta.extra.iter() {
        entries.push((Mp::from(k.as_str()), mv_to_mp(v)));
    }

    match d.value() {
        Value::Array(store) => {
            let shape: Vec<Mp> = store.shape().iter().map(|&d| Mp::from(d as u64)).collect();
            entries.push((Mp::from("shape"), Mp::Array(shape)));
            entries.push((Mp::from("dtype"), Mp::from(store.dtype().numpy_name())));
            entries.push((Mp::from("channels"), channels_to_mp(&meta.channels)));
        }
        Value::Str(_) => {
            entries.push((Mp::from("dtype"), Mp::from("str")));
        }
        Value::Table(_) => {
            entries.push((Mp::from("dtype"), Mp::from("table")));
        }
    }

    if let Some(sf) = meta.sfreq {
        entries.push((Mp::from("sfreq"), Mp::from(sf)));
    }
    if let Some(ix) = meta.index {
        entries.push((Mp::from("index"), Mp::from(ix)));
    }
    if let Some(r) = &meta.reduced {
        entries.push((Mp::from("reduced"), mv_to_mp(r)));
    }

    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &Mp::Map(entries)).expect("msgpack meta encode");
    buf
}

fn channels_to_mp(ch: &goofi_core::Channels) -> Mp {
    let mut entries: Vec<(Mp, Mp)> = Vec::new();
    for (&dim, coords) in ch.0.iter() {
        let list: Vec<Mp> = coords.iter().map(coord_to_mp).collect();
        entries.push((Mp::from(format!("dim{dim}")), Mp::Array(list)));
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
