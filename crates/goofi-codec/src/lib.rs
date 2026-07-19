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

/// The array-body layout `[u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]`,
/// shared by the GOOF frame body and the subprocess transport (which reuses it verbatim, with a
/// typed sfreq/index prefix instead of msgpack meta). Inverse: [`decode_array_body`].
pub fn encode_array_body(store: &ArrayStore, out: &mut Vec<u8>) {
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
    if let Some(uf) = meta.ufreq {
        entries.push((Mp::from("ufreq"), Mp::from(uf)));
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
/// Shared by the GOOF frame decoder and the subprocess transport.
pub fn decode_array_body(body: &[u8], meta: goofi_core::Meta) -> std::result::Result<Data, String> {
    // [u8 ndim][u8 dtype_str_len][dtype_str][ndim × u32 shape][raw bytes]
    let mut cur = Cursor::new(body);
    let ndim = cur.u8("array ndim")?;
    let dslen = cur.u8("array dtype len")?;
    let dstr = std::str::from_utf8(cur.take(dslen, "array dtype string")?).map_err(|e| e.to_string())?;
    let dtype = goofi_core::DType::from_numpy_typestr(dstr)
        .ok_or_else(|| format!("unsupported dtype `{dstr}`"))?;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(cur.u32("array shape")?);
    }
    // The shape×itemsize overflow guard lives in from_array_bytes — kept there deliberately.
    Data::from_array_bytes(dtype, shape, cur.rest().to_vec(), meta).map_err(|e| e.to_string())
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
            "channels" => meta.channels = parse_channels(&val),
            "sfreq" => meta.sfreq = val.as_f64(),
            "ufreq" => meta.ufreq = val.as_f64(),
            "index" => meta.index = val.as_u64(),
            "reduced" => meta.reduced = Some(mp_to_mv(&val)),
            other => {
                meta.extra.insert(other.to_string(), mp_to_mv(&val));
            }
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

#[cfg(test)]
mod decode_tests {
    use super::*;
    use goofi_core::{Axes, Axis, Coord, DType, Data, Meta, MetaValue, Value};

    fn arr_bytes(d: &Data) -> (DType, Vec<usize>, Vec<u8>) {
        match d.value() {
            Value::Array(s) => (s.dtype(), s.shape().to_vec(), s.as_bytes().to_vec()),
            _ => panic!("not array"),
        }
    }

    #[test]
    fn array_roundtrips_with_meta() {
        let mut meta = Meta::empty();
        meta.sfreq = Some(256.0);
        meta.ufreq = Some(128.0);
        meta.index = Some(42);
        meta.extra.insert("label".into(), MetaValue::Str("eeg".into()));
        meta.channels = Axes::new().with(0, Axis::coords(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())]));
        let buf: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![2], buf, meta).unwrap();

        let back = decode(&encode(&d)).expect("decode");
        let (dt, sh, by) = arr_bytes(&back);
        assert_eq!(dt, DType::F32);
        assert_eq!(sh, vec![2]);
        assert_eq!(arr_bytes(&d).2, by);
        assert_eq!(back.meta().sfreq, Some(256.0));
        assert_eq!(back.meta().ufreq, Some(128.0));
        assert_eq!(back.meta().index, Some(42));
        assert_eq!(back.meta().extra.get("label"), Some(&MetaValue::Str("eeg".into())));
        assert!(!back.meta().extra.contains_key("ufreq"), "ufreq is a typed field, never extra");
        let ch = back.meta().channels.get(0).and_then(|a| a.coords.clone()).expect("dim0 channels");
        assert_eq!(ch.as_ref(), &[Coord::Str("Fz".into()), Coord::Str("Cz".into())]);
    }

    #[test]
    fn ufreq_none_is_absent_from_the_wire() {
        // An unmeasured slot (ufreq == None) must not emit the key, so frames that
        // predate ufreq (and the Python-golden fixtures) stay byte-identical.
        let d = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
        assert_eq!(d.meta().ufreq, None);
        let bytes = encode(&d);
        assert!(
            !bytes.windows(5).any(|w| w == b"ufreq"),
            "None ufreq must not appear in the encoded meta"
        );
        assert_eq!(decode(&bytes).unwrap().meta().ufreq, None);

        // A measured slot emits the key.
        let mut meta = Meta::empty();
        meta.ufreq = Some(60.0);
        let d2 = Data::from_array_bytes(DType::F32, vec![1], 1.0f32.to_le_bytes().to_vec(), meta).unwrap();
        assert!(encode(&d2).windows(5).any(|w| w == b"ufreq"), "measured ufreq must appear");
    }

    #[test]
    fn string_roundtrips() {
        let d = Data::string("hello world", Meta::empty());
        let back = decode(&encode(&d)).unwrap();
        match back.value() {
            Value::Str(s) => assert_eq!(&**s, "hello world"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn table_roundtrips_nested() {
        let mut map = indexmap::IndexMap::new();
        map.insert(
            "a".to_string(),
            Data::from_array_bytes(DType::F32, vec![1], 3.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap(),
        );
        map.insert("b".to_string(), Data::string("x", Meta::empty()));
        let d = Data::table(map, Meta::empty());

        let back = decode(&encode(&d)).unwrap();
        match back.value() {
            Value::Table(m) => {
                assert_eq!(m.len(), 2);
                assert!(matches!(m.get("a").unwrap().value(), Value::Array(_)));
                match m.get("b").unwrap().value() {
                    Value::Str(s) => assert_eq!(&**s, "x"),
                    _ => panic!("nested string"),
                }
            }
            _ => panic!("expected table"),
        }
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(decode(b"XXXX").is_err());
        let d = Data::string("abc", Meta::empty());
        let mut frame = encode(&d);
        frame.truncate(frame.len() - 1); // drop a body byte
        assert!(decode(&frame).is_err());
    }

    // -------------------------------------------------------------------------
    // Adversarial input: a malformed/hostile frame must yield Err, never panic,
    // wrap, or over-read. `decode` is public and (in future P2P) may see untrusted
    // bytes, so it must be robust to garbage.
    // -------------------------------------------------------------------------

    #[test]
    fn decode_never_panics_on_arbitrary_prefixes_of_a_valid_frame() {
        // Every strict prefix of a real frame is incomplete and must Err —
        // never panic or read out of bounds.
        let mut meta = Meta::empty();
        meta.sfreq = Some(64.0);
        let buf: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![2, 4], buf, meta).unwrap();
        let frame = encode(&d);
        for n in 0..frame.len() {
            // Any strict prefix is incomplete → must Err, must not panic.
            assert!(decode(&frame[..n]).is_err(), "prefix len {n} should be rejected");
        }
        assert!(decode(&frame).is_ok(), "the full frame still decodes");
    }

    #[test]
    fn decode_rejects_overflowing_shape_without_wrapping() {
        // Hand-craft an ARRAY frame whose header claims a 2-D shape of
        // [2^32-1, 2^32-1] with an empty body. Without checked arithmetic the
        // element-count product * itemsize would wrap and spuriously accept the
        // 0-byte body; it must instead error cleanly.
        let mut body = Vec::new();
        body.push(2u8); // ndim
        let dstr = b"<f4";
        body.push(dstr.len() as u8);
        body.extend_from_slice(dstr);
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        // no raw bytes (claims a gigantic array, sends nothing)

        let meta = b""; // empty msgpack meta
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.push(VERSION);
        frame.push(0); // ARRAY tag
        frame.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(meta);
        frame.extend_from_slice(&body);

        assert!(decode(&frame).is_err(), "an overflowing shape must be rejected, not wrapped");
    }

    #[test]
    fn decode_rejects_declared_lengths_exceeding_the_frame() {
        // meta_len / body_len larger than the actual bytes must be rejected by
        // split_frame rather than slicing out of bounds.
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.push(VERSION);
        frame.push(0);
        frame.extend_from_slice(&1000u32.to_le_bytes()); // meta_len way past the end
        frame.extend_from_slice(&1000u32.to_le_bytes());
        frame.extend_from_slice(b"short");
        assert!(decode(&frame).is_err());
        assert!(split_frame(&frame).is_err());
    }

    #[test]
    fn decode_rejects_bad_dtype_and_unknown_tag() {
        // Unknown dtype string.
        let mut body = vec![1u8, 3u8];
        body.extend_from_slice(b"<z9"); // not a real dtype
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&[0, 0, 0, 0]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.push(VERSION);
        frame.push(0);
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(&body);
        assert!(decode(&frame).is_err());

        // Unknown dtype tag (3).
        let mut f2 = Vec::new();
        f2.extend_from_slice(MAGIC);
        f2.push(VERSION);
        f2.push(3); // invalid tag
        f2.extend_from_slice(&0u32.to_le_bytes());
        f2.extend_from_slice(&0u32.to_le_bytes());
        assert!(decode(&f2).is_err());
    }
}
