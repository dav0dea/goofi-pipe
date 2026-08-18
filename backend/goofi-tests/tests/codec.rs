//! The GOOF wire format and the subprocess frames — mirrored in `frontend/src/lib/codec/`, so a
//! change here is a change there. `codec_golden.json` is the pin that says so.

use goofi_codec::*;
use goofi_core::{Axes, Axis, Coord, Data, Meta, MetaValue, Param, Value};
use indexmap::IndexMap;

fn arr_bytes(d: &Data) -> (Vec<usize>, Vec<u8>) {
    match d.value() {
        Value::Array(s) => (s.shape().to_vec(), s.as_bytes().to_vec()),
        _ => panic!("not array"),
    }
}

#[test]
fn array_roundtrips_with_meta() {
    let mut meta = Meta::empty();
    meta.set_sfreq(Some(256.0));
    meta.set_ufreq(Some(128.0));
    meta.set_index(Some(42));
    meta.set("label", MetaValue::Str("eeg".into()));
    meta.set_channels(Axes::new().with(0, Axis::coords(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())])));
    let buf: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let d = Data::array_f32(vec![2], buf, meta).unwrap();

    let back = decode(&encode(&d)).expect("decode");
    let (sh, by) = arr_bytes(&back);
    assert_eq!(sh, vec![2]);
    assert_eq!(arr_bytes(&d).1, by);
    assert_eq!(back.meta().sfreq(), Some(256.0));
    assert_eq!(back.meta().ufreq(), Some(128.0));
    assert_eq!(back.meta().index(), Some(42));
    assert_eq!(back.meta().get("label"), Some(&MetaValue::Str("eeg".into())));
    let ch = back.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("dim0 channels");
    assert_eq!(ch.as_ref(), &[Coord::Str("Fz".into()), Coord::Str("Cz".into())]);
}

#[test]
fn a_foreign_dtype_arrives_as_f32() {
    // Array `Data` is always f32, so a `<i2` body a Python node produced is CAST on the way in.
    // Built as a whole frame, because the frame is the contract — the body is an encoder step.
    let mut body = vec![1u8]; // ndim
    body.push(3); // dtype string length
    body.extend_from_slice(b"<i2");
    body.extend_from_slice(&2u32.to_le_bytes()); // shape[0]
    body.extend_from_slice(&3i16.to_le_bytes());
    body.extend_from_slice(&(-4i16).to_le_bytes());

    let mut frame = Vec::new();
    frame.extend_from_slice(MAGIC);
    frame.push(VERSION);
    frame.push(0); // dtype tag ARRAY
    frame.extend_from_slice(&0u32.to_le_bytes()); // no meta
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(&body);

    let (sh, by) = arr_bytes(&decode(&frame).expect("an int16 body decodes"));
    assert_eq!(sh, vec![2]);
    let vals: Vec<f32> = by.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(vals, vec![3.0, -4.0], "the int16 body was cast to f32");
}

#[test]
fn a_meta_shadowing_a_derived_key_does_not_duplicate_it() {
    // A Python node is free to put any string key in its meta dict, so a node returning
    // `meta={'shape': …}` used to emit a msgpack map with TWO `shape` entries — malformed, and left
    // to each decoder to resolve. shape and dtype are derived; the derived value is the only one.
    let mut meta = Meta::empty();
    meta.set("shape", MetaValue::Str("user".into()));
    meta.set("dtype", MetaValue::Str("userd".into()));
    let buf: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let d = Data::array_f32(vec![2], buf, meta).unwrap();

    let bytes = encode(&d);
    let count = |k: &[u8]| bytes.windows(k.len()).filter(|w| *w == k).count();
    assert_eq!(count(b"shape"), 1, "exactly one shape entry on the wire");
    assert_eq!(count(b"dtype"), 1, "exactly one dtype entry on the wire");
    assert_eq!(decode(&bytes).unwrap().meta().get("shape"), None,
               "shape is derived, so it is never carried back");
}

#[test]
fn ufreq_none_is_absent_from_the_wire() {
    // An unmeasured slot (ufreq == None) must not emit the key, so frames that
    // predate ufreq (and the Python-golden fixtures) stay byte-identical.
    let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap();
    assert_eq!(d.meta().ufreq(), None);
    let bytes = encode(&d);
    assert!(
        !bytes.windows(5).any(|w| w == b"ufreq"),
        "None ufreq must not appear in the encoded meta"
    );
    assert_eq!(decode(&bytes).unwrap().meta().ufreq(), None);

    // A measured slot emits the key.
    let mut meta = Meta::empty();
    meta.set_ufreq(Some(60.0));
    let d2 = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), meta).unwrap();
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
        Data::array_f32(vec![1], 3.0f32.to_le_bytes().to_vec(), Meta::empty()).unwrap(),
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
    meta.set_sfreq(Some(64.0));
    let buf: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let d = Data::array_f32(vec![2, 4], buf, meta).unwrap();
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

// The subprocess multi-slot request/response frames — the shared wire between the
// parent (`goofi-python`'s `subproc::RemoteNode`) and the child (`goofi.serve` in pymod).

fn arr(shape: Vec<usize>, vals: &[f32], meta: Meta) -> Data {
    Data::array_f32(shape, vals.iter().flat_map(|v| v.to_le_bytes()).collect(), meta).unwrap()
}
fn floats(d: &Data) -> Vec<f32> {
    match d.value() {
        Value::Array(s) => s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => panic!("not array"),
    }
}

#[test]
fn request_roundtrips_multislot_data_and_params() {
    // Slot A: [2,3] with dim0 channels + sfreq + index (the EEG regression case).
    let mut ma = Meta::empty();
    ma.set_sfreq(Some(250.0));
    ma.set_index(Some(9));
    ma.set_channels(Axes::new().with(0, Axis::coords(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())])));
    let a = arr(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ma);
    // Slot B: plain [4].
    let b = arr(vec![4], &[7.0, 8.0, 9.0, 10.0], Meta::empty());

    // Two param groups, one of every scalar variant.
    let mut params: IndexMap<String, IndexMap<String, Param>> = IndexMap::new();
    let mut welch = IndexMap::new();
    welch.insert("nperseg".to_string(), Param::int(256, 16, 4096));
    welch.insert("scale".to_string(), Param::float(1.5, 0.0, 10.0));
    params.insert("welch".to_string(), welch);
    let mut flags = IndexMap::new();
    flags.insert("norm".to_string(), Param::boolean(true));
    flags.insert("mode".to_string(), Param::str_free("welch"));
    params.insert("flags".to_string(), flags);

    let frame = encode_request(&params, &[("data", &a), ("aux", &b)]);
    let (p2, slots) = decode_request(&frame).expect("decode request");

    // Params survived value-for-value.
    assert_eq!(p2, params, "params round-trip losslessly via serde");

    // Slots survived (order, names, shapes, values, meta).
    assert_eq!(slots.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["data", "aux"]);
    assert_eq!(floats(&slots[0].1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    match slots[0].1.value() {
        Value::Array(s) => assert_eq!(s.shape(), &[2, 3]),
        _ => panic!("expected array"),
    }
    assert_eq!(slots[0].1.meta().sfreq(), Some(250.0));
    assert_eq!(slots[0].1.meta().index(), Some(9));
    let ch = slots[0].1.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("dim0 channels survive");
    assert_eq!(ch.len(), 2);
    assert_eq!(floats(&slots[1].1), vec![7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn response_roundtrips_slots() {
    let a = arr(vec![2], &[1.0, 2.0], Meta::empty());
    let b = arr(vec![1, 3], &[3.0, 4.0, 5.0], Meta::empty());
    match decode_response(&encode_response(&[("psd", &a), ("extra", &b)])).expect("decode response") {
        Response::Slots(out) => {
            assert_eq!(out.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["psd", "extra"]);
            assert_eq!(floats(&out[0].1), vec![1.0, 2.0]);
            assert_eq!(floats(&out[1].1), vec![3.0, 4.0, 5.0]);
        }
        Response::NodeError(e) => panic!("expected slots, got error {e}"),
    }
}

#[test]
fn error_response_roundtrips_the_message() {
    // A per-tick node raise crosses as a NodeError variant (distinct from an OK slots
    // response and from a malformed frame), carrying the exception text verbatim.
    match decode_response(&encode_error_response("ValueError: boom")).expect("decode error response") {
        Response::NodeError(msg) => assert_eq!(msg, "ValueError: boom"),
        Response::Slots(_) => panic!("expected a NodeError"),
    }
    // An empty buffer is a malformed frame (outer Err), NOT a silent empty slots response.
    assert!(decode_response(&[]).is_err(), "empty response frame is malformed");
}

#[test]
fn empty_params_and_single_slot() {
    let params: IndexMap<String, IndexMap<String, Param>> = IndexMap::new();
    let a = arr(vec![1], &[42.0], Meta::empty());
    let (p2, slots) = decode_request(&encode_request(&params, &[("data", &a)])).unwrap();
    assert!(p2.is_empty());
    assert_eq!(slots.len(), 1);
    assert_eq!(floats(&slots[0].1), vec![42.0]);
}

// ---------------------------------------------------------------------------
// Liveness frames
// ---------------------------------------------------------------------------

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use goofi_codec::liveness::wait_for_parent_exit;

/// Run the watcher on a thread; the returned flag flips when it decides the parent died.
fn watch(reader: std::io::PipeReader) -> Arc<AtomicBool> {
    let died = Arc::new(AtomicBool::new(false));
    let flag = died.clone();
    std::thread::spawn(move || {
        wait_for_parent_exit(reader);
        flag.store(true, Ordering::SeqCst);
    });
    died
}

/// Bounded wait for the flag, with an actionable message — never an unbounded hang.
fn assert_flips(died: &AtomicBool, what: &str) {
    let t = Instant::now();
    while !died.load(Ordering::SeqCst) {
        assert!(t.elapsed() < Duration::from_secs(2), "the watcher never noticed {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_watcher_blocks_while_the_parent_lives_and_returns_on_eof() {
    let (reader, writer) = io::pipe().expect("pipe");
    let died = watch(reader);
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        !died.load(Ordering::SeqCst),
        "the watcher must BLOCK while the parent still holds the write end"
    );
    drop(writer); // the parent dies — the OS closes its write end
    assert_flips(&died, "the closed write end (EOF)");
}

#[test]
fn a_stray_byte_is_not_the_parents_death() {
    let (reader, mut writer) = io::pipe().expect("pipe");
    let died = watch(reader);
    writer.write_all(b"ping").expect("write");
    std::thread::sleep(Duration::from_millis(100));
    assert!(!died.load(Ordering::SeqCst), "a byte on the pipe is not the parent's death");
    drop(writer);
    assert_flips(&died, "the closed write end after a stray byte");
}
