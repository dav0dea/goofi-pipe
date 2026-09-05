//! Golden conformance against the legacy Python codec: byte-for-byte on the deterministic parts,
//! semantically on the msgpack meta. The case list below MIRRORS `tests/gen_golden.py`.

use std::collections::BTreeMap;

use goofi_codec::{encode, split_frame};
use goofi_core::{Axes, Axis, Coord, Data, Meta};
use indexmap::IndexMap;

fn arr(shape: &[usize], buf: Vec<u8>, meta: Meta) -> Data {
    Data::array_f32(shape.to_vec(), buf, meta).unwrap()
}

fn le_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn build_cases() -> Vec<(&'static str, Data)> {
    let meta_sfreq_ch = Meta::new()
        .with_sfreq(Some(250.0))
        .with_channels(Axes::new().with(0, Axis::coords(vec![Coord::Str("Fz".into()), Coord::Str("Cz".into())])));
    let meta_index = Meta::new().with_index(Some(42));

    let mut table = IndexMap::new();
    table.insert("a".to_string(), arr(&[2], le_bytes(&[1.0, 2.0]), Meta::empty()));
    table.insert("b".to_string(), Data::string("x", Meta::empty()));

    vec![
        ("f32_1d", arr(&[3], le_bytes(&[1.0, 2.0, 3.0]), Meta::empty())),
        // 0-d scalar promotes to shape (1,)
        ("scalar_0d", arr(&[], le_bytes(&[3.0]), Meta::empty())),
        ("empty_array", arr(&[0], vec![], Meta::empty())),
        (
            "with_sfreq_channels",
            arr(&[2, 3], le_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), meta_sfreq_ch),
        ),
        ("with_index", arr(&[4], le_bytes(&[1.0, 2.0, 3.0, 4.0]), meta_index)),
        ("string", Data::string("hello world éà", Meta::empty())),
        ("string_empty", Data::string("", Meta::empty())),
        ("table", Data::table(table, Meta::empty())),
        ("table_empty", Data::table(IndexMap::new(), Meta::empty())),
    ]
}

#[derive(PartialEq, Debug)]
enum Canon {
    Nil,
    Bool(bool),
    Int(i128),
    F(u64),
    Str(String),
    Bytes(Vec<u8>),
    Arr(Vec<Canon>),
    Map(BTreeMap<String, Canon>),
}

fn canon(v: &rmpv::Value) -> Canon {
    use rmpv::Value::*;
    match v {
        Nil => Canon::Nil,
        Boolean(b) => Canon::Bool(*b),
        Integer(i) => Canon::Int(
            i.as_i64()
                .map(|x| x as i128)
                .or_else(|| i.as_u64().map(|x| x as i128))
                .expect("integer fits"),
        ),
        F32(f) => Canon::F((*f as f64).to_bits()),
        F64(f) => Canon::F(f.to_bits()),
        String(s) => Canon::Str(s.as_str().unwrap_or_default().to_string()),
        Binary(b) => Canon::Bytes(b.clone()),
        Array(a) => Canon::Arr(a.iter().map(canon).collect()),
        Map(m) => Canon::Map(
            m.iter()
                .map(|(k, v)| (k.as_str().unwrap().to_string(), canon(v)))
                .collect(),
        ),
        Ext(_, _) => panic!("unexpected msgpack ext in meta"),
    }
}

fn decode_meta(bytes: &[u8]) -> rmpv::Value {
    let mut rd = bytes;
    rmpv::decode::read_value(&mut rd).expect("decode meta msgpack")
}

fn load_fixture() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/goof_golden.json");
    let text = std::fs::read_to_string(path).expect("read fixture (run gen_golden.py)");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[test]
fn goof_encoder_matches_python_golden() {
    let fixture = load_fixture();
    let cases_json = fixture.get("cases").expect("cases key");
    let native = build_cases();

    let py_names: Vec<&str> = cases_json.as_object().unwrap().keys().map(|s| s.as_str()).collect();
    for name in &py_names {
        assert!(
            native.iter().any(|(n, _)| n == name),
            "Python golden case `{name}` has no matching Rust case"
        );
    }
    assert_eq!(native.len(), py_names.len(), "case count mismatch Rust vs Python");

    for (name, data) in &native {
        let py_frame: Vec<u8> = cases_json[name]["frame"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as u8)
            .collect();

        let rust_frame = encode(data);

        let (rt_tag, rt_meta, rt_body) = split_frame(&rust_frame).expect("rust frame valid");
        let (py_tag, py_meta, py_body) = split_frame(&py_frame).expect("py frame valid");

        assert_eq!(rt_tag, py_tag, "[{name}] dtype tag");
        assert_eq!(rt_body, py_body, "[{name}] body bytes must be byte-identical");
        assert_eq!(
            canon(&decode_meta(rt_meta)),
            canon(&decode_meta(py_meta)),
            "[{name}] meta must be semantically equal"
        );
    }
}

#[test]
fn a_request_carries_each_multi_frame_with_its_source() {
    // A multi slot's entries cross under one name repeated, each with the `node.slot` that sent
    // it; a single slot's entry crosses with no source, and an output never has one.
    let cases = build_cases();
    let (a, b) = (&cases[0].1, &cases[1].1);
    let params = goofi_codec::ParamMap::new();
    let bytes = goofi_codec::encode_request(&params, &[("input", "alpha.out", a), ("input", "beta.out", b), ("gate", "", a)]);
    let goofi_codec::Request::Process { slots, .. } = goofi_codec::decode_request(&bytes).expect("a request") else {
        panic!("a run, not a refresh");
    };
    let named: Vec<(&str, &str)> = slots.iter().map(|(n, s, _)| (n.as_str(), s.as_str())).collect();
    assert_eq!(named, vec![("input", "alpha.out"), ("input", "beta.out"), ("gate", "")]);
    assert_eq!(encode(&slots[1].2), encode(b), "the second entry is beta's own frame");
    let reply = goofi_codec::decode_response(&goofi_codec::encode_response(&[("out", b)])).expect("a response");
    let goofi_codec::Response::Slots(outs) = reply else { panic!("slots") };
    assert_eq!((outs[0].0.as_str(), encode(&outs[0].1)), ("out", encode(b)));
}
