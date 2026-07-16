//! SetMeta — pass an array straight through while stamping ONE metadata
//! key/value pair onto it, type-cast per a `type` param (int/float/bool/str).
//! The array payload is left byte-for-byte identical; only the `Meta` sidecar
//! gains the key. Ported from `nodes/misc/setmeta.py`.

use goofi_core::SlotType;
use goofi_core::{Data, DType, MetaValue, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct SetMeta {
    key: String,
    value: String,
    // Kept as the raw param string (not a parsed enum) so an out-of-options
    // `type` still lands in the defensive no-emit branch, mirroring Python's
    // `else: raise`.
    ty: String,
}

impl Node for SetMeta {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("array") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }

        // Cast `value` into a typed meta value per `type`. A bad numeric parse
        // (or an unknown `type`) is an invalid edit -> silent no-op, where the
        // Python raises. Python `int`/`float` strip surrounding whitespace, so
        // trim to match; `bool`/`str` take the raw string (see below).
        let mv = match self.ty.as_str() {
            "int" => match self.value.trim().parse::<i64>() {
                Ok(n) => MetaValue::Int(n),
                Err(_) => return Ok(()),
            },
            "float" => match self.value.trim().parse::<f64>() {
                Ok(f) => MetaValue::Float(f),
                Err(_) => return Ok(()),
            },
            // Python `bool(str)` is truthy for EVERY non-empty string — even
            // "false" and "0"; only "" is falsy. Replicate that exactly (raw
            // string, no trim: a whitespace-only value is non-empty -> true).
            "bool" => MetaValue::Bool(!self.value.is_empty()),
            "str" => MetaValue::Str(self.value.clone()),
            _ => return Ok(()),
        };

        // The Rust `Meta` hoists the reserved keys out of the open map into
        // derived/typed fields: shape+dtype are derived from the array store,
        // and channels/sfreq/index/reduced are structured fields. None can be
        // expressed as a free scalar in `extra` (channels least of all), so
        // SetMeta only writes open-map keys and no-ops on a reserved name.
        if matches!(
            self.key.as_str(),
            "shape" | "dtype" | "channels" | "sfreq" | "index" | "reduced"
        ) {
            return Ok(());
        }

        // Clone the input meta (never mutate the producer's immutable copy) and
        // write the one key into the open map.
        let mut meta = d.meta().clone();
        meta.extra.insert(self.key.clone(), mv);

        // Pass the F32 payload through unchanged (same bytes, same shape); the
        // meta clone above carries the single new key.
        let shape = store.shape().to_vec();
        let buf = store.as_bytes().to_vec();
        let data =
            Data::from_array_bytes(DType::F32, shape, buf, meta).map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("meta", "key") => {
                if let Some(s) = v.as_str() {
                    self.key = s.to_string();
                }
            }
            ("meta", "value") => {
                if let Some(s) = v.as_str() {
                    self.value = s.to_string();
                }
            }
            ("meta", "type") => {
                if let Some(s) = v.as_str() {
                    self.ty = s.to_string();
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("key".to_string(), Param::str_free("key"));
    g.insert("value".to_string(), Param::str_free("value"));
    g.insert(
        "type".to_string(),
        Param::Str {
            value: "float".to_string(),
            options: Some(vec![
                "int".to_string(),
                "float".to_string(),
                "bool".to_string(),
                "str".to_string(),
            ]),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("meta".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let key = param(p, "meta", "key").and_then(Param::as_str).unwrap_or("key").to_string();
    let value = param(p, "meta", "value").and_then(Param::as_str).unwrap_or("value").to_string();
    let ty = param(p, "meta", "type").and_then(Param::as_str).unwrap_or("float").to_string();
    Box::new(SetMeta { key, value, ty })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "array",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "SetMeta",
        category: "misc",
        doc: "Pass an array through unchanged while stamping one type-cast metadata key onto it.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, DType, Meta, MetaValue, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Drive SetMeta on a pre-built frame; returns the emitted `out` Data (or
    /// `None` when the node no-ops).
    fn run(key: &str, value: &str, ty: &str, frame: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("SetMeta").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("meta", "key"), &goofi_core::Param::str_free(key)).unwrap();
        node.on_param_changed(&ParamKey::new("meta", "value"), &goofi_core::Param::str_free(value)).unwrap();
        node.on_param_changed(&ParamKey::new("meta", "type"), &goofi_core::Param::str_free(ty)).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", frame);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("out").unwrap().as_ref().cloned()
    }

    fn f32_frame(shape: Vec<usize>, data: &[f32], meta: Meta) -> Data {
        let buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, shape, buf, meta).unwrap()
    }

    fn vals(d: &Data) -> (Vec<usize>, Vec<f32>) {
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn writes_float_key_and_passes_array_through() {
        // The array must survive byte-for-byte; only the meta gains the key.
        let f = f32_frame(vec![3], &[1.0, 2.0, 3.0], Meta::empty());
        let d = run("gain", "2.5", "float", Some(f)).expect("emits");
        assert_eq!(vals(&d), (vec![3], vec![1.0, 2.0, 3.0]), "payload unchanged");
        assert_eq!(d.meta().extra.get("gain"), Some(&MetaValue::Float(2.5)));
    }

    #[test]
    fn writes_int_key() {
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        let d = run("n", "42", "int", Some(f)).expect("emits");
        assert_eq!(d.meta().extra.get("n"), Some(&MetaValue::Int(42)));
    }

    #[test]
    fn writes_str_key_verbatim() {
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        let d = run("label", "hello world", "str", Some(f)).expect("emits");
        assert_eq!(d.meta().extra.get("label"), Some(&MetaValue::Str("hello world".into())));
    }

    #[test]
    fn bool_follows_python_truthiness() {
        // Python bool(str): only "" is falsy; "false" and "0" are both truthy.
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        for truthy in ["false", "0", "False", "x", " "] {
            let d = run("flag", truthy, "bool", Some(f.clone())).expect("emits");
            assert_eq!(d.meta().extra.get("flag"), Some(&MetaValue::Bool(true)), "{truthy:?} -> true");
        }
        let d = run("flag", "", "bool", Some(f)).expect("emits");
        assert_eq!(d.meta().extra.get("flag"), Some(&MetaValue::Bool(false)), "empty -> false");
    }

    #[test]
    fn preserves_existing_meta_and_leaves_producer_untouched() {
        let producer = f32_frame(vec![2], &[10.0, 20.0], Meta { sfreq: Some(256.0), ..Default::default() });
        let d = run("tag", "1", "int", Some(producer.clone())).expect("emits");
        assert_eq!(d.meta().sfreq, Some(256.0), "unrelated meta survives the passthrough");
        assert_eq!(d.meta().extra.get("tag"), Some(&MetaValue::Int(1)));
        // The producer's immutable meta never gained the key (clone-on-write).
        assert!(producer.meta().extra.is_empty(), "producer meta not mutated");
    }

    #[test]
    fn preserves_channel_coords() {
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("Fp1".into()), Coord::Str("Fp2".into())]));
        let f = f32_frame(vec![2], &[1.0, 2.0], Meta { channels: Channels(ch), ..Default::default() });
        let d = run("k", "0", "int", Some(f)).expect("emits");
        let kept = d.meta().channels.0.get(&0).expect("channel coords survive");
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn no_emit_on_unparseable_number() {
        // Python float("value") raises -> we no-op (this is also the default-param case).
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        assert!(run("k", "value", "float", Some(f.clone())).is_none());
        assert!(run("k", "3.5", "int", Some(f)).is_none(), "int rejects a decimal, like Python int()");
    }

    #[test]
    fn no_emit_on_unknown_type() {
        // Defensive: an out-of-options `type` behaves like Python's `else: raise`.
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        assert!(run("k", "1", "complex", Some(f)).is_none());
    }

    #[test]
    fn no_emit_on_reserved_key() {
        // Reserved names live in typed/derived Meta fields, not the open map.
        let f = f32_frame(vec![1], &[0.0], Meta::empty());
        for reserved in ["sfreq", "index", "channels", "reduced", "shape", "dtype"] {
            assert!(run(reserved, "1", "int", Some(f.clone())).is_none(), "{reserved} -> no-op");
        }
    }

    #[test]
    fn no_emit_on_missing_input() {
        assert!(run("k", "1", "int", None).is_none());
    }

    #[test]
    fn no_emit_on_non_f32_dtype() {
        let f = Data::from_array_bytes(DType::U8, vec![3], vec![1u8, 2, 3], Meta::empty()).unwrap();
        assert!(run("k", "1", "int", Some(f)).is_none());
    }
}
