//! TableSelectString — pull one entry out of a TABLE by key and emit it as a
//! STRING. The key comes from the `selection.key` param, but an optional `key`
//! STRING input overrides it (and sticks — the Python assigns the input back onto
//! the param before clearing the slot, so a later table re-emits with the same
//! key). Ported from `nodes/misc/tableselectstring.py`.
//!
//! The selected entry is stringified like the Python `str(value.data)`: a STRING
//! passes through verbatim; anything else is rendered to a compact, deterministic
//! label (see `stringify`) rather than reproducing numpy/dict text exactly — the
//! output is a human-readable string, not a wire-load-bearing value.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct TableSelectString {
    key: String,
}

/// Mirror Python's `str(value.data)` for the selected table entry. A STRING is
/// returned verbatim (Python skips coercion for it). Exact numpy/dict formatting
/// isn't reproduced; instead an array becomes its flat f32 values in Rust debug
/// form (`[1.0, 2.0]`) and a nested table becomes its key list (`["a", "b"]`) —
/// faithful, compact, and deterministic.
fn stringify(value: &Value) -> String {
    match value {
        Value::Str(s) => s.to_string(),
        Value::Array(store) => {
            if store.dtype() == DType::F32 {
                let vals: Vec<f32> = store
                    .as_bytes()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                format!("{vals:?}")
            } else {
                // A non-f32 array would misdecode if read as f32; describe it instead.
                format!("<array {} shape={:?}>", store.dtype().numpy_name(), store.shape())
            }
        }
        Value::Table(map) => {
            let keys: Vec<&str> = map.keys().map(String::as_str).collect();
            format!("{keys:?}")
        }
    }
}

impl Node for TableSelectString {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // input_table is required; a missing or wrong-typed input is a no-emit tick
        // (Python's `if input_table is None: return None`, checked before the key).
        let Some(td) = inp.get("input_table") else {
            return Ok(());
        };
        let Value::Table(map) = td.value() else {
            return Ok(());
        };

        // A fresh `key` string input overrides the param and persists on the field,
        // mirroring the Python writing it back onto `params.selection.key` before
        // clearing the slot (so a later table re-emits with the same key).
        if let Some(kd) = inp.get("key") {
            if let Value::Str(s) = kd.value() {
                self.key = s.to_string();
            }
        }

        // Key absent from the table -> no emit (`if key not in input_table.data`).
        let Some(selected) = map.get(self.key.as_str()) else {
            return Ok(());
        };

        // Output carries the INPUT TABLE's meta, not the selected value's — faithful
        // to Python returning `(value.data, input_table.meta)`.
        out.set("output_string", Data::string(stringify(selected.value()), td.meta().clone()));
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "selection" && key.name == "key" {
            if let Some(s) = v.as_str() {
                self.key = s.to_string();
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("key".to_string(), Param::str_free("default_key"));
    let mut groups = ParamGroups::new();
    groups.insert("selection".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let key = param(p, "selection", "key")
        .and_then(Param::as_str)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default_key".to_string());
    Box::new(TableSelectString { key })
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "input_table", kind: SlotType::Table, trigger_process: true, multi: false },
    SlotDecl { name: "key", kind: SlotType::String, trigger_process: true, multi: false },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output_string",
    kind: SlotType::String,
}];

inventory::submit! {
    NodeManifest {
        type_name: "TableSelectString",
        category: "misc",
        doc: "Select a table entry by key and emit it as a string.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    /// A TABLE Data from `(key, Data)` pairs (insertion order preserved), with meta.
    fn table(pairs: Vec<(&str, Data)>, meta: Meta) -> Data {
        let mut m: IndexMap<String, Data> = IndexMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), v);
        }
        Data::table(m, meta)
    }

    fn arr(vals: &[f32]) -> Data {
        let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], buf, Meta::empty()).unwrap()
    }

    fn str_data(s: &str) -> Data {
        Data::string(s, Meta::empty())
    }

    /// Drive the node once. `param_key` overrides the default `selection.key`;
    /// `key_input` supplies the optional `key` STRING slot; `table` is the
    /// `input_table` slot (absent when `None`). Returns the emitted Data, if any.
    fn run(param_key: Option<&str>, key_input: Option<&str>, table: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("TableSelectString").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        if let Some(k) = param_key {
            node.on_param_changed(&ParamKey::new("selection", "key"), &Param::str_free(k))
                .unwrap();
        }
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("input_table", table);
        inmap.insert("key", key_input.map(str_data));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new())
            .unwrap();
        outbuf.get("output_string").unwrap().clone()
    }

    fn out_str(d: &Data) -> String {
        match d.value() {
            Value::Str(s) => s.to_string(),
            _ => panic!("expected string output"),
        }
    }

    #[test]
    fn selects_string_entry_by_default_param_key() {
        // No param override, no key input -> uses the default_key param.
        let t = table(vec![("default_key", str_data("hello"))], Meta::empty());
        let d = run(None, None, Some(t)).unwrap();
        assert_eq!(out_str(&d), "hello");
    }

    #[test]
    fn param_key_selects_named_entry() {
        let t = table(vec![("a", str_data("x")), ("b", str_data("y"))], Meta::empty());
        let d = run(Some("b"), None, Some(t)).unwrap();
        assert_eq!(out_str(&d), "y");
    }

    #[test]
    fn key_input_overrides_param() {
        // Param says "a" but the key input says "b" -> selects "b".
        let t = table(vec![("a", str_data("x")), ("b", str_data("y"))], Meta::empty());
        let d = run(Some("a"), Some("b"), Some(t)).unwrap();
        assert_eq!(out_str(&d), "y");
    }

    #[test]
    fn missing_key_is_no_emit() {
        let t = table(vec![("a", str_data("x"))], Meta::empty());
        assert!(run(Some("z"), None, Some(t)).is_none());
    }

    #[test]
    fn absent_table_input_is_no_emit() {
        assert!(run(Some("a"), Some("a"), None).is_none());
    }

    #[test]
    fn wrong_typed_table_input_is_no_emit() {
        // input_table carries a STRING, not a TABLE -> no emit (not a panic).
        assert!(run(Some("a"), None, Some(str_data("not a table"))).is_none());
    }

    #[test]
    fn coerces_array_entry_to_debug_string() {
        // A non-STRING entry is stringified: flat f32 values in Rust debug form.
        let t = table(vec![("k", arr(&[1.0, 2.0, 3.0]))], Meta::empty());
        let d = run(Some("k"), None, Some(t)).unwrap();
        assert_eq!(out_str(&d), "[1.0, 2.0, 3.0]");
    }

    #[test]
    fn coerces_nested_table_entry_to_key_list() {
        // A nested TABLE entry stringifies to its key list (insertion order).
        let inner = table(vec![("x", str_data("1")), ("y", str_data("2"))], Meta::empty());
        let t = table(vec![("k", inner)], Meta::empty());
        let d = run(Some("k"), None, Some(t)).unwrap();
        assert_eq!(out_str(&d), "[\"x\", \"y\"]");
    }

    #[test]
    fn output_carries_input_table_meta() {
        // The emitted string's meta is the INPUT TABLE's meta, not the entry's.
        let mut meta = Meta::empty();
        meta.sfreq = Some(250.0);
        let t = table(vec![("k", str_data("v"))], meta);
        let d = run(Some("k"), None, Some(t)).unwrap();
        assert_eq!(d.meta().sfreq, Some(250.0));
    }

    #[test]
    fn key_input_sticks_across_ticks() {
        // Python assigns the key input back onto the param, so a later tick with no
        // key input still uses the last-supplied key.
        let m = goofi_node::find("TableSelectString").unwrap();
        let mut node = (m.make)(&(m.default_params)());

        let make_table = || table(vec![("a", str_data("x")), ("b", str_data("y"))], Meta::empty());

        // Tick 1: key input "b" -> selects "b", and sticks.
        let mut in1: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        in1.insert("input_table", Some(make_table()));
        in1.insert("key", Some(str_data("b")));
        let mut out1 = m.output_buffer();
        node.process(&Inputs::new(&in1), &mut Outputs::new(&mut out1), &mut NodeCtx::new())
            .unwrap();
        assert_eq!(out_str(out1.get("output_string").unwrap().as_ref().unwrap()), "y");

        // Tick 2: no key input, only the table -> still uses the stuck "b".
        let mut in2: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        in2.insert("input_table", Some(make_table()));
        in2.insert("key", None);
        let mut out2 = m.output_buffer();
        node.process(&Inputs::new(&in2), &mut Outputs::new(&mut out2), &mut NodeCtx::new())
            .unwrap();
        assert_eq!(out_str(out2.get("output_string").unwrap().as_ref().unwrap()), "y");
    }

    #[test]
    fn registered_in_catalog_with_correct_slots() {
        let m = goofi_node::find("TableSelectString").expect("registered via inventory");
        assert_eq!(m.category, "misc");
        assert_eq!(m.inputs.len(), 2);
        assert_eq!(m.inputs[0].name, "input_table");
        assert_eq!(m.inputs[0].kind, goofi_core::SlotType::Table);
        assert_eq!(m.inputs[1].name, "key");
        assert_eq!(m.inputs[1].kind, goofi_core::SlotType::String);
        assert_eq!(m.outputs.len(), 1);
        assert_eq!(m.outputs[0].name, "output_string");
        assert_eq!(m.outputs[0].kind, goofi_core::SlotType::String);
    }
}
