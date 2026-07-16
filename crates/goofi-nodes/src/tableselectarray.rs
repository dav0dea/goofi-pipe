//! TableSelectArray — pick one entry out of a TABLE by key and emit it as an
//! ARRAY. Ported from `nodes/misc/tableselectarray.py`. The key comes from the
//! `selection.key` param, but a live `key` STRING input overrides it for that
//! tick. Reading is by `Value` variant, so a missing/wrong-typed input or a
//! non-array value at the key is a silent no-emit (the Python raises on a
//! non-array; the Rust contract stays silent instead).

use goofi_core::SlotType;
use goofi_core::{Data, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct TableSelectArray {
    key: String,
}

impl Node for TableSelectArray {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Required TABLE input; absent or wrong-typed -> no emit.
        let Some(table_data) = inp.get("input_table") else {
            return Ok(());
        };
        let Value::Table(map) = table_data.value() else {
            return Ok(());
        };
        // Effective key: prefer this tick's `key` string input over the param, but
        // never mutate the param (the engine owns param state — we just read the
        // input this tick). The slot is STRING-typed, so a non-string there is
        // pathological; fall back to the param rather than crash.
        let effective_key: &str = match inp.get("key").map(|d| d.value()) {
            Some(Value::Str(s)) => s.as_ref(),
            _ => self.key.as_str(),
        };
        // Key not present in the table -> no emit (Python returns None).
        let Some(selected) = map.get(effective_key) else {
            return Ok(());
        };
        // Only an ARRAY entry is emittable; anything else is a no-emit (Python
        // raises ValueError, but bad data is a silent skip in the Rust contract).
        let Value::Array(store) = selected.value() else {
            return Ok(());
        };
        // Emit the selected array carrying the INPUT TABLE's meta — faithful to the
        // Python `(selected_value.data, input_table.meta)`, not the entry's own meta.
        out.set("output_array", Data::array(store.clone(), table_data.meta().clone()));
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
        .unwrap_or("default_key")
        .to_string();
    Box::new(TableSelectArray { key })
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "input_table", kind: SlotType::Table, trigger_process: true, multi: false },
    SlotDecl { name: "key", kind: SlotType::String, trigger_process: true, multi: false },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output_array",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "TableSelectArray",
        category: "misc",
        doc: "Select one array from a table by key (a live `key` string input overrides the param).",
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
    use goofi_node::{Inputs, Node, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    /// A fresh node with an optional `selection.key` param override.
    fn make_node(param_key: Option<&str>) -> Box<dyn Node> {
        let m = goofi_node::find("TableSelectArray").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        if let Some(k) = param_key {
            node.on_param_changed(&ParamKey::new("selection", "key"), &Param::str_free(k))
                .unwrap();
        }
        node
    }

    /// Drive one tick with the given table and optional `key` string input; return
    /// whatever landed on `output_array` (None = no emit).
    fn drive(node: &mut Box<dyn Node>, table: Option<Data>, key_input: Option<&str>) -> Option<Data> {
        let m = goofi_node::find("TableSelectArray").unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("input_table", table);
        inmap.insert("key", key_input.map(|s| Data::string(s, Meta::empty())));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        outbuf.get("output_array").unwrap().as_ref().cloned()
    }

    fn arr(vals: &[f32], meta: Meta) -> Data {
        let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], buf, meta).unwrap()
    }

    fn as_f32(d: &Data) -> Vec<f32> {
        match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("expected array output"),
        }
    }

    fn table(entries: Vec<(&str, Data)>, meta: Meta) -> Data {
        let mut map: IndexMap<String, Data> = IndexMap::new();
        for (k, v) in entries {
            map.insert(k.to_string(), v);
        }
        Data::table(map, meta)
    }

    #[test]
    fn emits_array_at_default_param_key() {
        // Default param key is "default_key" (no override, no key input).
        let mut node = make_node(None);
        let tbl = table(vec![("default_key", arr(&[1.0, 2.0, 3.0], Meta::empty()))], Meta::empty());
        let out = drive(&mut node, Some(tbl), None).expect("emits the default_key array");
        assert_eq!(as_f32(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn param_key_selects_entry() {
        let mut node = make_node(Some("b"));
        let tbl = table(
            vec![("a", arr(&[1.0], Meta::empty())), ("b", arr(&[9.0, 8.0], Meta::empty()))],
            Meta::empty(),
        );
        let out = drive(&mut node, Some(tbl), None).expect("emits");
        assert_eq!(as_f32(&out), vec![9.0, 8.0]);
    }

    #[test]
    fn key_input_overrides_param() {
        // Param says "a" but the live key input says "b" -> "b" wins this tick.
        let mut node = make_node(Some("a"));
        let tbl = table(
            vec![("a", arr(&[1.0], Meta::empty())), ("b", arr(&[2.0], Meta::empty()))],
            Meta::empty(),
        );
        let out = drive(&mut node, Some(tbl), Some("b")).expect("emits");
        assert_eq!(as_f32(&out), vec![2.0]);
    }

    #[test]
    fn missing_key_no_emit() {
        let mut node = make_node(Some("nope"));
        let tbl = table(vec![("a", arr(&[1.0], Meta::empty()))], Meta::empty());
        assert!(drive(&mut node, Some(tbl), None).is_none());
    }

    #[test]
    fn absent_table_no_emit() {
        let mut node = make_node(None);
        assert!(drive(&mut node, None, None).is_none());
    }

    #[test]
    fn wrong_typed_table_no_emit() {
        // A STRING on the input_table slot is not a table -> no emit, no panic.
        let mut node = make_node(None);
        let not_a_table = Data::string("hi", Meta::empty());
        assert!(drive(&mut node, Some(not_a_table), None).is_none());
    }

    #[test]
    fn non_array_value_no_emit() {
        // The entry exists but is a STRING, not an ARRAY -> no emit (Python raises).
        let mut node = make_node(Some("s"));
        let tbl = table(vec![("s", Data::string("text", Meta::empty()))], Meta::empty());
        assert!(drive(&mut node, Some(tbl), None).is_none());
    }

    #[test]
    fn output_carries_table_meta_not_inner() {
        // The emitted array must carry the input table's meta, not the entry's own.
        let mut node = make_node(Some("k"));
        let mut inner_meta = Meta::empty();
        inner_meta.sfreq = Some(999.0);
        let mut table_meta = Meta::empty();
        table_meta.sfreq = Some(250.0);
        let tbl = table(vec![("k", arr(&[1.0], inner_meta))], table_meta);
        let out = drive(&mut node, Some(tbl), None).expect("emits");
        assert_eq!(out.meta().sfreq, Some(250.0), "uses the table's meta, not the inner array's");
    }
}
