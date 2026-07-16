//! AppendTables — combine two TABLE inputs into one. With both present it merges
//! their entry maps and their `Meta` sidecars (table2 wins on a key collision);
//! with only one present it passes that table through untouched; with neither it
//! emits nothing. Ported from `nodes/misc/appendtables.py`. Reads/writes TABLE, so
//! the input guard is on `Value::Table` (a wrong-typed frame counts as absent —
//! the slot is TABLE-typed, so this is purely defensive).

use goofi_core::SlotType;
use goofi_core::{Data, Meta, Value};
use goofi_node::{
    Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs, ParamGroups,
    SlotDecl,
};
use indexmap::IndexMap;

struct AppendTables;

/// Overlay `over` onto a clone of `base`, mirroring Python's flat `{**m1, **m2}`
/// merge of the two meta dicts: table2's keys win. The Rust `Meta` splits those
/// keys into typed fields + an open map, so overlay each present field (a `Some`
/// typed field / a non-empty channels map / every `extra` entry) and otherwise
/// keep table1's — a field table2 never set is absent in its dict, so it must not
/// clobber table1's value.
fn merge_meta(base: &Meta, over: &Meta) -> Meta {
    let mut m = base.clone();
    if over.sfreq.is_some() {
        m.sfreq = over.sfreq;
    }
    if over.index.is_some() {
        m.index = over.index;
    }
    if over.reduced.is_some() {
        m.reduced = over.reduced.clone();
    }
    if !over.channels.0.is_empty() {
        m.channels = over.channels.clone();
    }
    for (k, v) in &over.extra {
        m.extra.insert(k.clone(), v.clone());
    }
    m
}

impl Node for AppendTables {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Only a Value::Table counts as present; a missing or wrong-typed frame is
        // treated as absent (Python's slot layer guarantees TABLE-or-None).
        let t1 = inp.get("table1").filter(|d| matches!(d.value(), Value::Table(_)));
        let t2 = inp.get("table2").filter(|d| matches!(d.value(), Value::Table(_)));
        match (t1, t2) {
            // Both absent -> no emit (Python `return None`).
            (None, None) => {}
            // One present -> pass it through unchanged (its data + meta ride along
            // in the single Arc bump of `clone`).
            (Some(d), None) | (None, Some(d)) => out.set("output_table", d.clone()),
            // Both present -> merge. Insert all of table1 then all of table2, so
            // table2 overrides on a key collision while keeping table1's insertion
            // slot (IndexMap semantics == Python `{**t1, **t2}`); merge meta likewise.
            (Some(d1), Some(d2)) => {
                if let (Value::Table(m1), Value::Table(m2)) = (d1.value(), d2.value()) {
                    let mut merged: IndexMap<String, Data> = (**m1).clone();
                    for (k, v) in m2.iter() {
                        merged.insert(k.clone(), v.clone());
                    }
                    out.set("output_table", Data::table(merged, merge_meta(d1.meta(), d2.meta())));
                }
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    ParamGroups::new()
}

fn make(_p: &ParamGroups) -> Box<dyn Node> {
    Box::new(AppendTables)
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "table1", kind: SlotType::Table, trigger_process: true, multi: false },
    SlotDecl { name: "table2", kind: SlotType::Table, trigger_process: true, multi: false },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output_table",
    kind: SlotType::Table,
}];

inventory::submit! {
    NodeManifest {
        type_name: "AppendTables",
        category: "misc",
        doc: "Merge two tables into one (table2 wins on key collision); pass a lone input through.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Channels, Coord, Data, Meta, MetaValue, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// A one-cell string `Data` — table values here are strings so a test can
    /// assert *which* input's value survived a key collision.
    fn cell(s: &str) -> Data {
        Data::string(s, Meta::empty())
    }

    /// Build a TABLE `Data` from ordered (key, value) string pairs + a meta.
    fn table(pairs: &[(&str, &str)], meta: Meta) -> Data {
        let mut m: IndexMap<String, Data> = IndexMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), cell(v));
        }
        Data::table(m, meta)
    }

    /// Drive AppendTables once with the two (optional) inputs; return the emitted
    /// `output_table` Data, or `None` when the node no-ops.
    fn run(t1: Option<Data>, t2: Option<Data>) -> Option<Data> {
        let m = goofi_node::find("AppendTables").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("table1", t1);
        inmap.insert("table2", t2);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        outbuf.get("output_table").unwrap().as_ref().cloned()
    }

    /// Flatten a TABLE `Data` to ordered (key, string-value) pairs.
    fn as_map(d: &Data) -> Vec<(String, String)> {
        match d.value() {
            Value::Table(m) => m
                .iter()
                .map(|(k, v)| match v.value() {
                    Value::Str(s) => (k.clone(), s.to_string()),
                    _ => panic!("expected string cell"),
                })
                .collect(),
            _ => panic!("expected table output"),
        }
    }

    #[test]
    fn merges_disjoint_keys_in_table1_then_table2_order() {
        let t1 = table(&[("a", "1"), ("b", "2")], Meta::empty());
        let t2 = table(&[("c", "3")], Meta::empty());
        let d = run(Some(t1), Some(t2)).expect("emits");
        assert_eq!(
            as_map(&d),
            vec![("a".into(), "1".into()), ("b".into(), "2".into()), ("c".into(), "3".into())],
        );
    }

    #[test]
    fn table2_overrides_on_collision_keeping_table1_position() {
        // `b` exists in both: table2's value wins but the entry stays in table1's
        // slot (IndexMap insert keeps position), exactly like Python `{**t1, **t2}`.
        let t1 = table(&[("a", "1"), ("b", "2")], Meta::empty());
        let t2 = table(&[("b", "99"), ("c", "3")], Meta::empty());
        let d = run(Some(t1), Some(t2)).expect("emits");
        assert_eq!(
            as_map(&d),
            vec![("a".into(), "1".into()), ("b".into(), "99".into()), ("c".into(), "3".into())],
        );
    }

    #[test]
    fn output_is_a_table() {
        let d = run(Some(table(&[("a", "1")], Meta::empty())), Some(table(&[("b", "2")], Meta::empty())))
            .expect("emits");
        assert!(matches!(d.value(), Value::Table(_)));
    }

    #[test]
    fn passes_table1_through_when_table2_absent() {
        let mut meta = Meta::empty();
        meta.sfreq = Some(128.0);
        let t1 = table(&[("x", "7")], meta);
        let d = run(Some(t1), None).expect("emits");
        assert_eq!(as_map(&d), vec![("x".into(), "7".into())]);
        assert_eq!(d.meta().sfreq, Some(128.0), "passthrough carries its meta");
    }

    #[test]
    fn passes_table2_through_when_table1_absent() {
        let mut meta = Meta::empty();
        meta.sfreq = Some(64.0);
        let t2 = table(&[("y", "9")], meta);
        let d = run(None, Some(t2)).expect("emits");
        assert_eq!(as_map(&d), vec![("y".into(), "9".into())]);
        assert_eq!(d.meta().sfreq, Some(64.0));
    }

    #[test]
    fn no_emit_when_both_absent() {
        assert!(run(None, None).is_none());
    }

    #[test]
    fn wrong_typed_input_is_treated_as_absent() {
        // A STRING on table1 is not a table -> treated absent; table2 passes through.
        let d = run(Some(Data::string("nope", Meta::empty())), Some(table(&[("k", "v")], Meta::empty())))
            .expect("emits table2");
        assert_eq!(as_map(&d), vec![("k".into(), "v".into())]);
        // Both wrong-typed -> nothing present -> no emit.
        assert!(run(Some(Data::string("a", Meta::empty())), Some(Data::string("b", Meta::empty()))).is_none());
    }

    #[test]
    fn merges_meta_typed_field_and_open_map() {
        let mut m1 = Meta::empty();
        m1.sfreq = Some(100.0); // table2 leaves sfreq unset -> this must survive
        m1.extra.insert("src".into(), MetaValue::Str("one".into()));
        m1.extra.insert("only1".into(), MetaValue::Int(1));
        let mut m2 = Meta::empty();
        m2.extra.insert("src".into(), MetaValue::Str("two".into())); // overrides
        m2.extra.insert("only2".into(), MetaValue::Int(2));
        let d = run(Some(table(&[("a", "1")], m1)), Some(table(&[("b", "2")], m2))).expect("emits");
        assert_eq!(d.meta().sfreq, Some(100.0), "table1 sfreq survives (table2 unset)");
        assert_eq!(d.meta().extra.get("src"), Some(&MetaValue::Str("two".into())), "table2 wins");
        assert_eq!(d.meta().extra.get("only1"), Some(&MetaValue::Int(1)));
        assert_eq!(d.meta().extra.get("only2"), Some(&MetaValue::Int(2)));
    }

    #[test]
    fn table2_typed_field_overrides_when_set() {
        let mut m1 = Meta::empty();
        m1.sfreq = Some(100.0);
        let mut m2 = Meta::empty();
        m2.sfreq = Some(200.0);
        let d = run(Some(table(&[("a", "1")], m1)), Some(table(&[("b", "2")], m2))).expect("emits");
        assert_eq!(d.meta().sfreq, Some(200.0), "table2 sfreq wins when set");
    }

    #[test]
    fn channels_overlay_follows_presence() {
        let chan = |name: &str| {
            let mut ch = BTreeMap::new();
            ch.insert(0usize, Arc::new(vec![Coord::Str(name.into())]));
            Channels(ch)
        };
        // table2 has no channels -> table1's survive.
        let m1 = Meta { channels: chan("A"), ..Default::default() };
        let d = run(Some(table(&[("a", "1")], m1)), Some(table(&[("b", "2")], Meta::empty()))).expect("emits");
        assert!(d.meta().channels.0.contains_key(&0), "table1 channels kept when table2 has none");

        // Both have channels -> table2 replaces table1's wholesale.
        let m1 = Meta { channels: chan("A"), ..Default::default() };
        let m2 = Meta { channels: chan("Z"), ..Default::default() };
        let d = run(Some(table(&[("a", "1")], m1)), Some(table(&[("b", "2")], m2))).expect("emits");
        let coords = d.meta().channels.0.get(&0).expect("channels present");
        assert_eq!(coords.as_slice(), &[Coord::Str("Z".into())], "table2 channels replace table1's");
    }

    #[test]
    fn does_not_mutate_the_producer_tables() {
        // The merge clones table1's map; the original inputs stay untouched.
        let t1 = table(&[("a", "1")], Meta::empty());
        let t2 = table(&[("a", "2")], Meta::empty());
        let _ = run(Some(t1.clone()), Some(t2)).expect("emits");
        assert_eq!(as_map(&t1), vec![("a".into(), "1".into())], "table1 input unchanged");
    }
}
