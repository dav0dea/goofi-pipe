//! AppendTables — merge N TABLE inputs into one. Its single `tables` input is a
//! `multi` slot: an arbitrary number of wires arrive as an ordered `&[Data]` in
//! connection order, and are merged left-to-right so a later table overrides an
//! earlier one on a key collision (chained `{**t0, **t1, …}`); their `Meta`
//! sidecars fold the same way. A lone present table passes through untouched; none
//! present emits nothing. Reads/writes TABLE, so the guard is on `Value::Table` (a
//! wrong-typed frame counts as absent — the slot is TABLE-typed, purely defensive).
//! Generalizes the Python `nodes/misc/appendtables.py` (fixed two inputs) and the
//! fixed-arity ExtendedTable family via the multi-input primitive.

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
        // The `tables` multi slot delivers the present wires in connection order.
        // Only a Value::Table counts (a wrong-typed frame is defensively skipped).
        let tables: Vec<&Data> = inp
            .get_multi("tables")
            .iter()
            .filter(|d| matches!(d.value(), Value::Table(_)))
            .collect();
        match tables.as_slice() {
            // Nothing present -> no emit.
            [] => {}
            // A lone table passes through unchanged (its data + meta ride the Arc bump).
            [only] => out.set("output_table", (*only).clone()),
            // Merge left-to-right: insert every table's entries in order so a later
            // table overrides on a key collision while earlier keys keep their slot
            // (IndexMap == chained `{**t0, **t1, …}`); fold the meta the same way.
            [first, rest @ ..] => {
                let mut merged: IndexMap<String, Data> = match first.value() {
                    Value::Table(m) => (**m).clone(),
                    _ => unreachable!("filtered to tables"),
                };
                let mut meta = first.meta().clone();
                for d in rest {
                    if let Value::Table(m) = d.value() {
                        for (k, v) in m.iter() {
                            merged.insert(k.clone(), v.clone());
                        }
                        meta = merge_meta(&meta, d.meta());
                    }
                }
                out.set("output_table", Data::table(merged, meta));
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

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "tables",
    kind: SlotType::Table,
    trigger_process: true,
    multi: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output_table",
    kind: SlotType::Table,
}];

inventory::submit! {
    NodeManifest {
        type_name: "AppendTables",
        category: "misc",
        doc: "Merge N tables into one (a later table wins on key collision); pass a lone table through.",
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

    /// Drive AppendTables once with the `tables` multi slot fed the given ordered
    /// wire list; return the emitted `output_table` Data, or `None` when it no-ops.
    fn run(tables: Vec<Data>) -> Option<Data> {
        let m = goofi_node::find("AppendTables").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let singles: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        let mut multis: IndexMap<&'static str, Vec<Data>> = IndexMap::new();
        multis.insert("tables", tables);
        let inp = Inputs::with_multi(&singles, &multis);
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
    fn merges_disjoint_keys_in_connection_order() {
        let t1 = table(&[("a", "1"), ("b", "2")], Meta::empty());
        let t2 = table(&[("c", "3")], Meta::empty());
        let d = run(vec![t1, t2]).expect("emits");
        assert_eq!(
            as_map(&d),
            vec![("a".into(), "1".into()), ("b".into(), "2".into()), ("c".into(), "3".into())],
        );
    }

    #[test]
    fn merges_three_tables_a_later_wins_on_collision() {
        // Three wires in connection order; `b` appears in t1 and t3 -> t3 wins but
        // keeps t1's slot; new keys append in order (chained `{**t1, **t2, **t3}`).
        let t1 = table(&[("a", "1"), ("b", "2")], Meta::empty());
        let t2 = table(&[("c", "3")], Meta::empty());
        let t3 = table(&[("b", "99"), ("d", "4")], Meta::empty());
        let d = run(vec![t1, t2, t3]).expect("emits");
        assert_eq!(
            as_map(&d),
            vec![
                ("a".into(), "1".into()),
                ("b".into(), "99".into()),
                ("c".into(), "3".into()),
                ("d".into(), "4".into()),
            ],
        );
    }

    #[test]
    fn later_table_overrides_on_collision_keeping_earlier_position() {
        let t1 = table(&[("a", "1"), ("b", "2")], Meta::empty());
        let t2 = table(&[("b", "99"), ("c", "3")], Meta::empty());
        let d = run(vec![t1, t2]).expect("emits");
        assert_eq!(
            as_map(&d),
            vec![("a".into(), "1".into()), ("b".into(), "99".into()), ("c".into(), "3".into())],
        );
    }

    #[test]
    fn output_is_a_table() {
        let d = run(vec![table(&[("a", "1")], Meta::empty()), table(&[("b", "2")], Meta::empty())])
            .expect("emits");
        assert!(matches!(d.value(), Value::Table(_)));
    }

    #[test]
    fn passes_a_lone_table_through_with_its_meta() {
        let mut meta = Meta::empty();
        meta.sfreq = Some(128.0);
        let d = run(vec![table(&[("x", "7")], meta)]).expect("emits");
        assert_eq!(as_map(&d), vec![("x".into(), "7".into())]);
        assert_eq!(d.meta().sfreq, Some(128.0), "lone passthrough carries its meta");
    }

    #[test]
    fn no_emit_when_no_tables() {
        assert!(run(vec![]).is_none());
    }

    #[test]
    fn wrong_typed_wires_are_skipped() {
        // A STRING is not a table -> skipped; the lone table passes through.
        let d = run(vec![Data::string("nope", Meta::empty()), table(&[("k", "v")], Meta::empty())])
            .expect("emits the table");
        assert_eq!(as_map(&d), vec![("k".into(), "v".into())]);
        // All wrong-typed -> nothing present -> no emit.
        assert!(run(vec![Data::string("a", Meta::empty()), Data::string("b", Meta::empty())]).is_none());
    }

    #[test]
    fn merges_meta_typed_field_and_open_map() {
        let mut m1 = Meta::empty();
        m1.sfreq = Some(100.0); // t2 leaves sfreq unset -> this must survive
        m1.extra.insert("src".into(), MetaValue::Str("one".into()));
        m1.extra.insert("only1".into(), MetaValue::Int(1));
        let mut m2 = Meta::empty();
        m2.extra.insert("src".into(), MetaValue::Str("two".into())); // overrides
        m2.extra.insert("only2".into(), MetaValue::Int(2));
        let d = run(vec![table(&[("a", "1")], m1), table(&[("b", "2")], m2)]).expect("emits");
        assert_eq!(d.meta().sfreq, Some(100.0), "t1 sfreq survives (t2 unset)");
        assert_eq!(d.meta().extra.get("src"), Some(&MetaValue::Str("two".into())), "later wins");
        assert_eq!(d.meta().extra.get("only1"), Some(&MetaValue::Int(1)));
        assert_eq!(d.meta().extra.get("only2"), Some(&MetaValue::Int(2)));
    }

    #[test]
    fn later_typed_field_overrides_when_set() {
        let mut m1 = Meta::empty();
        m1.sfreq = Some(100.0);
        let mut m2 = Meta::empty();
        m2.sfreq = Some(200.0);
        let d = run(vec![table(&[("a", "1")], m1), table(&[("b", "2")], m2)]).expect("emits");
        assert_eq!(d.meta().sfreq, Some(200.0), "later sfreq wins when set");
    }

    #[test]
    fn channels_overlay_follows_presence() {
        let chan = |name: &str| {
            let mut ch = BTreeMap::new();
            ch.insert(0usize, Arc::new(vec![Coord::Str(name.into())]));
            Channels(ch)
        };
        // t2 has no channels -> t1's survive.
        let m1 = Meta { channels: chan("A"), ..Default::default() };
        let d = run(vec![table(&[("a", "1")], m1), table(&[("b", "2")], Meta::empty())]).expect("emits");
        assert!(d.meta().channels.0.contains_key(&0), "t1 channels kept when t2 has none");

        // Both have channels -> t2 replaces t1's wholesale.
        let m1 = Meta { channels: chan("A"), ..Default::default() };
        let m2 = Meta { channels: chan("Z"), ..Default::default() };
        let d = run(vec![table(&[("a", "1")], m1), table(&[("b", "2")], m2)]).expect("emits");
        let coords = d.meta().channels.0.get(&0).expect("channels present");
        assert_eq!(coords.as_slice(), &[Coord::Str("Z".into())], "later channels replace earlier");
    }

    #[test]
    fn does_not_mutate_the_producer_tables() {
        // The merge clones the first table's map; the original inputs stay untouched.
        let t1 = table(&[("a", "1")], Meta::empty());
        let t2 = table(&[("a", "2")], Meta::empty());
        let _ = run(vec![t1.clone(), t2]).expect("emits");
        assert_eq!(as_map(&t1), vec![("a".into(), "1".into())], "first input unchanged");
    }
}
