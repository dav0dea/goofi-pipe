//! The node contract: manifests, slot and param declarations, and the introspection probe's
//! schema — what a Python node file becomes on the way into the catalog.

use goofi_core::{Data, Meta, Param, SlotType};
use goofi_node::discover::*;
use goofi_node::*;
use indexmap::IndexMap;

#[test]
fn inputs_and_outputs_tick_io() {
    let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    inmap.insert("data", None);
    let inp = Inputs::new(&inmap);
    assert!(inp.get("data").is_none());
    assert!(inp.get("missing").is_none());

    let d = Data::array_f32(vec![1], 1.0f32.to_le_bytes().to_vec(), Meta::empty())
        .unwrap();
    let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    outmap.insert("out", None);
    {
        let mut out = Outputs::new(&mut outmap);
        out.set("out", d.clone());
        out.set("nonexistent", d); // writing an unknown slot is a no-op
    }
    assert!(outmap.get("out").unwrap().is_some());
}

#[test]
fn get_multi_returns_present_frames_in_connection_order() {
    fn mk(v: f32) -> Data {
        Data::array_f32(vec![1], v.to_le_bytes().to_vec(), Meta::empty()).unwrap()
    }
    fn val(d: &Data) -> f32 {
        match d.value() {
            goofi_core::Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
            _ => panic!(),
        }
    }
    let singles: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    let mut multis: IndexMap<&'static str, Vec<Data>> = IndexMap::new();
    multis.insert("ins", vec![mk(1.0), mk(2.0), mk(3.0)]);
    let inp = Inputs::with_multi(&singles, &multis);
    let got = inp.get_multi("ins");
    assert_eq!(got.len(), 3);
    assert_eq!([val(&got[0]), val(&got[1]), val(&got[2])], [1.0, 2.0, 3.0], "order preserved");
    assert_eq!(inp.get("ins").map(val), Some(1.0), "get() on a multi slot -> first present");
    assert!(inp.get_multi("absent").is_empty());

    // get_multi on a single slot is total: 0/1-element slice.
    let mut singles2: IndexMap<&'static str, Option<Data>> = IndexMap::new();
    singles2.insert("one", Some(mk(9.0)));
    singles2.insert("empty", None);
    let inp2 = Inputs::new(&singles2);
    assert_eq!(inp2.get_multi("one").len(), 1);
    assert_eq!(val(&inp2.get_multi("one")[0]), 9.0);
    assert!(inp2.get_multi("empty").is_empty());
    assert!(inp2.get_multi("missing").is_empty());
}

#[test]
fn with_common_materializes_the_documented_defaults_exactly() {
    // Every node in the system carries this group, and its values are persisted into every
    // `.gfi`. A bound or default edited here changes behavior everywhere, silently — so pin
    // the materialized values, not just their presence.
    let common = with_common(ParamGroups::new(), &probe_manifest(false));
    let c = common.get("common").expect("the group is always present");

    assert_eq!(c.get("autotrigger"), Some(&Param::boolean(false)));
    assert_eq!(c.get("max_frequency"), Some(&Param::float(0.0, 0.0, 100.0)));
    assert_eq!(
        c.get("frequency_mode"),
        Some(&Param::Str {
            value: FREQ_MODE_UPDATES_PER_SECOND.to_string(),
            options: Some(vec![
                FREQ_MODE_UPDATES_PER_SECOND.to_string(),
                FREQ_MODE_SECONDS_PER_UPDATE.to_string(),
            ]),
            refresh: false,
        })
    );
    assert_eq!(c.len(), 3, "no param silently joins the universal group");
    assert_eq!(common.keys().next().map(String::as_str), Some("common"), "placed first");
}

#[test]
fn producer_defaults_autotrigger_on_and_every_node_carries_the_ufreq_expression() {
    // `producer` is the ONLY pacing an author declares (spec §1.2), and a node that is not a
    // producer must not get its autotrigger.
    let p = with_common(ParamGroups::new(), &probe_manifest(true));
    assert_eq!(param(&p, "common", "autotrigger").and_then(Param::as_bool), Some(true));
    let c = with_common(ParamGroups::new(), &probe_manifest(false));
    assert_eq!(param(&c, "common", "autotrigger").and_then(Param::as_bool), Some(false));

    // The rate expression is carried by EVERY node, live only on a producer — `Off` is the
    // inspector's fx toggle waiting to be flipped, not a binding.
    let expr = |producer| {
        find_common(producer, "max_frequency").expression.expect("every node carries it")
    };
    assert_eq!(expr(false).source, "globals.default_ufreq");
    assert_eq!(expr(false).mode, ExprMode::Off, "carried, not imposed");
    // Declared uniformly, inert on `common.*` (spec §1.1) — pinned so the two producer
    // variants stay identical in it, not because pacing depends on it.
    assert!(expr(false).trigger, "declared for interface completeness, ignored on common.*");
    assert_eq!(expr(true).mode, ExprMode::On, "a producer IS paced by the patch rate");
    assert_eq!(expr(true).source, expr(false).source, "the producer variant rewrites only mode");
    assert_eq!(expr(true).trigger, expr(false).trigger);
}

/// A bare manifest that differs from its sibling only in `producer` — the input the universal
/// declarations are a function of.
fn probe_manifest(producer: bool) -> NodeManifest {
    NodeManifest {
        type_name: "_CommonDeclProbe",
        category: "test",
        doc: "",
        inputs: &[],
        outputs: NOP_OUT,
        params: NOP_PARAMS,
        isolation: Isolation::InProcess,
        producer,
        factory: default_factory::<Nop>,
    }
}

/// The one universal declaration named `name`, as a node of this kind sees it.
fn find_common(producer: bool, name: &str) -> ParamDecl {
    common_decls(&probe_manifest(producer)).find(|d| d.name == name).expect("declared")
}

#[test]
fn every_common_declaration_states_its_own_condition_on_the_manifest() {
    // Each universal param IS a function of the manifest, so a param is defined in one place
    // and states its own condition there — `with_common` needs no name match, and a fourth
    // param is added to COMMON_DECLS and nowhere else. All three are pinned, INCLUDING the one
    // that reads nothing: "no condition" is a claim, and it has to be checked like the rest.
    assert_eq!(
        common_decls(&probe_manifest(false)).count(),
        3,
        "no declaration silently joins the group"
    );

    // The spec DEFAULT moves, not just the materialized value, so that a consumer describing
    // the declaration and a consumer materializing it cannot disagree.
    let autotrigger = |p| match find_common(p, "autotrigger").spec {
        ParamSpec::Bool { default } => default,
        _ => panic!("autotrigger is a bool"),
    };
    assert!(!autotrigger(false));
    assert!(autotrigger(true), "a source paces itself");
    assert_eq!(
        with_common(ParamGroups::new(), &probe_manifest(true))["common"]["autotrigger"],
        Param::boolean(autotrigger(true)),
        "the value `with_common` materializes IS the declared default, for a producer too",
    );

    // frequency_mode reads nothing from the manifest — how to read the cap is the user's choice.
    let mode = |p| match find_common(p, "frequency_mode").spec {
        ParamSpec::Str { default, .. } => default,
        _ => panic!("frequency_mode is a string"),
    };
    assert_eq!(mode(true), mode(false), "being a source says nothing about this one");
    assert!(find_common(true, "frequency_mode").expression.is_none());
}

#[test]
fn with_common_keeps_a_param_the_node_declared_itself() {
    // A `common.*` key already present wins over BOTH the universal fallback and the producer
    // default — the latter matters on the restore path, where a user who turned a source's
    // autotrigger off must not have it turned back on by their own patch loading.
    let declared = |v: bool| {
        let mut groups = ParamGroups::new();
        let mut group = IndexMap::new();
        group.insert("autotrigger".to_string(), Param::boolean(v));
        groups.insert("common".to_string(), group);
        groups
    };

    let common = with_common(declared(true), &probe_manifest(false));
    assert_eq!(common["common"].get("autotrigger"), Some(&Param::boolean(true)));
    assert!(common["common"].contains_key("max_frequency"), "the rest is still filled in");

    let producer = with_common(declared(false), &probe_manifest(true));
    assert_eq!(producer["common"].get("autotrigger"), Some(&Param::boolean(false)));
}

static DECL_PARAMS: &[ParamDecl] = &[
    ParamDecl { group: "g", name: "freq", spec: ParamSpec::Float { default: 1.0, min: 0.0, max: 10.0 },
        expression: None, doc: None },
    ParamDecl { group: "g", name: "n", spec: ParamSpec::Int { default: 4, min: 1, max: 9 },
        expression: None, doc: None },
    ParamDecl { group: "g", name: "wave", spec: ParamSpec::Str { default: "sine", options: &["sine", "saw"], refresh: false },
        expression: None, doc: None },
    ParamDecl { group: "z", name: "on", spec: ParamSpec::Bool { default: true },
        expression: None, doc: None },
];

#[test]
fn params_from_decls_preserves_group_and_name_order_and_values() {
    let p = params_from_decls(DECL_PARAMS);
    // Group order = first-seen ("g" before "z"); name order = declaration order.
    assert_eq!(p.keys().collect::<Vec<_>>(), vec!["g", "z"]);
    assert_eq!(p["g"].keys().collect::<Vec<_>>(), vec!["freq", "n", "wave"]);
    assert_eq!(param(&p, "g", "freq").and_then(Param::as_f64), Some(1.0));
    assert_eq!(param(&p, "g", "n").and_then(Param::as_i64), Some(4));
    assert_eq!(param(&p, "z", "on").and_then(Param::as_bool), Some(true));
    // A Str with options materializes them; refresh flag carried through.
    match param(&p, "g", "wave").unwrap() {
        Param::Str { value, options, refresh } => {
            assert_eq!(value, "sine");
            assert_eq!(options.as_deref(), Some(&["sine".to_string(), "saw".to_string()][..]));
            assert!(!refresh);
        }
        _ => panic!("expected Str"),
    }
}

#[test]
fn params_view_reads_typed_values() {
    let p = params_from_decls(DECL_PARAMS);
    let view = Params::new(&p);
    assert_eq!(view.f64("g", "freq"), Some(1.0));
    assert_eq!(view.i64("g", "n"), Some(4));
    assert_eq!(view.str("g", "wave"), Some("sine"));
    assert_eq!(view.bool("z", "on"), Some(true));
    assert_eq!(view.f64("g", "missing"), None);
}

#[test]
fn manifest_default_params_builds_from_decls() {
    let m = find("_NodeTestNop").unwrap();
    assert!(m.default_params().is_empty(), "Nop declares no params");
}

#[test]
fn a_globals_read_needs_a_word_boundary_and_a_real_identifier() {
    // The other scanner's rule, and the three ways it says no. `globals.` with nothing after it
    // and `globals.1x` are not references — a digit-led name is not a Python identifier, so
    // reading one would mint a variable for a term the evaluator could never name.
    let names = |s| scan_globals(s).into_iter().map(|r| r.name.to_string()).collect::<Vec<_>>();
    assert_eq!(names("globals.default_ufreq * 2"), ["default_ufreq"]);
    assert!(names("myglobals.foo").is_empty(), "only the `globals` namespace matches");
    assert!(names("globals.").is_empty(), "a bare `globals.` is not a ref");
    assert!(names("globals.1x").is_empty(), "and neither is a digit-led name");
    assert_eq!(names("globals._x + globals.a1"), ["_x", "a1"], "but underscores and digits WITHIN one are fine");
}

#[test]
fn a_call_is_found_only_at_a_word_boundary() {
    // The rule BOTH consumers inherit — the rename rewriter and the expression rewrite. It is
    // pinned here, once, because that is the whole reason they share this scan.
    let names = |s| scan_nd_calls(s).into_iter().map(|c| c.name.to_string()).collect::<Vec<_>>();
    assert_eq!(names("nd('s').find('sub')"), ["s"], "only nd(), not .find()");
    assert!(names("round('x')").is_empty(), "round( is not nd(");
    assert!(names("grand('z')").is_empty(), "grand( is not nd(");
    assert_eq!(names("nd ('sig') * 2"), ["sig"], "whitespace before ( tolerated");
    assert!(names("t * 2").is_empty(), "no calls in a time expression");
}

#[test]
fn a_call_reports_both_the_name_span_and_the_term_span() {
    // The two consumers take different halves: a rename replaces `name_start..name_end`, the
    // expression rewrite replaces `start..end`. Spelled out as literal offsets rather than
    // asked of the scanner itself, which would agree with any answer it gave.
    let src = "x + nd( \"psd\" ).out";
    let calls = scan_nd_calls(src);
    assert_eq!(calls.len(), 1);
    assert_eq!(&src[calls[0].name_start..calls[0].name_end], "psd");
    assert_eq!(&src[calls[0].start..calls[0].end.unwrap()], "nd( \"psd\" )", "the whole call");

    // A call that does not close with a `)` has no term span — a rename still applies to it,
    // and the rewrite leaves it alone rather than spanning past the argument it cannot read.
    let extra = scan_nd_calls("nd('a', 2)");
    assert_eq!(extra[0].name, "a");
    assert_eq!(extra[0].end, None);
    assert_eq!(
        rewrite_nd_refs("nd('a', 2)", |n| (n == "a").then(|| "b".to_string())).unwrap(),
        "nd('b', 2)",
        "a rename does not depend on the term span",
    );
}

#[test]
fn rewrite_nd_refs_renames_only_matching_literals() {
    // Both single- and double-quoted refs to `lfo` follow; the quote style is kept;
    // a non-matching name and a look-alike token are left untouched.
    let src = "nd('lfo') + nd(\"lfo\").out - nd('psd') + grand('lfo')";
    let out = rewrite_nd_refs(src, |n| (n == "lfo").then(|| "osc".to_string())).unwrap();
    assert_eq!(out, "nd('osc') + nd(\"osc\").out - nd('psd') + grand('lfo')");
}

#[test]
fn rewrite_nd_refs_returns_none_when_nothing_changes() {
    assert!(rewrite_nd_refs("nd('psd') + t", |n| (n == "lfo").then(|| "osc".into())).is_none());
    assert!(rewrite_nd_refs("t * 2", |_| Some("x".to_string())).is_none(), "no nd() at all");
}

#[test]
fn param_lookup() {
    let mut g = IndexMap::new();
    g.insert("x".to_string(), Param::float(1.0, 0.0, 2.0));
    let mut groups: ParamGroups = IndexMap::new();
    groups.insert("grp".to_string(), g);
    assert_eq!(param(&groups, "grp", "x").and_then(Param::as_f64), Some(1.0));
    assert!(param(&groups, "grp", "missing").is_none());
    assert!(param(&groups, "nogroup", "x").is_none());
}

#[derive(Default)]
struct Nop;
impl Node for Nop {
    fn process(
        &mut self,
        _i: &Inputs<'_>,
        _o: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        _p: &Params<'_>,
    ) -> NodeResult {
        Ok(())
    }
}
static NOP_OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];
static NOP_PARAMS: &[ParamDecl] = &[];
inventory::submit! {
    NodeManifest {
        type_name: "_NodeTestNop",
        category: "test",
        doc: "",
        inputs: &[],
        outputs: NOP_OUT,
        params: NOP_PARAMS,
        isolation: Isolation::InProcess,
        producer: true,
        factory: default_factory::<Nop>,
    }
}

#[test]
fn run_policy_period_is_reciprocal_of_rate() {
    // Unbounded when max_frequency <= 0.
    assert_eq!(RunPolicy::default().period(), None);
    // `max_frequency` is always a Hz rate now: period = 1/f.
    let ups = RunPolicy { max_frequency: 4.0, ..Default::default() };
    assert_eq!(ups.period(), Some(0.25));
}

#[test]
fn run_policy_from_params_reads_common_group() {
    let policy = |freq: f64, mode: &str| {
        let mut common = IndexMap::new();
        common.insert("autotrigger".to_string(), Param::boolean(true));
        common.insert("max_frequency".to_string(), Param::float(freq, 0.0, 60.0));
        common.insert("frequency_mode".to_string(), Param::str_free(mode));
        let mut groups: ParamGroups = IndexMap::new();
        groups.insert("common".to_string(), common);
        RunPolicy::from_params(&groups)
    };
    // seconds-per-update is normalized to a Hz rate: a 30s period runs at 1/30 Hz,
    // i.e. `period()` still yields 30s.
    let spu = policy(30.0, "seconds-per-update");
    assert!(spu.autotrigger);
    assert_eq!(spu.period(), Some(30.0));
    // updates-per-second is taken verbatim as the rate.
    assert_eq!(policy(4.0, "updates-per-second").period(), Some(0.25));
    // A zero cap stays unbounded even in seconds-per-update mode (no 1/0).
    assert_eq!(policy(0.0, "seconds-per-update").period(), None);
    // A node with no `common` group defaults to triggered + unbounded.
    assert_eq!(RunPolicy::from_params(&ParamGroups::new()), RunPolicy::default());
}

#[test]
fn catalog_registration_and_output_buffer() {
    let m = find("_NodeTestNop").expect("registered via inventory");
    assert_eq!(m.outputs.len(), 1);
    assert_eq!(m.isolation, Isolation::InProcess);
    let buf = m.output_buffer();
    assert!(buf.contains_key("out"));
    assert!(catalog().any(|m| m.type_name == "_NodeTestNop"));
}

// ---------------------------------------------------------------------------
// The discovery probe
// ---------------------------------------------------------------------------


#[test]
fn camel_case_conversion() {
    assert_eq!(camel("double"), "Double");
    assert_eq!(camel("my_band_filter"), "MyBandFilter");
    assert_eq!(camel(""), "");
    assert_eq!(camel("__weird__name"), "WeirdName");
}

const SAMPLE: &str = r#"{"gil_safe":true,"doc":"PSD",
    "inputs":[{"name":"data","kind":"ARRAY","trigger":true,"multi":false}],
    "outputs":[{"name":"psd","kind":"ARRAY"}],
    "params":[{"group":"welch","name":"nperseg","kind":"int","default":256,"min":16,"max":4096},
              {"group":"welch","name":"tag","kind":"str","default":"a","options":["a","b"],"refresh":false}]}"#;

#[test]
fn parse_and_leak_builds_a_rich_manifest() {
    let intro = parse_introspection(SAMPLE).expect("parse");
    assert!(intro.gil_safe);
    let m = leak_manifest("Psd".into(), &intro, "python", Isolation::Subprocess);
    assert_eq!(m.type_name, "Psd");
    assert_eq!(m.doc, "PSD");
    assert_eq!(m.inputs.len(), 1);
    assert_eq!(m.inputs[0].name, "data");
    assert_eq!(m.inputs[0].kind, SlotType::Array);
    assert!(m.inputs[0].trigger_process);
    // SAMPLE carries neither a `required` nor a `producer` key — the shape a stale installed
    // `goofi` wheel emits. Both must default, not fail the whole introspection.
    assert!(!m.inputs[0].required);
    assert!(!m.producer);
    assert_eq!(m.outputs[0].name, "psd");
    assert_eq!(m.params.len(), 2);
    assert_eq!(m.params[0].group, "welch");
    assert_eq!(m.params[0].name, "nperseg");
    assert_eq!(m.params[1].name, "tag");
    // The int param carries its bounds; the str param carries its options.
    assert!(matches!(m.params[0].spec, crate::ParamSpec::Int { default: 256, min: 16, max: 4096 }));
    assert!(matches!(
        m.params[1].spec,
        crate::ParamSpec::Str { default: "a", options: [_, _], refresh: false }
    ));
}

#[test]
fn a_required_slot_crosses_the_probe() {
    // `goofi.InputSlot(..., required=True)` on a Python node must reach the manifest as a
    // required slot, or the engine never enforces the contract the author declared.
    const REQ: &str = r#"{"gil_safe":true,"doc":"Lz",
        "inputs":[{"name":"data","kind":"ARRAY","trigger":false,"multi":false,"required":true}],
        "outputs":[{"name":"out","kind":"ARRAY"}],"params":[]}"#;
    let intro = parse_introspection(REQ).expect("parse");
    let m = leak_manifest("Lz".into(), &intro, "python", Isolation::Subprocess);
    assert!(m.inputs[0].required);
    // `required` and `trigger` are independent (D2) — an authored `trigger=False` survives too.
    assert!(!m.inputs[0].trigger_process);
}

#[test]
fn a_refreshable_string_param_crosses_the_probe() {
    // `StringParam(..., refresh=True)` on a Python node must reach the manifest as a
    // refreshable spec, or the UI never renders the re-enumerate button for it.
    const PICKER: &str = r#"{"gil_safe":true,"doc":"Audio",
        "inputs":[],"outputs":[{"name":"out","kind":"ARRAY"}],
        "params":[{"group":"audio","name":"device","kind":"str","default":"none","options":[],"refresh":true}]}"#;
    let intro = parse_introspection(PICKER).expect("parse");
    let m = leak_manifest("Audio".into(), &intro, "python", Isolation::Subprocess);
    assert!(matches!(m.params[0].spec, crate::ParamSpec::Str { refresh: true, .. }));
}
