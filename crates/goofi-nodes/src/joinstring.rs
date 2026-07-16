//! JoinString — concatenate up to five STRING inputs with a configurable
//! separator. Each input has a per-slot enable toggle; a value is included only
//! when its toggle is on AND the slot is present. Ported from
//! `nodes/misc/joinstring.py`. Unlike the array nodes this reads/writes STRING,
//! so the input guard is on `Value::Str`, not a dtype.

use goofi_core::SlotType;
use goofi_core::{Data, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

/// The five input/toggle names, in join order. Slot names and the per-input bool
/// param names deliberately coincide (matches the Python `params.join[slot]`).
const NAMES: [&str; 5] = ["string1", "string2", "string3", "string4", "string5"];

struct JoinString {
    separator: String,
    enabled: [bool; 5],
}

impl Node for JoinString {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        // Collect the selected + present inputs in fixed string1..string5 order.
        // Faithful to the Python *code*: it filters only on (toggle && present) —
        // it does NOT drop empty-string values, despite the docstring's claim, so
        // neither do we.
        let mut parts: Vec<&str> = Vec::new();
        for (i, name) in NAMES.iter().enumerate() {
            if !self.enabled[i] {
                continue;
            }
            let Some(d) = inp.get(name) else {
                continue;
            };
            if let Value::Str(s) = d.value() {
                parts.push(s.as_ref());
            }
        }
        // With nothing selected/present the join is "" — Python returns "" here,
        // so emit the empty string rather than skipping the emit.
        let joined = parts.join(self.separator.as_str());
        out.set("output", Data::string(joined, Meta::empty()));
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group != "join" {
            return Ok(());
        }
        if key.name == "separator" {
            if let Some(s) = v.as_str() {
                self.separator = s.to_string();
            }
        } else if let Some(i) = NAMES.iter().position(|n| *n == key.name) {
            if let Some(b) = v.as_bool() {
                self.enabled[i] = b;
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("separator".to_string(), Param::str_free(", "));
    for name in NAMES {
        g.insert(name.to_string(), Param::boolean(true));
    }
    let mut groups = ParamGroups::new();
    groups.insert("join".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let separator = param(p, "join", "separator")
        .and_then(Param::as_str)
        .map(|s| s.to_string())
        .unwrap_or_else(|| ", ".to_string());
    let mut enabled = [true; 5];
    for (i, name) in NAMES.iter().enumerate() {
        enabled[i] = param(p, "join", name).and_then(Param::as_bool).unwrap_or(true);
    }
    Box::new(JoinString { separator, enabled })
}

static INPUTS: &[SlotDecl] = &[
    SlotDecl { name: "string1", kind: SlotType::String, trigger_process: true },
    SlotDecl { name: "string2", kind: SlotType::String, trigger_process: true },
    SlotDecl { name: "string3", kind: SlotType::String, trigger_process: true },
    SlotDecl { name: "string4", kind: SlotType::String, trigger_process: true },
    SlotDecl { name: "string5", kind: SlotType::String, trigger_process: true },
];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output",
    kind: SlotType::String,
}];

inventory::submit! {
    NodeManifest {
        type_name: "JoinString",
        category: "misc",
        doc: "Join up to five strings with a configurable separator (per-input enable toggles).",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    /// Drive the node once. `present[i]` supplies string(i+1) when `Some`; `None`
    /// leaves that slot absent. `toggles` overrides the default (all-on) enables.
    fn run(separator: Option<&str>, toggles: [bool; 5], present: [Option<&str>; 5]) -> String {
        let m = goofi_node::find("JoinString").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        if let Some(sep) = separator {
            node.on_param_changed(&ParamKey::new("join", "separator"), &Param::str_free(sep))
                .unwrap();
        }
        for (i, name) in super::NAMES.iter().enumerate() {
            node.on_param_changed(&ParamKey::new("join", *name), &Param::boolean(toggles[i]))
                .unwrap();
        }
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        for (i, name) in super::NAMES.iter().enumerate() {
            inmap.insert(name, present[i].map(|s| Data::string(s, Meta::empty())));
        }
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("output").unwrap().as_ref().unwrap();
        match d.value() {
            Value::Str(s) => s.to_string(),
            _ => panic!("expected string output"),
        }
    }

    #[test]
    fn joins_all_present_with_default_separator() {
        // Default separator is ", "; all five toggles default on.
        let out = run(None, [true; 5], [Some("a"), Some("b"), Some("c"), Some("d"), Some("e")]);
        assert_eq!(out, "a, b, c, d, e");
    }

    #[test]
    fn absent_slots_are_skipped_order_preserved() {
        // string2 and string4 absent -> joined in string1,3,5 order.
        let out = run(Some("-"), [true; 5], [Some("x"), None, Some("y"), None, Some("z")]);
        assert_eq!(out, "x-y-z");
    }

    #[test]
    fn disabled_toggle_excludes_even_when_present() {
        // string2 present but its toggle is off -> excluded.
        let out = run(Some("|"), [true, false, true, true, true], [Some("a"), Some("b"), Some("c"), None, None]);
        assert_eq!(out, "a|c");
    }

    #[test]
    fn empty_strings_are_included_not_dropped() {
        // The Python code filters only on toggle && present; empty values ride
        // through (the docstring's "non-empty" claim is not implemented). A single
        // present empty value yields "" with no separator.
        assert_eq!(run(Some(","), [true; 5], [Some(""), None, None, None, None]), "");
        // Two present, one empty -> a leading separator survives.
        assert_eq!(run(Some(","), [true; 5], [Some(""), Some("b"), None, None, None]), ",b");
    }

    #[test]
    fn nothing_selected_emits_empty_string() {
        // No inputs present at all still emits "" (faithful to Python's return "").
        let out = run(Some(", "), [true; 5], [None, None, None, None, None]);
        assert_eq!(out, "");
    }

    #[test]
    fn custom_separator_applied() {
        let out = run(Some(" :: "), [true; 5], [Some("one"), Some("two"), None, None, None]);
        assert_eq!(out, "one :: two");
    }

    #[test]
    fn single_present_value_has_no_separator() {
        let out = run(Some(", "), [true; 5], [None, None, Some("solo"), None, None]);
        assert_eq!(out, "solo");
    }
}
