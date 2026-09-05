//! Format — several strings become one: joined in wire order, or placed into a template that
//! says where each wire goes and how wide it sits.

use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct Format;

/// One field's `[[fill]align][width][.precision]`, applied to the text that fills it.
fn render(spec: &str, value: &str) -> String {
    let mut c: Vec<char> = spec.chars().collect();
    let (mut fill, mut align) = (' ', '<');
    if c.len() >= 2 && matches!(c[1], '<' | '>' | '^') {
        (fill, align) = (c[0], c[1]);
        c.drain(..2);
    } else if !c.is_empty() && matches!(c[0], '<' | '>' | '^') {
        align = c[0];
        c.drain(..1);
    }
    let rest: String = c.into_iter().collect();
    let (width, precision) = match rest.split_once('.') {
        Some((w, p)) => (w.parse().unwrap_or(0), p.parse::<usize>().ok()),
        None => (rest.parse().unwrap_or(0), None),
    };
    let mut s: String = match precision {
        Some(n) => value.chars().take(n).collect(),
        None => value.to_string(),
    };
    let held = s.chars().count();
    if held < width {
        let pad = width - held;
        let (left, right) = match align {
            '>' => (pad, 0),
            '^' => (pad / 2, pad - pad / 2),
            _ => (0, pad),
        };
        s = format!("{}{s}{}", String::from(fill).repeat(left), String::from(fill).repeat(right));
    }
    s
}

/// The template with each `{}` replaced by the wire it names. `{{` and `}}` are literal braces.
fn fill(template: &str, wires: &[&str]) -> Result<String, String> {
    let mut out = String::new();
    let mut next = 0usize;
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut body = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(c) => body.push(c),
                        None => return Err("a `{` in the template is never closed".into()),
                    }
                }
                let (index, spec) = body.split_once(':').unwrap_or((body.as_str(), ""));
                let at = if index.is_empty() {
                    next += 1;
                    next - 1
                } else {
                    index.parse().map_err(|_| format!("`{index}` is not a wire number"))?
                };
                out.push_str(&render(spec, wires.get(at).copied().unwrap_or("")));
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

impl Node for Format {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let wires = inp.get_multi("input");
        if wires.is_empty() {
            return Err("`input` needs at least one wire".to_string().into());
        }
        let mut parts = Vec::with_capacity(wires.len());
        for (source, d) in wires {
            parts.push(d.as_str().map_err(|e| format!("`{source}`: {e}"))?);
        }
        let text = if p.str("format", "mode").unwrap_or("concat") == "template" {
            fill(p.str("format", "template").unwrap_or("{}"), &parts)?
        } else {
            parts.join(p.str("format", "separator").unwrap_or(" "))
        };
        out.set("out", Data::string(text, Meta::new()));
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "format",
        name: "mode",
        spec: ParamSpec::Str { default: "concat", options: &["concat", "template"], refresh: false },
        expression: None,
        doc: Some("`concat` joins the wires in order; `template` places each one where it is named."),
    },
    ParamDecl {
        group: "format",
        name: "separator",
        spec: ParamSpec::Str { default: " ", options: &[], refresh: false },
        expression: None,
        doc: Some("What goes between two wires in `concat` mode."),
    },
    ParamDecl {
        group: "format",
        name: "template",
        spec: ParamSpec::Str { default: "{}", options: &[], refresh: false },
        expression: None,
        doc: Some(
            "Where each wire goes: `{}` takes the next one, `{2}` the third, and `{0:>8}` sets \
             it right in a field eight wide. A wire that is not there is empty.",
        ),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::String,
    trigger_process: true,
    multi: true,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::String }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform, Tag::Text],
    doc: "Several strings become one, joined in order or placed into a template.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(Format, MANIFEST);
