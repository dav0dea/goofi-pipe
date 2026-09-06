//! ToJson — a table becomes JSON text. A table is an object, an array a list, a string a string.

use goofi_core::{Data, Meta, SlotType, Value};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, ParamDecl, Params, ParamSpec, SlotDecl, Tag};

#[derive(Default)]
struct ToJson;

fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn number(v: f32, decimals: i64, out: &mut String) {
    if !v.is_finite() {
        // JSON has no infinity and no NaN; null is what every reader takes for a missing number.
        out.push_str("null");
    } else if decimals < 0 {
        out.push_str(&format!("{v}"));
    } else {
        out.push_str(&format!("{v:.*}", decimals as usize));
    }
}

struct Writer {
    indent: usize,
    decimals: i64,
}

impl Writer {
    fn newline(&self, depth: usize, out: &mut String) {
        if self.indent > 0 {
            out.push('\n');
            out.push_str(&" ".repeat(self.indent * depth));
        }
    }

    /// The array as nested lists: one level of list per axis, in the shape's own order.
    fn rows(&self, shape: &[usize], values: &[f32], depth: usize, out: &mut String) {
        let Some((&span, rest)) = shape.split_first() else {
            return number(values[0], self.decimals, out);
        };
        let step = values.len().checked_div(span.max(1)).unwrap_or(0);
        out.push('[');
        for i in 0..span {
            if i > 0 {
                out.push(',');
                if self.indent == 0 {
                    out.push(' ');
                }
            }
            self.newline(depth + 1, out);
            self.rows(rest, &values[i * step..(i + 1) * step], depth + 1, out);
        }
        if span > 0 {
            self.newline(depth, out);
        }
        out.push(']');
    }

    fn value(&self, d: &Data, depth: usize, out: &mut String) {
        match d.value() {
            Value::Str(s) => quote(s, out),
            Value::Array(a) => {
                let values: Vec<f32> = a
                    .as_bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                    .collect();
                self.rows(a.shape(), &values, depth, out);
            }
            Value::Table(t) => {
                out.push('{');
                for (i, (k, v)) in t.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if self.indent == 0 {
                            out.push(' ');
                        }
                    }
                    self.newline(depth + 1, out);
                    quote(k, out);
                    out.push(':');
                    out.push(' ');
                    self.value(v, depth + 1, out);
                }
                if !t.is_empty() {
                    self.newline(depth, out);
                }
                out.push('}');
            }
        }
    }
}

impl Node for ToJson {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let d = inp.get("input").ok_or("`input` is required")?;
        d.as_table()?;
        let w = Writer {
            indent: p.i64("json", "indent").unwrap_or(0).clamp(0, 8) as usize,
            decimals: p.i64("json", "decimals").unwrap_or(-1).clamp(-1, 17),
        };
        let mut text = String::new();
        w.value(d, 0, &mut text);
        out.set("out", Data::string(text, Meta::new()));
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "json",
        name: "indent",
        spec: ParamSpec::Int { default: 0, min: 0, max: 8 },
        expression: None,
        doc: Some("Spaces of indent per level. 0 writes the whole table on one line."),
    },
    ParamDecl {
        group: "json",
        name: "decimals",
        spec: ParamSpec::Int { default: -1, min: -1, max: 17 },
        expression: None,
        doc: Some("Digits after the point on every number. -1 writes each one in full."),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::Table,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::String }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform, Tag::Text],
    doc: "A table becomes JSON text.\n\
          An object, with arrays as lists and strings as strings.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: PARAMS,
    producer: false,
};

goofi_signal_sdk::export!(ToJson, MANIFEST);
