//! FromJson — the first balanced object in a piece of text becomes a table. An object is a table,
//! a string a String, and everything else an Array.

use goofi_core::indexmap::IndexMap;
use goofi_core::{Data, Meta, SlotType};
use goofi_signal_sdk::{Inputs, Manifest, Node, NodeCtx, NodeResult, OutputDecl, Outputs, Params, SlotDecl, Tag};

#[derive(Default)]
struct FromJson;

enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct Reader {
    c: Vec<char>,
    at: usize,
}

impl Reader {
    fn space(&mut self) {
        while self.c.get(self.at).is_some_and(|c| c.is_whitespace()) {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Result<char, String> {
        self.space();
        self.c.get(self.at).copied().ok_or_else(|| "the text ends before the value does".into())
    }

    fn take(&mut self, want: char) -> Result<(), String> {
        if self.peek()? != want {
            return Err(format!("expected `{want}` at character {}", self.at));
        }
        self.at += 1;
        Ok(())
    }

    fn word(&mut self, want: &str) -> bool {
        if self.c[self.at..].starts_with(&want.chars().collect::<Vec<char>>()[..]) {
            self.at += want.chars().count();
            return true;
        }
        false
    }

    fn string(&mut self) -> Result<String, String> {
        self.take('"')?;
        let mut s = String::new();
        loop {
            let c = *self.c.get(self.at).ok_or("a string is never closed")?;
            self.at += 1;
            match c {
                '"' => return Ok(s),
                '\\' => {
                    let e = *self.c.get(self.at).ok_or("an escape is never finished")?;
                    self.at += 1;
                    s.push(match e {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'u' => {
                            let hex: String = self.c.get(self.at..self.at + 4).unwrap_or(&[]).iter().collect();
                            let code = u32::from_str_radix(&hex, 16)
                                .map_err(|_| format!("`\\u{hex}` is not four hex digits"))?;
                            self.at += 4;
                            char::from_u32(code).ok_or(format!("`\\u{hex}` is not a character"))?
                        }
                        other => other,
                    });
                }
                c => s.push(c),
            }
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        let start = self.at;
        while self.c.get(self.at).is_some_and(|c| "+-.eE0123456789".contains(*c)) {
            self.at += 1;
        }
        let text: String = self.c[start..self.at].iter().collect();
        text.parse().map_err(|_| format!("`{text}` is not a number"))
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek()? {
            '{' => {
                self.at += 1;
                let mut kv = Vec::new();
                if self.peek()? == '}' {
                    self.at += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    let key = self.string()?;
                    self.take(':')?;
                    kv.push((key, self.value()?));
                    match self.peek()? {
                        ',' => self.at += 1,
                        _ => break,
                    }
                }
                self.take('}')?;
                Ok(Json::Obj(kv))
            }
            '[' => {
                self.at += 1;
                let mut items = Vec::new();
                if self.peek()? == ']' {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    match self.peek()? {
                        ',' => self.at += 1,
                        _ => break,
                    }
                }
                self.take(']')?;
                Ok(Json::Arr(items))
            }
            '"' => Ok(Json::Str(self.string()?)),
            _ => {
                if self.word("true") {
                    Ok(Json::Bool(true))
                } else if self.word("false") {
                    Ok(Json::Bool(false))
                } else if self.word("null") {
                    Ok(Json::Null)
                } else {
                    Ok(Json::Num(self.number()?))
                }
            }
        }
    }
}

/// The shape this value would have as an array, when every leaf under it is a number.
fn shape(j: &Json) -> Option<Vec<usize>> {
    match j {
        Json::Null | Json::Bool(_) | Json::Num(_) => Some(Vec::new()),
        Json::Arr(items) => {
            let mut inner = if items.is_empty() { Some(Vec::new()) } else { shape(&items[0]) }?;
            if items.iter().any(|i| shape(i).as_ref() != Some(&inner)) {
                return None;
            }
            inner.insert(0, items.len());
            Some(inner)
        }
        _ => None,
    }
}

fn flatten(j: &Json, into: &mut Vec<u8>) {
    match j {
        Json::Null => into.extend_from_slice(&f32::NAN.to_le_bytes()),
        Json::Bool(b) => into.extend_from_slice(&(*b as u8 as f32).to_le_bytes()),
        Json::Num(v) => into.extend_from_slice(&(*v as f32).to_le_bytes()),
        Json::Arr(items) => items.iter().for_each(|i| flatten(i, into)),
        _ => {}
    }
}

fn frame(j: Json) -> Result<Data, String> {
    match j {
        Json::Str(s) => Ok(Data::string(s, Meta::new())),
        Json::Obj(kv) => {
            let mut map = IndexMap::new();
            for (k, v) in kv {
                map.insert(k, frame(v)?);
            }
            Ok(Data::table(map, Meta::new()))
        }
        other => match shape(&other) {
            Some(dims) => {
                let mut buf = Vec::new();
                flatten(&other, &mut buf);
                Data::array_f32(dims, buf, Meta::new()).map_err(|e| e.to_string())
            }
            // A list of strings, or a ragged one, is no array: it comes through keyed by position.
            None => {
                let Json::Arr(items) = other else { return Err("a value with no shape".into()) };
                let mut map = IndexMap::new();
                for (i, v) in items.into_iter().enumerate() {
                    map.insert(i.to_string(), frame(v)?);
                }
                Ok(Data::table(map, Meta::new()))
            }
        },
    }
}

impl Node for FromJson {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        _p: &Params<'_>,
    ) -> NodeResult {
        let text = inp.get("input").ok_or("`input` is required")?.as_str()?;
        let c: Vec<char> = text.chars().collect();
        let at = c.iter().position(|c| *c == '{').ok_or("the text holds no object")?;
        let value = Reader { c, at }.value()?;
        out.set("out", frame(value)?);
        Ok(())
    }
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "input",
    kind: SlotType::String,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl { name: "out", kind: SlotType::Table }];

static MANIFEST: Manifest = Manifest {
    tags: &[Tag::Transform, Tag::Text],
    doc: "The first object in a piece of text becomes a table.",
    inputs: INPUTS,
    outputs: OUTPUTS,
    params: &[],
    producer: false,
};

goofi_signal_sdk::export!(FromJson, MANIFEST);
