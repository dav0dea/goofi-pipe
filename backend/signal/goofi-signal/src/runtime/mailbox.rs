//! A bound param's latest-wins mailboxes (spec §5.3).

use goofi_core::{Data, Param};
use goofi_node::{BindingId, EvalCtx, ExprEvaluator, Local};
use indexmap::IndexMap;

use super::wire::{Var, VarName};

/// One variable's cell. It holds a [`Local`] because that is what the evaluator's locals channel
/// takes: a `globals.*` term is a scalar, an `nd()` term a whole frame.
#[derive(Clone, Debug, Default)]
pub struct Mailbox {
    value: Option<Local>,
    /// Why the graph could not resolve this variable; an arrival clears it.
    unresolved: Option<String>,
}

impl Mailbox {
    /// A variable awaiting its first arrival. Empty is not an error: the literal stands.
    pub fn empty() -> Mailbox {
        Mailbox::default()
    }
    /// A variable the graph resolved and delivered inline (a `globals.*` read).
    pub fn seeded(value: Param) -> Mailbox {
        Mailbox { value: Some(Local::Value(value)), unresolved: None }
    }
    /// A variable the graph could not resolve.
    pub fn missing(reason: impl Into<String>) -> Mailbox {
        Mailbox { value: None, unresolved: Some(reason.into()) }
    }
    pub fn put(&mut self, value: Local) {
        self.value = Some(value);
        self.unresolved = None;
    }
    pub fn value(&self) -> Option<&Local> {
        self.value.as_ref()
    }
    pub fn unresolved(&self) -> Option<&str> {
        self.unresolved.as_deref()
    }
}

/// One frame as a scalar param of `target`'s type — what a reference copies on arrival.
fn scalar_of(frame: &Data, target: &Param) -> Result<Param, String> {
    match (frame.value(), target) {
        (goofi_core::Value::Str(s), Param::Str { options, refresh, .. }) => {
            Ok(Param::Str { value: s.to_string(), options: options.clone(), refresh: *refresh })
        }
        (goofi_core::Value::Array(a), _) if a.shape().iter().product::<usize>() == 1 => {
            let bytes: [u8; 4] = a.as_bytes()[..4].try_into().expect("one f32");
            let x = f32::from_le_bytes(bytes) as f64;
            Ok(match target {
                Param::Float { vmin, vmax, .. } => Param::Float { value: x, vmin: *vmin, vmax: *vmax },
                Param::Int { vmin, vmax, .. } => Param::Int { value: x.round() as i64, vmin: *vmin, vmax: *vmax },
                Param::Bool { .. } => Param::Bool { value: x >= 0.5 },
                Param::Str { .. } => return Err("a string param references a STRING output".to_string()),
            })
        }
        (goofi_core::Value::Array(a), _) => {
            Err(format!("a reference needs one element, and this frame is {:?}", a.shape()))
        }
        (goofi_core::Value::Str(_), _) => Err("a STRING output references a string param".to_string()),
        (goofi_core::Value::Table(_), _) => Err("a reference cannot follow a TABLE output".to_string()),
    }
}

fn stream_of<'a>(streams: &'a [(VarName, super::ServiceName)], name: &str) -> Option<&'a super::ServiceName> {
    streams.iter().find(|(n, _)| n == name).map(|(_, service)| service)
}

/// A bound param: the rewritten source plus a mailbox per variable it names.
#[derive(Clone, Debug)]
pub struct Binding {
    pub source: String,
    /// Whether an ARRIVAL on this binding also wakes `process()` (path C).
    pub trigger: bool,
    /// The evaluator's handle for [`Self::source`]; `None` for a bare-variable source, which
    /// needs no evaluator.
    pub id: Option<BindingId>,
    pub vars: IndexMap<VarName, Mailbox>,
    /// The producer services this binding subscribes to, in variable order — a frame arrives by
    /// wire INDEX, and that index has to name a variable.
    pub streams: Vec<(VarName, super::ServiceName)>,
}

impl Binding {
    pub fn new(source: impl Into<String>, vars: Vec<Var>, trigger: bool, id: Option<BindingId>) -> Binding {
        let streams = vars
            .iter()
            .filter_map(|v| match v {
                Var::Stream { name, service, .. } => Some((name.clone(), service.clone())),
                _ => None,
            })
            .collect();
        let vars = vars
            .into_iter()
            .map(|v| {
                let name = v.name().to_string();
                let mailbox = match v {
                    Var::Stream { .. } => Mailbox::empty(),
                    Var::Value { value, .. } => Mailbox::seeded(value),
                    Var::Missing { reason, .. } => Mailbox::missing(reason),
                };
                (name, mailbox)
            })
            .collect();
        Binding { source: source.into(), trigger, id, vars, streams }
    }

    /// Re-resolve this binding in place, keeping what each surviving variable already holds — from
    /// the SAME producer only: a variable re-pointed at another stream starts empty, or a silent
    /// producer would stand in for the one it replaced.
    pub fn rebind(&mut self, next: Binding) {
        let previous = std::mem::replace(self, next);
        let streams = &self.streams;
        for (name, mailbox) in &mut self.vars {
            if mailbox.value().is_some() || mailbox.unresolved().is_some() {
                continue;
            }
            if stream_of(&previous.streams, name) != stream_of(streams, name) {
                continue;
            }
            if let Some(held) = previous.vars.get(name).and_then(Mailbox::value) {
                mailbox.put(held.clone());
            }
        }
    }

    /// Land a producer's frame in the variable at `wire` — its position in [`Self::streams`].
    pub fn deliver(&mut self, wire: usize, frame: Data) {
        let Some((name, _)) = self.streams.get(wire) else { return };
        if let Some(mailbox) = self.vars.get_mut(name) {
            mailbox.put(Local::Frame(frame));
        }
    }

    /// The binding's current value: `Ok(None)` when nothing has arrived yet (the literal stands),
    /// `Err` when it cannot be evaluated at all.
    pub fn evaluate(
        &self,
        evaluator: Option<&dyn ExprEvaluator>,
        t: f64,
        target: &Param,
    ) -> Result<Option<Param>, String> {
        if let Some(reason) = self.vars.values().find_map(Mailbox::unresolved) {
            return Err(reason.to_string());
        }
        // A bare variable is read without the evaluator: a global's value as it is, and a
        // referenced producer's frame as the one element it must hold.
        match self.vars.get(self.source.trim()).and_then(Mailbox::value) {
            Some(Local::Value(value)) => return Ok(Some(value.clone())),
            Some(Local::Frame(frame)) if self.id.is_none() => return scalar_of(frame, target).map(Some),
            _ => {}
        }
        if self.vars.values().any(|m| m.value().is_none()) {
            return Ok(None);
        }
        let (Some(evaluator), Some(id)) = (evaluator, self.id) else {
            return Err(format!("`{}` needs the expression evaluator", self.source));
        };
        let locals = self.vars.iter().map(|(n, m)| (n.clone(), m.value().cloned())).collect();
        evaluator
            .eval(id, &EvalCtx { locals: &locals, t, target })
            .map(Some)
            .map_err(|e| e.0)
    }
}

