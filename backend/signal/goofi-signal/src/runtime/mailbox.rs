//! A bound param as the node's thread holds it: the shared [`Expression`] plus what this engine
//! adds — the producer services it subscribes, and whether an arrival wakes `process()`.

use goofi_core::{Data, Param};
use goofi_node::{BindingId, ExprEvaluator, Expression, Mailbox};

use super::wire::{Var, VarName};

fn stream_of<'a>(streams: &'a [(VarName, super::ServiceName)], name: &str) -> Option<&'a super::ServiceName> {
    streams.iter().find(|(n, _)| n == name).map(|(_, service)| service)
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub expr: Expression,
    /// Whether an ARRIVAL on this binding also wakes `process()` (path C).
    pub trigger: bool,
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
        Binding { expr: Expression { source: source.into(), id, vars }, trigger, streams }
    }

    /// Re-resolve this binding in place, keeping what each surviving variable already holds — from
    /// the SAME producer only: a variable re-pointed at another stream starts empty, or a silent
    /// producer would stand in for the one it replaced.
    pub fn rebind(&mut self, next: Binding) {
        let previous = std::mem::replace(self, next);
        let streams = &self.streams;
        for (name, mailbox) in &mut self.expr.vars {
            if mailbox.value().is_some() || mailbox.unresolved().is_some() {
                continue;
            }
            if stream_of(&previous.streams, name) != stream_of(streams, name) {
                continue;
            }
            if let Some(held) = previous.expr.vars.get(name).and_then(Mailbox::value) {
                mailbox.put(held.clone());
            }
        }
    }

    /// Land a producer's frame in the variable at `wire` — its position in [`Self::streams`].
    pub fn deliver(&mut self, wire: usize, frame: Data) {
        let Some((name, _)) = self.streams.get(wire) else { return };
        self.expr.deliver(name, frame);
    }

    pub fn evaluate(
        &self,
        evaluator: Option<&dyn ExprEvaluator>,
        t: f64,
        target: &Param,
    ) -> Result<Option<Param>, String> {
        self.expr.evaluate(evaluator, t, target)
    }
}
