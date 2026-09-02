//! A bound param as the node's thread holds it: the shared [`Expression`] plus what this engine
//! adds — the producer services it subscribes, and whether an arrival wakes `process()`.

use goofi_core::Data;
use goofi_node::{BindingId, Expression};

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
    pub fn new(source: impl Into<String>, vars: Vec<(VarName, Var)>, trigger: bool, id: Option<BindingId>) -> Binding {
        let streams = vars
            .iter()
            .filter_map(|(name, v)| match v {
                Var::Stream(service) => Some((name.clone(), service.clone())),
                _ => None,
            })
            .collect();
        Binding { expr: Expression::new(source, id, vars), trigger, streams }
    }

    /// Re-resolve this binding in place; a variable still on the same producer keeps what it holds.
    pub fn rebind(&mut self, next: Binding) {
        let previous = std::mem::replace(self, next);
        let streams = &self.streams;
        self.expr.carry(&previous.expr, |name| stream_of(&previous.streams, name) == stream_of(streams, name));
    }

    /// Land a producer's frame in the variable at `wire` — its position in [`Self::streams`].
    pub fn deliver(&mut self, wire: usize, frame: Data) {
        let Some((name, _)) = self.streams.get(wire) else { return };
        self.expr.deliver(name, frame);
    }
}
