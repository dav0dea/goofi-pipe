//! A bound param's latest-wins mailboxes (spec §5.3).
//!
//! The graph hands the node a REWRITTEN source plus one resolved [`Var`] per term, so the node
//! keeps a cell per variable and never resolves a name. Latest-wins is the whole storage model: a
//! node only ever wants the newest value a producer sent, so a write overwrites whatever was
//! unread rather than queueing behind it.

use goofi_core::{Data, Param};
use goofi_node::{BindingId, EvalCtx, ExprEvaluator, Local};
use indexmap::IndexMap;

use super::wire::{Var, VarName};

/// One variable's cell. It holds a [`Local`] rather than a [`Param`] because that is what the
/// evaluator's locals channel takes: a `globals.*` term resolves to a scalar, an `nd()` term to a
/// whole frame, and `__v0.mean()` is only expressible if the frame arrives as one.
#[derive(Clone, Debug, Default)]
pub struct Mailbox {
    value: Option<Local>,
    /// Why the graph could not resolve this variable, when it could not. Held on the mailbox
    /// rather than on the binding so an arrival clears it by construction — a value landing IS
    /// the resolution the graph could not make earlier.
    unresolved: Option<String>,
}

impl Mailbox {
    /// A variable awaiting its first arrival — a producer that has not emitted yet. Empty is not
    /// an error: the param's literal simply stands until something lands.
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

/// A bound param: the rewritten source plus a mailbox per variable it names.
#[derive(Clone, Debug)]
pub struct Binding {
    pub source: String,
    /// Whether an ARRIVAL on this binding also wakes `process()` (path C). The key's namespace,
    /// not this flag, decides whether it is read at all — see [`super::NodeRuntime`].
    pub trigger: bool,
    /// The evaluator's handle for [`Self::source`], compiled by the graph and shipped with the
    /// binding. `None` is not an error state — a bare-variable source needs no evaluator at all.
    pub id: Option<BindingId>,
    pub vars: IndexMap<VarName, Mailbox>,
    /// The producer services this binding subscribes to, in variable order — the set the node hands
    /// [`super::Transport::wire_in`] on its pseudo-slot. Kept because a frame arrives by wire INDEX
    /// and that index has to name a variable.
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

    /// Re-resolve this binding in place, keeping what each surviving variable already holds:
    /// §5.3's re-send "lands in the same mailbox", so a globals edit, a rename or a re-plan must
    /// not silence a producer whose frame this node is already holding. A variable the graph has
    /// since lost keeps its `Missing` state — the point of the re-send is that it is now broken.
    pub fn rebind(&mut self, next: Binding) {
        let previous = std::mem::replace(self, next);
        for (name, mailbox) in &mut self.vars {
            if mailbox.value().is_none() && mailbox.unresolved().is_none() {
                if let Some(held) = previous.vars.get(name).and_then(Mailbox::value) {
                    mailbox.put(held.clone());
                }
            }
        }
    }

    /// Land a producer's frame in the variable at `wire` — its position in [`Self::streams`], which
    /// is the order the services were handed to `wire_in` and therefore the order the transport
    /// numbers arrivals by.
    pub fn deliver(&mut self, wire: usize, frame: Data) {
        let Some((name, _)) = self.streams.get(wire) else { return };
        if let Some(mailbox) = self.vars.get_mut(name) {
            mailbox.put(Local::Frame(frame));
        }
    }

    /// The binding's current value: `Ok(None)` when nothing has arrived yet (the literal stands,
    /// and that is not an error), `Err` when it cannot be evaluated at all.
    ///
    /// A rewritten source that is exactly one variable holding a SCALAR — which is what §5.3's
    /// rewrite produces for `globals.default_ufreq` — IS that variable, and evaluates with no
    /// evaluator in the loop. Everything else goes to the evaluator with §5.3's locals channel: a
    /// frame has to be coerced to the target param's type, and only the evaluator knows how.
    pub fn evaluate(
        &self,
        evaluator: Option<&dyn ExprEvaluator>,
        t: f64,
        target: &Param,
    ) -> Result<Option<Param>, String> {
        if let Some(reason) = self.vars.values().find_map(Mailbox::unresolved) {
            return Err(reason.to_string());
        }
        if let Some(Local::Value(value)) = self.vars.get(self.source.trim()).and_then(Mailbox::value) {
            return Ok(Some(value.clone()));
        }
        // A variable that has not arrived is not an error and not a reason to evaluate: the
        // expression would see it as absent and the literal is the better answer (§2.1).
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

