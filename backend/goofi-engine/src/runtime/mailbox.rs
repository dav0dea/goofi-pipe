//! A bound param's latest-wins mailboxes (spec §5.3).
//!
//! The graph hands the node a REWRITTEN source plus one resolved [`Var`] per term, so the node
//! keeps a cell per variable and never resolves a name. Latest-wins is the whole storage model: a
//! node only ever wants the newest value a producer sent, so a write overwrites whatever was
//! unread rather than queueing behind it.

use goofi_core::Param;
use indexmap::IndexMap;

use super::wire::{Var, VarName};

/// One variable's cell.
#[derive(Clone, Debug, Default)]
pub struct Mailbox {
    value: Option<Param>,
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
        Mailbox { value: Some(value), unresolved: None }
    }
    /// A variable the graph could not resolve.
    pub fn missing(reason: impl Into<String>) -> Mailbox {
        Mailbox { value: None, unresolved: Some(reason.into()) }
    }
    pub fn put(&mut self, value: Param) {
        self.value = Some(value);
        self.unresolved = None;
    }
    pub fn value(&self) -> Option<&Param> {
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
    pub vars: IndexMap<VarName, Mailbox>,
}

impl Binding {
    pub fn new(source: impl Into<String>, vars: Vec<Var>, trigger: bool) -> Binding {
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
        Binding { source: source.into(), trigger, vars }
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

    /// Land an arrival addressed to this BINDING rather than to one of its variables. A delivery
    /// is addressed by variable once the doorbell carries an `EventId` to map back; until then the
    /// distinction is nothing to the scheduling rule this seam exists for, which reads the param
    /// key's namespace either way.
    pub fn deliver(&mut self, value: Param) {
        if let Some(mailbox) = self.vars.values_mut().next() {
            mailbox.put(value);
        }
    }

    /// The binding's current value: `Ok(None)` when nothing has arrived yet (the literal stands,
    /// and that is not an error), `Err` when it cannot be evaluated at all.
    ///
    /// A rewritten source that is exactly one variable — which is what §5.3's rewrite produces for
    /// `globals.default_ufreq` and for a bare `nd('lfo')` — IS that variable, and evaluates with no
    /// evaluator in the loop. Anything richer needs the `ExprEvaluator` and §5.3's locals channel;
    /// saying so is better than quietly handing back the wrong term's value.
    pub fn evaluate(&self) -> Result<Option<Param>, String> {
        let Some(mailbox) = self.vars.get(self.source.trim()) else {
            return Err(format!(
                "`{}` needs the expression evaluator, which this node is not yet wired to",
                self.source
            ));
        };
        match mailbox.unresolved() {
            Some(reason) => Err(reason.to_string()),
            None => Ok(mailbox.value().cloned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(name: &str) -> Var {
        Var::Stream { name: name.to_string(), service: "svc".to_string(), event_id: 65 }
    }

    #[test]
    fn a_mailbox_keeps_only_the_newest_value() {
        let mut m = Mailbox::empty();
        assert!(m.value().is_none());
        m.put(Param::float(1.0, 0.0, 2.0));
        m.put(Param::float(2.0, 0.0, 2.0));
        assert_eq!(m.value(), Some(&Param::float(2.0, 0.0, 2.0)), "the unread first value is gone");
    }

    #[test]
    fn an_arrival_resolves_a_variable_the_graph_could_not() {
        // The producer the graph could not find at bind time exists now — nothing has to tell the
        // mailbox that separately, or a node keeps reporting an error it has the answer to.
        let mut m = Mailbox::missing("no node named `ghost`");
        assert_eq!(m.unresolved(), Some("no node named `ghost`"));
        m.put(Param::boolean(true));
        assert_eq!(m.unresolved(), None);
        assert_eq!(m.value(), Some(&Param::boolean(true)));
    }

    #[test]
    fn a_binding_evaluates_to_its_variable_and_reports_what_it_cannot() {
        let mut b = Binding::new("__v0", vec![stream("__v0")], true);
        assert_eq!(b.evaluate(), Ok(None), "nothing has arrived: the literal stands, no error");
        b.deliver(Param::float(60.0, 0.0, 100.0));
        assert_eq!(b.evaluate(), Ok(Some(Param::float(60.0, 0.0, 100.0))));

        let missing = Binding::new(
            "__v0",
            vec![Var::Missing { name: "__v0".to_string(), reason: "no node named `ghost`".to_string() }],
            true,
        );
        assert_eq!(missing.evaluate(), Err("no node named `ghost`".to_string()));

        let rich = Binding::new("__v0.mean() * __v1", vec![stream("__v0"), stream("__v1")], true);
        assert!(rich.evaluate().is_err(), "an expression that is not a bare variable says so");
    }

    #[test]
    fn a_rebind_lands_in_the_same_mailbox() {
        // A globals edit re-sends the whole binding. Rebuilding it from scratch would empty every
        // stream cell, so a node holding a producer's frame would fall back to its literal until
        // that producer emitted again — a re-plan silently un-driving a param.
        let mut b = Binding::new("__v0", vec![stream("__v0")], true);
        b.deliver(Param::float(7.0, 0.0, 10.0));
        b.rebind(Binding::new("__v0", vec![stream("__v0")], true));
        assert_eq!(b.evaluate(), Ok(Some(Param::float(7.0, 0.0, 10.0))), "the held frame survived");

        // A variable the graph has since lost is the exception: the re-send is what says so.
        b.rebind(Binding::new(
            "__v0",
            vec![Var::Missing { name: "__v0".to_string(), reason: "no node named `lfo`".to_string() }],
            true,
        ));
        assert_eq!(b.evaluate(), Err("no node named `lfo`".to_string()));

        // And a re-sent value wins over the one it replaces.
        b.rebind(Binding::new(
            "__v0",
            vec![Var::Value { name: "__v0".to_string(), value: Param::float(9.0, 0.0, 10.0) }],
            true,
        ));
        assert_eq!(b.evaluate(), Ok(Some(Param::float(9.0, 0.0, 10.0))));
    }

    #[test]
    fn a_value_variable_arrives_already_seeded() {
        // A `globals.*` read is delivered inline with the binding, so the node evaluates it without
        // waiting for anything — which is what makes a globals edit re-pace a sleeping node.
        let b = Binding::new("__v0", vec![Var::Value { name: "__v0".to_string(), value: Param::float(30.0, 0.0, 100.0) }], true);
        assert_eq!(b.evaluate(), Ok(Some(Param::float(30.0, 0.0, 100.0))));
    }
}
