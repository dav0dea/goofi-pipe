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

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::Meta;
    use std::sync::Mutex;

    fn stream(name: &str) -> Var {
        Var::Stream { name: name.to_string(), service: "svc".to_string(), event_id: 65 }
    }

    fn frame(v: f32) -> Data {
        Data::array_f32(vec![1], v.to_le_bytes().to_vec(), Meta::empty()).unwrap()
    }

    fn scalar(v: f64) -> Param {
        Param::float(v, 0.0, 1000.0)
    }

    /// An evaluator that answers with the SUM of its locals (a frame's first sample, a value's
    /// number) plus `t`. Deliberately not a constant: an oracle that cannot tell one locals map
    /// from another would pass for an `eval` handed the wrong variables, or none.
    #[derive(Default)]
    struct SumEval(Mutex<Vec<(BindingId, Vec<String>)>>);

    impl ExprEvaluator for SumEval {
        fn compile(&self, _source: &str) -> Result<goofi_node::Compiled, goofi_node::ExprError> {
            Ok(goofi_node::Compiled { id: 1 })
        }
        fn eval(&self, id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, goofi_node::ExprError> {
            let mut names: Vec<String> = ctx.locals.keys().cloned().collect();
            names.sort();
            self.0.lock().unwrap().push((id, names));
            let mut total = ctx.t;
            for local in ctx.locals.values().flatten() {
                total += match local {
                    Local::Frame(d) => match d.value() {
                        goofi_core::Value::Array(s) => {
                            f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()) as f64
                        }
                        _ => return Err("not an array".into()),
                    },
                    Local::Value(p) => p.as_f64().unwrap_or_default(),
                };
            }
            Ok(Param::float(total, 0.0, 1000.0))
        }
        fn release(&self, _id: BindingId) {}
    }

    #[test]
    fn a_mailbox_keeps_only_the_newest_value() {
        let mut m = Mailbox::empty();
        assert!(m.value().is_none());
        m.put(Local::Value(scalar(1.0)));
        m.put(Local::Value(scalar(2.0)));
        assert!(
            matches!(m.value(), Some(Local::Value(p)) if *p == scalar(2.0)),
            "the unread first value is gone",
        );
    }

    #[test]
    fn an_arrival_resolves_a_variable_the_graph_could_not() {
        // The producer the graph could not find at bind time exists now — nothing has to tell the
        // mailbox that separately, or a node keeps reporting an error it has the answer to.
        let mut m = Mailbox::missing("no node named `ghost`");
        assert_eq!(m.unresolved(), Some("no node named `ghost`"));
        m.put(Local::Frame(frame(1.0)));
        assert_eq!(m.unresolved(), None);
        assert!(m.value().is_some());
    }

    #[test]
    fn a_binding_evaluates_through_the_evaluator_and_reports_what_it_cannot() {
        let evaluator = SumEval::default();
        let mut b = Binding::new("__v0", vec![stream("__v0")], true, Some(1));
        assert_eq!(
            b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)),
            Ok(None),
            "nothing has arrived: the literal stands, no error",
        );
        b.deliver(0, frame(60.0));
        assert_eq!(b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)), Ok(Some(scalar(60.0))));

        // The frame reached the EVALUATOR under the name the graph minted — an assertion on the
        // number alone holds for an `eval` handed an empty locals map by a fixture that answers 60
        // regardless.
        assert_eq!(evaluator.0.lock().unwrap().as_slice(), [(1, vec!["__v0".to_string()])]);

        let missing = Binding::new(
            "__v0",
            vec![Var::Missing { name: "__v0".to_string(), reason: "no node named `ghost`".to_string() }],
            true,
            None,
        );
        assert_eq!(
            missing.evaluate(Some(&evaluator), 0.0, &scalar(0.0)),
            Err("no node named `ghost`".to_string()),
        );
    }

    #[test]
    fn a_rich_source_reaches_the_evaluator_and_says_so_without_one() {
        // `__v0.mean() * __v1` is exactly what the bare-variable shortcut cannot answer, and the
        // node-side handle is what the cutover added. Both halves in one test: with an evaluator it
        // COMPUTES, without one it refuses — pinning only the refusal would hold for a binding that
        // never reaches an evaluator it has.
        let evaluator = SumEval::default();
        let mut rich = Binding::new(
            "__v0.mean() * __v1",
            vec![
                stream("__v0"),
                Var::Value { name: "__v1".to_string(), value: scalar(10.0) },
            ],
            true,
            Some(7),
        );
        rich.deliver(0, frame(5.0));
        assert_eq!(rich.evaluate(Some(&evaluator), 2.0, &scalar(0.0)), Ok(Some(scalar(17.0))));
        assert_eq!(
            evaluator.0.lock().unwrap().as_slice(),
            [(7, vec!["__v0".to_string(), "__v1".to_string()])],
            "both variables travelled, under the names the rewrite minted",
        );
        assert!(rich.evaluate(None, 2.0, &scalar(0.0)).is_err(), "and without one it says so");
    }

    #[test]
    fn a_variable_less_source_is_evaluated_rather_than_skipped() {
        // A constant expression (`2 + 3`) names nothing, so the "has a variable that has not
        // arrived" guard must not swallow it — a binding that can never arrive would otherwise
        // never drive its param.
        let evaluator = SumEval::default();
        let b = Binding::new("2 + 3", vec![], true, Some(3));
        assert_eq!(b.evaluate(Some(&evaluator), 5.0, &scalar(0.0)), Ok(Some(scalar(5.0))));
    }

    #[test]
    fn a_rebind_lands_in_the_same_mailbox() {
        // A globals edit re-sends the whole binding. Rebuilding it from scratch would empty every
        // stream cell, so a node holding a producer's frame would fall back to its literal until
        // that producer emitted again — a re-plan silently un-driving a param.
        let evaluator = SumEval::default();
        let mut b = Binding::new("__v0", vec![stream("__v0")], true, Some(1));
        b.deliver(0, frame(7.0));
        b.rebind(Binding::new("__v0", vec![stream("__v0")], true, Some(1)));
        assert_eq!(
            b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)),
            Ok(Some(scalar(7.0))),
            "the held frame survived",
        );

        // A variable the graph has since lost is the exception: the re-send is what says so.
        b.rebind(Binding::new(
            "__v0",
            vec![Var::Missing { name: "__v0".to_string(), reason: "no node named `lfo`".to_string() }],
            true,
            None,
        ));
        assert_eq!(
            b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)),
            Err("no node named `lfo`".to_string()),
        );

        // And a re-sent value wins over the one it replaces.
        b.rebind(Binding::new(
            "__v0",
            vec![Var::Value { name: "__v0".to_string(), value: scalar(9.0) }],
            true,
            None,
        ));
        assert_eq!(b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)), Ok(Some(scalar(9.0))));
    }

    #[test]
    fn a_value_variable_arrives_already_seeded() {
        // A `globals.*` read is delivered inline with the binding, so the node evaluates it without
        // waiting for anything — which is what makes a globals edit re-pace a sleeping node. And it
        // needs NO evaluator: `common.max_frequency` is bound this way on every node in the patch.
        let b = Binding::new(
            "__v0",
            vec![Var::Value { name: "__v0".to_string(), value: scalar(30.0) }],
            true,
            None,
        );
        assert_eq!(b.evaluate(None, 0.0, &scalar(0.0)), Ok(Some(scalar(30.0))));
    }

    #[test]
    fn a_frame_lands_in_the_variable_its_wire_names() {
        // The wire INDEX is the whole addressing scheme now that a binding subscribes through
        // `wire_in` like any input slot. Delivering to the first mailbox regardless — which is what
        // this replaced — silently drives a two-reference expression from one producer.
        let evaluator = SumEval::default();
        let mut b = Binding::new(
            "__v0 + __v1",
            vec![stream("__v0"), stream("__v1")],
            true,
            Some(2),
        );
        b.deliver(1, frame(4.0));
        assert_eq!(b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)), Ok(None), "__v0 is still empty");
        b.deliver(0, frame(1.0));
        assert_eq!(b.evaluate(Some(&evaluator), 0.0, &scalar(0.0)), Ok(Some(scalar(5.0))));
    }
}
