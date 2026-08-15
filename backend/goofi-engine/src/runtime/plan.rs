//! The graph-side wire planner (spec §4): one sequence per consumer slot, three phases, ordered by
//! acks.
//!
//! Attach, detach and replace are ONE operation, because a slot message carries the full desired set
//! and can add and remove at once. The order is what closes the attach window a history-less
//! transport leaves open:
//!
//! ```text
//! 1. producer-shrink   → OutSlot with removed targets gone          ack
//! 2. consumer-apply    → InSlot with the full new service set       ack
//! 3. producer-grow     → OutSlot with added targets present         ack
//! ```
//!
//! Removals leave the producer first, so no frame lands on a torn-down consumer; additions reach the
//! producer last, so it is never told to notify a subscriber that does not exist yet.
//!
//! **Supersede at the sequence level, not the message level.** A later change to a slot cancels the
//! in-flight sequence and starts again from phase 1 against the new desired set — superseding
//! individual messages would collapse phase 1 into phase 3 and destroy the ordering the phases exist
//! to establish. A cancelled sequence's ack is inert rather than an error: its seq is no longer
//! awaited, so answering it advances nothing. What the cancelled messages already SAID still stands,
//! because a slot message is declarative and its delivery never depended on the ack.
//!
//! What replans today is every link change ([`Graph::add_link`], [`Graph::remove_link`]) and a
//! channel being attached, which plans that node's slots from an empty base because it heard
//! nothing said before it arrived. The cutover adds the rest of §4's callers — node removal,
//! `restart_node`, a load, and an expression binding, which joins a producer's target set without
//! being a link at all.
//!
//! [`Graph::add_link`]: crate::Graph::add_link
//! [`Graph::remove_link`]: crate::Graph::remove_link

use std::collections::HashMap;
use std::sync::Arc;

use super::wire::{Control, ControlSink, Envelope};
use crate::Uid;

/// The producer end of a wire: a node and one of its output slots.
pub(crate) type Wire = (Uid, &'static str);

/// The consumer input slot a sequence is about.
pub(crate) type SlotKey = (Uid, &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Shrink,
    Apply,
    Grow,
}

/// One slot's in-flight wire change.
struct Sequence {
    /// Every producer that should feed this slot when the sequence completes, in wire order.
    desired: Vec<Wire>,
    /// The producers that lost this consumer, and the ones that gained it.
    removed: Vec<Wire>,
    added: Vec<Wire>,
    /// `None` until the first [`WirePlanner::step`] — a sequence begins before phase 1, not in it.
    phase: Option<Phase>,
}

impl Sequence {
    /// Whether the consumer has NOT yet acked a set from this sequence — it has been told nothing,
    /// or has been sent an `InSlot` it has not answered. Past `Apply` it holds `desired`.
    fn unapplied(&self) -> bool {
        matches!(self.phase, None | Some(Phase::Shrink) | Some(Phase::Apply))
    }
}

/// The graph's end of the wire plane: who to talk to, what each slot was last told, what is in
/// flight, and the birth generation of every uid this graph has ever held.
#[derive(Default)]
pub(crate) struct WirePlanner {
    /// One per live node. A uid with no channel is not addressable — its messages are dropped and
    /// never awaited, so a partially attached graph converges instead of stalling.
    sinks: HashMap<Uid, Arc<dyn ControlSink>>,
    /// Bumped on EVERY birth at a uid and never reset: it is what keeps a reborn node's service
    /// names clear of its predecessor's, whose teardown does not block (§3.1).
    generations: HashMap<Uid, u64>,
    sequences: HashMap<SlotKey, Sequence>,
    /// seq → the sequence waiting on it, and the ONLY record of what is outstanding: a phase is
    /// complete when no entry here still names its slot. That is also what makes a cancelled
    /// sequence's ack inert — cancelling drops its entries, so the late answer finds nothing to
    /// advance, a refusal included.
    awaiting: HashMap<u64, SlotKey>,
    /// What each slot was last PLANNED to hold — not what it is confirmed to hold. It is the base a
    /// shrink/grow diff is taken against, and it moves when the plan is made because a slot message
    /// is declarative: the node applies what it was sent whether or not its ack came back.
    planned: HashMap<SlotKey, Vec<Wire>>,
    next_seq: u64,
}

impl WirePlanner {
    /// Whether there is anyone to plan for. Nothing attaches a channel until the cutover, so this is
    /// what keeps the planner off the paths that call it — DELETE IT AT THE CUTOVER, when every live
    /// node has a channel: without it the planner is still correct (a sequence whose recipients have
    /// no sink simply walks to completion), it just does the work for nobody.
    pub(crate) fn is_idle(&self) -> bool {
        self.sinks.is_empty()
    }

    pub(crate) fn attach(&mut self, uid: Uid, sink: Arc<dyn ControlSink>) {
        self.sinks.insert(uid, sink);
    }

    /// The generation of the node about to be born at `uid`: 0 for a first birth, one more than the
    /// last for every rebirth.
    pub(crate) fn bump_generation(&mut self, uid: Uid) -> u64 {
        let next = self.generations.get(&uid).map_or(0, |g| g + 1);
        self.generations.insert(uid, next);
        next
    }

    pub(crate) fn generation(&self, uid: Uid) -> u64 {
        self.generations.get(&uid).copied().unwrap_or(0)
    }

    /// Drop every channel and everything in flight, keeping the generations: a `clear` destroys the
    /// nodes those channels addressed, and a channel held past its node's death would deliver one
    /// node's wiring to another born at the same uid.
    pub(crate) fn reset_channels(&mut self) {
        self.sinks.clear();
        self.sequences.clear();
        self.awaiting.clear();
        self.planned.clear();
    }

    /// Start a slot's sequence, cancelling whatever it had in flight.
    pub(crate) fn begin(&mut self, key: SlotKey, desired: Vec<Wire>, removed: Vec<Wire>, added: Vec<Wire>) {
        // A cancelled sequence that had not applied yet leaves its OWN additions unapplied, and the
        // diff base has already moved past them — so unless they are carried, the evidence that this
        // consumer never subscribed to them disappears with the sequence that held it, and some
        // other slot's phase 3 tells their producer to ring it. Carrying them also re-owes the grow
        // the cancelled sequence never sent. Rebuilt in `desired` order, so a phase's messages go
        // out in the order the set names them however they were merged.
        let carried = self
            .sequences
            .get(&key)
            .filter(|previous| previous.unapplied())
            .map(|previous| previous.added.clone())
            .unwrap_or_default();
        let added: Vec<Wire> =
            desired.iter().copied().filter(|w| added.contains(w) || carried.contains(w)).collect();
        self.abandon(key);
        self.planned.insert(key, desired.clone());
        self.sequences.insert(key, Sequence { desired, removed, added, phase: None });
    }

    /// Forget a slot's sequence and everything it was waiting on.
    fn abandon(&mut self, key: SlotKey) {
        self.sequences.remove(&key);
        self.awaiting.retain(|_, waiting| *waiting != key);
    }

    /// What this slot was last planned to hold — the set a change is diffed against.
    pub(crate) fn planned(&self, key: SlotKey) -> Vec<Wire> {
        self.planned.get(&key).cloned().unwrap_or_default()
    }

    /// Forget what a slot was planned to hold, so the next plan runs against nothing. What a node
    /// that has just become addressable is owed: it heard none of it.
    pub(crate) fn forget_planned(&mut self, key: SlotKey) {
        self.planned.remove(&key);
    }

    /// Whether `key`'s in-flight sequence is still ABOUT to subscribe `wire` — the consumer has been
    /// told nothing yet, or has been sent an `InSlot` it has not acked. A producer must not be told
    /// to ring it until then (§4).
    pub(crate) fn unapplied(&self, key: SlotKey, wire: Wire) -> bool {
        self.sequences.get(&key).is_some_and(|s| s.unapplied() && s.added.contains(&wire))
    }

    /// Move to the next phase, or finish the sequence and answer `None`.
    pub(crate) fn step(&mut self, key: SlotKey) -> Option<Phase> {
        let sequence = self.sequences.get_mut(&key)?;
        let next = match sequence.phase {
            None => Some(Phase::Shrink),
            Some(Phase::Shrink) => Some(Phase::Apply),
            Some(Phase::Apply) => Some(Phase::Grow),
            Some(Phase::Grow) => None,
        };
        sequence.phase = next;
        if next.is_none() {
            self.sequences.remove(&key);
        }
        next
    }

    /// The recipients of one phase: the producers that lost this consumer, or the ones that gained
    /// it. Phase 2 addresses the consumer itself, which the caller already knows.
    pub(crate) fn recipients(&self, key: SlotKey, phase: Phase) -> Vec<Wire> {
        let Some(sequence) = self.sequences.get(&key) else { return Vec::new() };
        match phase {
            Phase::Shrink => sequence.removed.clone(),
            Phase::Grow => sequence.added.clone(),
            Phase::Apply => Vec::new(),
        }
    }

    /// The full desired set of the sequence in flight on this slot.
    pub(crate) fn desired(&self, key: SlotKey) -> Vec<Wire> {
        self.sequences.get(&key).map(|s| s.desired.clone()).unwrap_or_default()
    }

    /// Send one phase's messages and start awaiting their acks. Answers whether anything is now
    /// awaited — a phase with nothing to say, or one every recipient of which is unaddressable, must
    /// not park the sequence on an ack that will never come.
    pub(crate) fn dispatch(&mut self, key: SlotKey, messages: Vec<(Uid, Control)>) -> bool {
        let mut waiting = false;
        for (uid, control) in messages {
            let Some(sink) = self.sinks.get(&uid).cloned() else { continue };
            self.next_seq += 1;
            let seq = self.next_seq;
            self.awaiting.insert(seq, key);
            waiting = true;
            sink.send(Envelope { seq, control });
        }
        waiting
    }

    /// Answer one ack. `Some(key)` when it completed a phase and the sequence is ready to advance;
    /// `None` when the phase is still outstanding, when the ack is a cancelled sequence's, or when
    /// the node refused — a refusal abandons the sequence rather than leaving it half applied, since
    /// there is no retry and `restart_node` is the recovery door (§4).
    pub(crate) fn ack(&mut self, seq: u64, ok: Result<(), String>) -> Option<SlotKey> {
        let key = self.awaiting.remove(&seq)?;
        if ok.is_err() {
            self.abandon(key);
            return None;
        }
        (!self.awaiting.values().any(|waiting| *waiting == key)).then_some(key)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::runtime::EventId;
    use crate::Graph;

    /// Every message the graph sent, in order, whoever it was for. A shared log rather than one
    /// recorder per node, because the ordering ACROSS nodes is the whole subject: phase 1 leaves a
    /// producer before phase 2 reaches the consumer.
    #[derive(Default)]
    struct Log(Mutex<Vec<(Uid, Envelope)>>);

    struct Recorder {
        uid: Uid,
        log: Arc<Log>,
    }

    impl ControlSink for Recorder {
        fn send(&self, envelope: Envelope) {
            self.log.0.lock().unwrap().push((self.uid, envelope));
        }
    }

    impl Log {
        /// Take what has been sent since the last look.
        fn take(&self) -> Vec<(Uid, Envelope)> {
            std::mem::take(&mut self.0.lock().unwrap())
        }
    }

    fn attach(g: &mut Graph, uids: &[Uid]) -> Arc<Log> {
        let log = Arc::new(Log::default());
        attach_to(g, &log, uids);
        log
    }

    /// Attach more nodes to a log that is already recording — a node becoming addressable LATER is
    /// its own case, and it has to be visible in the same order as everything else.
    fn attach_to(g: &mut Graph, log: &Arc<Log>, uids: &[Uid]) {
        for uid in uids {
            g.attach_control_sink(*uid, Arc::new(Recorder { uid: *uid, log: log.clone() }));
        }
    }

    fn source(g: &mut Graph) -> Uid {
        g.add_node("_TestGated", None).unwrap()
    }

    fn sink(g: &mut Graph) -> Uid {
        g.add_node("_TestSink", None).unwrap()
    }

    /// One message, projected to everything a test can assert about it: who it went to, which slot
    /// it is about, and the payload. Nothing is dropped — a projection that discards the slot name
    /// or the event id is a defect's hiding place, and both have been one.
    #[derive(Debug, PartialEq)]
    enum Sent {
        In { to: Uid, slot: String, services: Vec<String> },
        Out { to: Uid, slot: String, targets: Vec<(String, EventId)> },
        Param { to: Uid },
    }

    fn sent(batch: &[(Uid, Envelope)]) -> Vec<Sent> {
        batch
            .iter()
            .map(|(to, e)| match &e.control {
                Control::InSlot { slot, services } => Sent::In {
                    to: *to,
                    slot: slot.clone(),
                    services: services.iter().map(|s| scoped(s)).collect(),
                },
                Control::OutSlot { slot, targets } => Sent::Out {
                    to: *to,
                    slot: slot.clone(),
                    targets: targets.iter().map(|(door, id)| (scoped(door), *id)).collect(),
                },
                Control::SetParam { .. } => Sent::Param { to: *to },
            })
            .collect()
    }

    /// A service name with the graph's random scope removed, leaving `<uid>_<gen>_<what>` — the one
    /// part a test cannot know, dropped so the rest can be spelled out rather than asked of the same
    /// function that built it. `goofi_<instance>_` is exactly two underscore-separated fields,
    /// because a scope carries no underscore of its own.
    fn scoped(service: &str) -> String {
        service.splitn(3, '_').nth(2).unwrap_or(service).to_string()
    }

    /// What the graph must have named for `(uid, slot)` at generation `gen` — §3.1's format, written
    /// out here so a wrong uid, a wrong slot or a dropped generation is visible.
    fn out_name(uid: Uid, gen: u64, slot: &str) -> String {
        format!("{}_{gen}_out_{slot}", uid.to_hex())
    }

    fn door_name(uid: Uid, gen: u64) -> String {
        format!("{}_{gen}_door", uid.to_hex())
    }

    fn ack_all(batch: &[(Uid, Envelope)], g: &mut Graph) {
        for (_, e) in batch {
            g.wire_ack(e.seq, Ok(()));
        }
    }

    /// Answer everything outstanding, the way live nodes would — so a test can start from a settled
    /// wire rather than from a half-applied one.
    fn settle(log: &Log, g: &mut Graph) {
        loop {
            let batch = log.take();
            if batch.is_empty() {
                return;
            }
            ack_all(&batch, g);
        }
    }

    #[test]
    fn a_wire_change_removes_before_it_applies_and_adds_after() {
        // spec §4: one sequence for attach, detach and replace, because a declarative set can add
        // and remove at once. Removals leave the producer FIRST so no frame lands on a torn-down
        // consumer; additions reach the producer LAST so it is never told to notify a subscriber
        // that does not exist yet.
        let mut g = Graph::new();
        let (a, b, c) = (source(&mut g), source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, b, c]);
        g.add_link(a, "out", c, "in").unwrap();
        settle(&log, &mut g);

        g.add_link(b, "out", c, "in").unwrap(); // displaces a -> c on a SINGLE input

        let phase1 = log.take();
        assert_eq!(
            sent(&phase1),
            [Sent::Out { to: a, slot: "out".to_string(), targets: Vec::new() }],
            "1. producer-shrink: a has lost its only consumer"
        );
        ack_all(&phase1, &mut g);

        let phase2 = log.take();
        assert_eq!(
            sent(&phase2),
            [Sent::In {
                to: c,
                slot: "in".to_string(),
                // The displaced wire needs no special case — the new set is simply [b].
                services: vec![out_name(b, 0, "out")],
            }],
            "2. consumer-apply, and only after the shrink"
        );
        ack_all(&phase2, &mut g);

        let phase3 = log.take();
        assert_eq!(
            sent(&phase3),
            [Sent::Out {
                to: b,
                slot: "out".to_string(),
                // §3.2: an input slot's event id is its position in the manifest, past the control
                // id — `_TestSink` declares `in` first, so this is 1 and never 0.
                targets: vec![(door_name(c, 0), 1)],
            }],
            "3. producer-grow, and only after the apply"
        );
        ack_all(&phase3, &mut g);
        assert!(log.take().is_empty(), "and the sequence is over");
    }

    #[test]
    fn a_later_change_restarts_the_sequence_rather_than_superseding_a_message() {
        // Superseding individual messages would collapse phase 1 into phase 3 and destroy the
        // ordering the phases exist to establish.
        let mut g = Graph::new();
        let (a, b, c) = (source(&mut g), source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, b, c]);
        g.add_link(a, "out", c, "in").unwrap();
        settle(&log, &mut g);

        g.add_link(b, "out", c, "in").unwrap(); // change 1, left unanswered in phase 1
        let stale = log.take();
        assert_eq!(sent(&stale), [Sent::Out { to: a, slot: "out".to_string(), targets: Vec::new() }]);

        g.add_link(a, "out", c, "in").unwrap(); // change 2, on the same slot
        let restarted = log.take();
        assert_eq!(
            sent(&restarted),
            [Sent::Out { to: b, slot: "out".to_string(), targets: Vec::new() }],
            "phase 1 again, against the NEW desired set"
        );

        // The cancelled sequence's ack is inert — a REFUSED one included, which would otherwise
        // abandon the live sequence that replaced it and leave the slot wired to nothing.
        g.wire_ack(stale[0].1.seq, Err("superseded".to_string()));
        assert!(log.take().is_empty(), "the cancelled sequence's ack advances nothing");

        g.wire_ack(restarted[0].1.seq, Ok(()));
        assert_eq!(
            sent(&log.take()),
            [Sent::In { to: c, slot: "in".to_string(), services: vec![out_name(a, 0, "out")] }],
            "and the live sequence still advances on ITS ack, against the new set"
        );
    }

    #[test]
    fn a_phase_with_nothing_to_say_is_skipped_rather_than_sent() {
        // A pure add runs phases 2–3, a pure removal 1–2. An empty phase sent anyway would tell a
        // producer that lost nothing to re-declare its targets, and — worse — park the sequence on
        // an ack for a message that says nothing.
        let mut g = Graph::new();
        let (a, c) = (source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, c]);

        g.add_link(a, "out", c, "in").unwrap();
        let first = log.take();
        assert_eq!(
            sent(&first),
            [Sent::In { to: c, slot: "in".to_string(), services: vec![out_name(a, 0, "out")] }],
            "a pure add opens at phase 2"
        );
        ack_all(&first, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out { to: a, slot: "out".to_string(), targets: vec![(door_name(c, 0), 1)] }],
            "then phase 3"
        );

        settle(&log, &mut g);
        g.remove_link(a, "out", c, "in").unwrap();
        let shrink = log.take();
        assert_eq!(
            sent(&shrink),
            [Sent::Out { to: a, slot: "out".to_string(), targets: Vec::new() }],
            "a pure removal opens at phase 1"
        );
        ack_all(&shrink, &mut g);
        let apply = log.take();
        assert_eq!(
            sent(&apply),
            [Sent::In { to: c, slot: "in".to_string(), services: Vec::new() }],
            "an empty set means disconnected"
        );
        ack_all(&apply, &mut g);
        assert!(log.take().is_empty(), "and there is no phase 3 to run");
    }

    #[test]
    fn a_refused_message_abandons_its_sequence() {
        // There is no retry (§4): a node that refuses is not in sync with the graph, and driving the
        // remaining phases at it would tell a producer to notify a consumer that never applied its
        // set. `restart_node` is the recovery door.
        let mut g = Graph::new();
        let (a, c) = (source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, c]);

        g.add_link(a, "out", c, "in").unwrap();
        let apply = log.take();
        assert_eq!(
            sent(&apply),
            [Sent::In { to: c, slot: "in".to_string(), services: vec![out_name(a, 0, "out")] }]
        );
        g.wire_ack(apply[0].1.seq, Err("no input slot `in`".to_string()));
        assert!(log.take().is_empty(), "the grow never runs");
    }

    #[test]
    fn a_generation_is_bumped_at_every_birth_and_survives_a_clear() {
        // The counter is runtime bookkeeping, not patch content: `load_doc` restores the uids the
        // patch was saved with, so loading the same `.gfi` twice would otherwise mint the identical
        // service names — while the previous generation's ports are still being torn down.
        //
        // Every birth means every DOOR into one: §3.1 names restart first, and it is the one that
        // does not go through `insert_node_at`.
        let mut g = Graph::new();
        let uid = source(&mut g);
        assert_eq!(g.node_generation(uid), 0);

        g.restart_node(uid).unwrap();
        assert_eq!(g.node_generation(uid), 1, "a restart is a birth at the same uid");

        g.remove_node(uid).unwrap();
        g.add_node_at("_TestGated", None, uid, "").unwrap();
        assert_eq!(g.node_generation(uid), 2, "and so is an undo of a delete");

        g.clear();
        g.add_node_at("_TestGated", None, uid, "").unwrap();
        assert_eq!(g.node_generation(uid), 3, "a load does not put the counter back");
    }

    #[test]
    fn a_rebirth_is_named_apart_from_the_generation_it_replaces() {
        // What the counter is FOR: the name a consumer is given must change, or the reborn node
        // re-opens ports its predecessor — whose teardown does not block — still holds. Spelled out
        // rather than compared against `output_service_of` itself, which answers the same either way.
        let mut g = Graph::new();
        let uid = source(&mut g);
        let born = g.output_service_of(uid, "out");
        assert!(born.ends_with(&format!("_{}_0_out_out", uid.to_hex())), "{born}");
        assert!(g.door_of(uid).ends_with(&format!("_{}_0_door", uid.to_hex())), "{}", g.door_of(uid));

        g.restart_node(uid).unwrap();
        let reborn = g.output_service_of(uid, "out");
        assert!(reborn.ends_with(&format!("_{}_1_out_out", uid.to_hex())), "{reborn}");
        assert_ne!(reborn, born);
    }

    #[test]
    fn a_multi_input_names_every_wire_in_connection_order() {
        // §3.5: a multi slot keeps one subscriber per wire, "ordered by that service's position in
        // the received `services` Vec" — which is where `Inputs::get_multi`'s connection-order
        // contract comes from. A suite of single inputs can never see an order at all.
        let mut g = Graph::new();
        let (a, b) = (source(&mut g), source(&mut g));
        let c = g.add_node("_TestCollect", None).unwrap();
        let log = attach(&mut g, &[a, b, c]);

        g.add_link(a, "out", c, "ins").unwrap();
        settle(&log, &mut g);
        g.add_link(b, "out", c, "ins").unwrap(); // appended — a multi slot displaces nothing

        let apply = log.take();
        assert_eq!(
            sent(&apply),
            [Sent::In {
                to: c,
                slot: "ins".to_string(),
                services: vec![out_name(a, 0, "out"), out_name(b, 0, "out")],
            }],
            "both wires, in the order they were added"
        );
        ack_all(&apply, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out { to: b, slot: "out".to_string(), targets: vec![(door_name(c, 0), 1)] }],
            "and only the producer that was added is grown"
        );
    }

    #[test]
    fn a_producer_is_not_told_to_ring_a_consumer_that_has_not_subscribed() {
        // §4's guarantee is per TARGET: `a` feeds `c` and `d`, and `d`'s sequence reaches phase 3
        // while `c`'s InSlot is still unanswered. Naming `c` there tells `a` to ring a subscriber
        // that does not exist yet, and with `history_size 0` a frame published in that window is
        // simply lost. `c`'s own phase 3 names it, against the same live target set.
        let mut g = Graph::new();
        let (a, c, d) = (source(&mut g), sink(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, c, d]);

        g.add_link(a, "out", c, "in").unwrap(); // c's phase 2, left unanswered
        let c_apply = log.take();
        assert_eq!(sent(&c_apply).len(), 1);

        g.add_link(a, "out", d, "in").unwrap();
        let d_apply = log.take();
        ack_all(&d_apply, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out { to: a, slot: "out".to_string(), targets: vec![(door_name(d, 0), 1)] }],
            "d only: c has not applied its InSlot"
        );

        ack_all(&c_apply, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out {
                to: a,
                slot: "out".to_string(),
                targets: vec![(door_name(c, 0), 1), (door_name(d, 0), 1)],
            }],
            "and c's own phase 3 names them both"
        );
    }

    #[test]
    fn a_superseded_sequence_leaves_its_own_additions_unapplied() {
        // The residual of the rule above: `unapplied` reads the LIVE sequence, and a supersede
        // recomputes `added` against a diff base that has already moved — so the cancelled
        // sequence's own additions would stop counting as unapplied the moment it was replaced,
        // even though the consumer never heard a set naming them. `d`'s phase 3 would then tell `a`
        // to ring `c`, which is the very thing the phases are ordered to prevent.
        let mut g = Graph::new();
        let (a, b) = (source(&mut g), source(&mut g));
        let c = g.add_node("_TestCollect", None).unwrap();
        let d = sink(&mut g);
        let log = attach(&mut g, &[a, b, c, d]);

        g.add_link(a, "out", c, "ins").unwrap(); // c's phase 2, left unanswered
        assert_eq!(sent(&log.take()).len(), 1);
        g.add_link(b, "out", c, "ins").unwrap(); // and superseded before it could be
        let restarted = log.take();
        assert_eq!(
            sent(&restarted),
            [Sent::In {
                to: c,
                slot: "ins".to_string(),
                services: vec![out_name(a, 0, "out"), out_name(b, 0, "out")],
            }],
            "the new set names both, and c has still acked neither"
        );

        g.add_link(a, "out", d, "in").unwrap();
        let d_apply = log.take();
        ack_all(&d_apply, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out { to: a, slot: "out".to_string(), targets: vec![(door_name(d, 0), 1)] }],
            "d only — c has never applied a set naming a, whichever sequence would have said so"
        );

        ack_all(&restarted, &mut g);
        assert_eq!(
            sent(&log.take()),
            [
                Sent::Out {
                    to: a,
                    slot: "out".to_string(),
                    targets: vec![(door_name(c, 0), 1), (door_name(d, 0), 1)],
                },
                Sent::Out { to: b, slot: "out".to_string(), targets: vec![(door_name(c, 0), 1)] },
            ],
            "and now both producers are grown: the cancelled sequence's grow was owed too"
        );
    }

    #[test]
    fn a_node_that_could_not_be_reached_is_wired_when_it_arrives() {
        // `dispatch` skips a uid with no channel so a partially attached graph converges instead of
        // stalling — but the diff base moves all the same, so without a re-plan on attach nothing
        // would ever resend what was dropped. A node that has just become addressable knows nothing,
        // whatever was planned while it could not hear.
        let mut g = Graph::new();
        let (a, c) = (source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[c]); // the producer is not addressable yet

        g.add_link(a, "out", c, "in").unwrap();
        let apply = log.take();
        assert_eq!(sent(&apply).len(), 1, "the consumer is told");
        ack_all(&apply, &mut g);
        assert!(log.take().is_empty(), "and the producer's phase 3 went nowhere");

        attach_to(&mut g, &log, &[a]);
        let replanned = log.take();
        assert_eq!(
            sent(&replanned),
            [Sent::In { to: c, slot: "in".to_string(), services: vec![out_name(a, 0, "out")] }],
            "the slot is planned again from nothing"
        );
        ack_all(&replanned, &mut g);
        assert_eq!(
            sent(&log.take()),
            [Sent::Out { to: a, slot: "out".to_string(), targets: vec![(door_name(c, 0), 1)] }],
            "and the newly addressable producer is told what it feeds"
        );
    }

    #[test]
    fn a_cleared_graph_wires_the_next_patch_from_nothing() {
        // `clear` drops what the planner believed as well as the channels: a load that rebuilds the
        // same uids and the same links would otherwise diff against the dead patch's plan and decide
        // the producers had nothing to hear.
        let mut g = Graph::new();
        let (a, c) = (source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, c]);
        g.add_link(a, "out", c, "in").unwrap();
        settle(&log, &mut g);

        g.clear();
        g.add_node_at("_TestGated", None, a, "").unwrap();
        g.add_node_at("_TestSink", None, c, "").unwrap();
        let reborn = attach(&mut g, &[a, c]);
        g.add_link(a, "out", c, "in").unwrap();

        let apply = reborn.take();
        assert_eq!(
            sent(&apply),
            [Sent::In { to: c, slot: "in".to_string(), services: vec![out_name(a, 1, "out")] }],
            "the new generation's name, planned from nothing"
        );
        ack_all(&apply, &mut g);
        assert_eq!(
            sent(&reborn.take()),
            [Sent::Out { to: a, slot: "out".to_string(), targets: vec![(door_name(c, 1), 1)] }],
            "and the producer is grown, which a stale diff base would have skipped"
        );
    }

    #[test]
    fn a_cleared_graph_does_not_talk_to_the_nodes_it_destroyed() {
        // A channel outliving its node would deliver one node's wiring to whatever is born at that
        // uid next — which, after a load, is a different node entirely.
        let mut g = Graph::new();
        let (a, c) = (source(&mut g), sink(&mut g));
        let log = attach(&mut g, &[a, c]);
        g.clear();

        g.add_node_at("_TestGated", None, a, "").unwrap();
        g.add_node_at("_TestSink", None, c, "").unwrap();
        g.add_link(a, "out", c, "in").unwrap();
        assert!(log.take().is_empty(), "the dead node's channel is not the new node's");
    }
}
