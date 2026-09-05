//! The graph-side wire planner (spec §4): one sequence per consumer slot, three phases ordered by
//! acks — producer-shrink, consumer-apply, producer-grow. Attach, detach and replace are one
//! operation, because a slot message carries the full desired set.

use std::collections::HashMap;
use std::sync::Arc;

use goofi_node::ParamKey;

use super::wire::{Control, ControlSink, Envelope};
use goofi_node::Uid;

/// The producer end of a wire: a node, one of its output slots, and the `node.slot` a consumer
/// hears it as. A rename changes the third and none of the first two.
pub(crate) type Wire = (Uid, &'static str, String);

/// Whether two wires share a producer end, whatever they are named.
pub(crate) fn same_end(a: &Wire, b: &Wire) -> bool {
    a.0 == b.0 && a.1 == b.1
}

/// What a consumer subscribes THROUGH. An expression binding attaches through the same three
/// phases as a link, so the planner is keyed by subscription rather than by input slot.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Slot {
    In(&'static str),
    Bind(ParamKey),
}

/// The consumer subscription a sequence is about.
pub(crate) type SlotKey = (Uid, Slot);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Shrink,
    Apply,
    Grow,
}

/// One slot's in-flight wire change. The set it applies is [`WirePlanner::planned`].
struct Sequence {
    /// The producers that lost this consumer, and the ones that gained it.
    removed: Vec<Wire>,
    added: Vec<Wire>,
    /// `None` until the first [`WirePlanner::step`] — a sequence begins before phase 1, not in it.
    phase: Option<Phase>,
}

impl Sequence {
    /// Whether the consumer has NOT yet acked a set from this sequence.
    fn unapplied(&self) -> bool {
        matches!(self.phase, None | Some(Phase::Shrink) | Some(Phase::Apply))
    }
}

/// The graph's end of the wire plane: who to talk to, what each slot was last told, and what is in
/// flight.
#[derive(Default)]
pub(crate) struct WirePlanner {
    /// One per live node. A uid with no channel is not addressable — its messages are dropped and
    /// never awaited, so a partially attached graph converges instead of stalling.
    sinks: HashMap<Uid, Arc<dyn ControlSink>>,
    sequences: HashMap<SlotKey, Sequence>,
    /// seq → the sequence waiting on it, and the ONLY record of what is outstanding: a phase is
    /// complete when no entry here still names its slot.
    awaiting: HashMap<u64, SlotKey>,
    /// What each slot was last PLANNED to hold, not what it is confirmed to hold — the base a
    /// shrink/grow diff is taken against. [`Self::abandon`] takes it back on a refusal.
    planned: HashMap<SlotKey, Vec<Wire>>,
    /// Messages for nodes that are not addressable yet. A wire change needs no queue — it is
    /// re-PLANNED on attach — but a request has no state to re-derive from, so it is held.
    pending: Vec<(Uid, Control)>,
    next_seq: u64,
}

impl WirePlanner {
    /// Send one message that belongs to no sequence, and await no ack for it. HELD for a node with
    /// no channel yet, and delivered when its channel attaches.
    pub(crate) fn send(&mut self, uid: Uid, control: Control) {
        let Some(sink) = self.sinks.get(&uid).cloned() else {
            self.pending.push((uid, control));
            return;
        };
        self.next_seq += 1;
        sink.send(Envelope { seq: self.next_seq, control });
    }

    pub(crate) fn attach(&mut self, uid: Uid, sink: Arc<dyn ControlSink>) {
        self.sinks.insert(uid, sink);
        let held: Vec<Control> = {
            let mut keep = Vec::new();
            let mut mine = Vec::new();
            for (to, control) in std::mem::take(&mut self.pending) {
                if to == uid {
                    mine.push(control);
                } else {
                    keep.push((to, control));
                }
            }
            self.pending = keep;
            mine
        };
        for control in held {
            self.send(uid, control);
        }
    }

    /// Forget ONE node the graph destroyed — a removal, or the corpse a restart replaces. The
    /// sink OWNS the graph's end of that node's services, so it is released here or not at all.
    pub(crate) fn forget(&mut self, uid: Uid) {
        self.sinks.remove(&uid);
        self.sequences.retain(|(consumer, _), _| *consumer != uid);
        self.awaiting.retain(|_, (consumer, _)| *consumer != uid);
        self.planned.retain(|(consumer, _), _| *consumer != uid);
        self.pending.retain(|(to, _)| *to != uid);
    }

    /// Forget every consumer `live` does not name — a foreign consumer's death reaches this
    /// planner only as its absence from the view.
    pub(crate) fn forget_absent(&mut self, live: impl Fn(Uid) -> bool) {
        let gone: Vec<Uid> = self
            .planned
            .keys()
            .chain(self.sequences.keys())
            .map(|(consumer, _)| *consumer)
            .filter(|uid| !live(*uid))
            .collect();
        for uid in gone {
            self.forget(uid);
        }
    }

    /// Drop every channel and everything in flight.
    pub(crate) fn reset_channels(&mut self) {
        self.sinks.clear();
        self.sequences.clear();
        self.awaiting.clear();
        self.planned.clear();
        // A held request addresses a node this clear destroyed, so it must not reach a successor.
        self.pending.clear();
    }

    /// Start a slot's sequence, cancelling whatever it had in flight.
    pub(crate) fn begin(&mut self, key: SlotKey, desired: Vec<Wire>, removed: Vec<Wire>, added: Vec<Wire>) {
        // A cancelled sequence's unapplied additions are carried, or the evidence that this
        // consumer never subscribed disappears and another slot's phase 3 rings it.
        let carried = self
            .sequences
            .get(&key)
            .filter(|previous| previous.unapplied())
            .map(|previous| previous.added.clone())
            .unwrap_or_default();
        let added: Vec<Wire> =
            desired.iter().filter(|w| added.iter().chain(&carried).any(|a| same_end(a, w))).cloned().collect();
        self.abandon(&key);
        self.planned.insert(key.clone(), desired);
        self.sequences.insert(key, Sequence { removed, added, phase: None });
    }

    /// Forget a slot's sequence, everything it was waiting on, and the base that sequence moved to.
    /// The base goes back because one claiming MORE than the node holds leaves a producer that is
    /// never told to ring this consumer, which no later edit repairs.
    fn abandon(&mut self, key: &SlotKey) {
        self.sequences.remove(key);
        self.awaiting.retain(|_, waiting| waiting != key);
        self.forget_planned(key);
    }

    /// What this slot was last planned to hold — the set a change is diffed against.
    pub(crate) fn planned(&self, key: &SlotKey) -> Vec<Wire> {
        self.planned.get(key).cloned().unwrap_or_default()
    }

    /// Forget what a slot was planned to hold, so the next plan runs against nothing.
    pub(crate) fn forget_planned(&mut self, key: &SlotKey) {
        self.planned.remove(key);
    }

    /// Every planner key ever planned for `uid` and not since forgotten — the record of the
    /// channels spoken on, which is what an attach re-plans.
    pub(crate) fn keys_for(&self, uid: Uid) -> Vec<SlotKey> {
        self.planned.keys().filter(|(owner, _)| *owner == uid).cloned().collect()
    }

    /// Whether `key`'s in-flight sequence is still ABOUT to subscribe `wire`. A producer must not
    /// be told to ring it until then (§4).
    pub(crate) fn unapplied(&self, key: &SlotKey, wire: (Uid, &str)) -> bool {
        self.sequences.get(key).is_some_and(|s| s.unapplied() && s.added.iter().any(|a| a.0 == wire.0 && a.1 == wire.1))
    }

    /// Move to the next phase, or finish the sequence and answer `None`.
    pub(crate) fn step(&mut self, key: &SlotKey) -> Option<Phase> {
        let sequence = self.sequences.get_mut(key)?;
        let next = match sequence.phase {
            None => Some(Phase::Shrink),
            Some(Phase::Shrink) => Some(Phase::Apply),
            Some(Phase::Apply) => Some(Phase::Grow),
            Some(Phase::Grow) => None,
        };
        sequence.phase = next;
        if next.is_none() {
            self.sequences.remove(key);
        }
        next
    }

    /// The recipients of one phase: the producers that lost this consumer, or the ones that gained
    /// it. Phase 2 addresses the consumer itself, which the caller already knows.
    pub(crate) fn recipients(&self, key: &SlotKey, phase: Phase) -> Vec<Wire> {
        let Some(sequence) = self.sequences.get(key) else { return Vec::new() };
        match phase {
            Phase::Shrink => sequence.removed.clone(),
            Phase::Grow => sequence.added.clone(),
            Phase::Apply => Vec::new(),
        }
    }

    /// The full desired set of the sequence in flight on this slot — the planned base itself, which
    /// only [`Self::begin`] writes.
    pub(crate) fn desired(&self, key: &SlotKey) -> Vec<Wire> {
        self.planned(key)
    }

    /// Send one phase's messages and start awaiting their acks. Answers whether anything is now
    /// awaited — a phase with nothing to say must not park on an ack that never comes.
    pub(crate) fn dispatch(&mut self, key: &SlotKey, messages: Vec<(Uid, Control)>) -> bool {
        let mut waiting = false;
        for (uid, control) in messages {
            let Some(sink) = self.sinks.get(&uid).cloned() else { continue };
            self.next_seq += 1;
            let seq = self.next_seq;
            self.awaiting.insert(seq, key.clone());
            waiting = true;
            sink.send(Envelope { seq, control });
        }
        waiting
    }

    /// Answer one ack. `Some(key)` when it completed a phase and the sequence may advance; `None`
    /// when it is still outstanding, cancelled, or refused — a refusal abandons the sequence.
    pub(crate) fn ack(&mut self, seq: u64, ok: Result<(), String>) -> Option<SlotKey> {
        let key = self.awaiting.remove(&seq)?;
        if ok.is_err() {
            self.abandon(&key);
            return None;
        }
        (!self.awaiting.values().any(|waiting| *waiting == key)).then_some(key)
    }
}
