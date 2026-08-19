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
//! What replans is every link change, every EXPRESSION BINDING change (a binding is a consumer
//! subscription like any other — what differs is only that its phase 2 is a `SetParam`), and a
//! channel being attached, which plans that node's slots from an EMPTY base because it heard
//! nothing said before it arrived.

use std::collections::HashMap;
use std::sync::Arc;

use super::wire::{Control, ControlSink, Envelope};
use crate::Uid;

/// The producer end of a wire: a node and one of its output slots. A producer end is always an
/// output slot, whichever kind of consumer it feeds.
pub(crate) type Wire = (Uid, &'static str);

/// What a consumer subscribes THROUGH. An expression reference IS a link — a bound param attaches
/// and detaches through the same three phases, its `SetParam` being the consumer-apply — so the
/// planner is keyed by subscription rather than by input slot. A binding names itself by a
/// graph-minted id that survives a rebind, so the key stays `Copy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Slot {
    In(&'static str),
    Bind(usize),
}

/// The consumer subscription a sequence is about.
pub(crate) type SlotKey = (Uid, Slot);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Shrink,
    Apply,
    Grow,
}

/// One slot's in-flight wire change. The set it applies is [`WirePlanner::planned`] — one holder,
/// written by `begin` and taken back by `abandon` together with the sequence itself.
struct Sequence {
    /// The producers that lost this consumer, and the ones that gained it.
    removed: Vec<Wire>,
    added: Vec<Wire>,
    /// `None` until the first [`WirePlanner::step`] — a sequence begins before phase 1, not in it.
    phase: Option<Phase>,
}

impl Sequence {
    /// Whether the consumer has NOT yet acked a set from this sequence — it has been told nothing,
    /// or has been sent an `InSlot` it has not answered. Past `Apply` it holds the planned set.
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
    /// is declarative: the node applies what it was sent whether or not its ack came back. A REFUSAL
    /// is the one answer that says otherwise, and [`Self::abandon`] takes the base back for it.
    planned: HashMap<SlotKey, Vec<Wire>>,
    /// Messages for nodes that are not addressable yet. The birth barrier is a WINDOW, not a state:
    /// a ⟳ clicked on a node the user has only just placed falls inside it, and a dropped request
    /// is a button that does nothing. A wire change needs no queue — it is re-PLANNED on attach —
    /// but a request has no state to re-derive from, so it is held.
    pending: Vec<(Uid, Control)>,
    next_seq: u64,
}

impl WirePlanner {
    /// Send one message that belongs to no sequence, and await no ack for it. A `RefreshParam` is
    /// the case: it is a request, not a wire change, and its answer comes back as its own
    /// [`super::Status`] rather than on the ack. HELD for a node with no channel yet — see
    /// [`Self::pending`] — and delivered when its channel attaches.
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

    /// Forget ONE node the graph destroyed — a removal, or the corpse a restart replaces.
    ///
    /// Not tidiness. The sink OWNS the graph's end of that node's services, so keeping it keeps them
    /// allocated for the rest of the process, and the startup sweep is blind to that: a live
    /// process's own nodes read `Alive`. It has to be released here or not at all. It is also what
    /// makes the birth barrier hold for a REBIRTH — a sink outliving its node makes the next node
    /// at that uid look addressable while it is not.
    ///
    /// Only what this node CONSUMED is dropped; what it produced belongs to its consumers.
    /// [`Self::pending`] SURVIVES: a rebirth is the same node, so a request queued before it ever
    /// attached is still a request for it.
    pub(crate) fn detach(&mut self, uid: Uid) {
        self.sinks.remove(&uid);
        self.sequences.retain(|(consumer, _), _| *consumer != uid);
        self.awaiting.retain(|_, (consumer, _)| *consumer != uid);
        self.planned.retain(|(consumer, _), _| *consumer != uid);
    }

    /// [`Self::detach`], plus the queue: the node at this uid is RETIRED rather than reborn, so
    /// anything still held for it addresses nobody. Delivering it to whatever is added at that uid
    /// next — an undo of the delete, say — would run one node's device scan against another's.
    pub(crate) fn forget(&mut self, uid: Uid) {
        self.detach(uid);
        self.pending.retain(|(to, _)| *to != uid);
    }

    /// Drop every channel and everything in flight, keeping the generations: a `clear` destroys the
    /// nodes those channels addressed, and a channel held past its node's death would deliver one
    /// node's wiring to another born at the same uid.
    pub(crate) fn reset_channels(&mut self) {
        self.sinks.clear();
        self.sequences.clear();
        self.awaiting.clear();
        self.planned.clear();
        // A held request addresses a node this clear destroyed. Delivering it to whatever is born
        // at that uid next would run one patch's device scan against another's node.
        self.pending.clear();
    }

    /// Start a slot's sequence, cancelling whatever it had in flight.
    pub(crate) fn begin(&mut self, key: SlotKey, desired: Vec<Wire>, removed: Vec<Wire>, added: Vec<Wire>) {
        // A cancelled sequence that never applied leaves its own additions unapplied while the diff
        // base has moved past them — so unless they are carried, the evidence that this consumer
        // never subscribed disappears, and some other slot's phase 3 tells their producer to ring
        // it. Rebuilt in `desired` order, so a phase's messages go out in the set's own order.
        let carried = self
            .sequences
            .get(&key)
            .filter(|previous| previous.unapplied())
            .map(|previous| previous.added.clone())
            .unwrap_or_default();
        let added: Vec<Wire> =
            desired.iter().copied().filter(|w| added.contains(w) || carried.contains(w)).collect();
        self.abandon(key);
        self.planned.insert(key, desired);
        self.sequences.insert(key, Sequence { removed, added, phase: None });
    }

    /// Forget a slot's sequence, everything it was waiting on, and the base that sequence moved to.
    ///
    /// The base is taken back because a refusal is what abandons a sequence, and the graph learns
    /// only THAT the node did not reach the planned set — never which wires it ended up holding. So
    /// the next change to this slot diffs against nothing and re-sends the whole set: a base that
    /// claims less than the node holds costs one redundant message, and one that claims more costs a
    /// producer that is never told to ring this consumer, which no later edit to the slot repairs.
    fn abandon(&mut self, key: SlotKey) {
        self.sequences.remove(&key);
        self.awaiting.retain(|_, waiting| *waiting != key);
        self.forget_planned(key);
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

    /// The full desired set of the sequence in flight on this slot — the planned base itself, since
    /// only [`Self::begin`] writes that base and it cancels what was in flight as it does. So a
    /// phase can never compose against a set a later plan has moved past.
    pub(crate) fn desired(&self, key: SlotKey) -> Vec<Wire> {
        self.planned(key)
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

