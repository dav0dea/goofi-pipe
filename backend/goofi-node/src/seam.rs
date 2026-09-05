//! The engine seam: the trait an engine registers behind, and the settled view its settle reads.
//! The graph looks down at this and nothing engine-specific; an engine looks down at this and the
//! transport, and never at the graph.

use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use goofi_core::Param;
use indexmap::IndexMap;

use crate::{
    BindingId, ExprEvaluator, Isolation, IsolationCell, NodeManifest, ParamDecl, ParamGroups,
    ParamKey, Status, Uid,
};

/// A doorbell id: `0` is a control message, `1..=64` the index of an input slot in
/// `manifest.inputs`, `65..=128` an expression channel the graph allocated at bind time.
pub type EventId = u8;

/// One resolved expression variable, graph-side: the model's spelling, which a view exposes and
/// an engine projects onto its own wire vocabulary.
#[derive(Clone, Debug)]
pub enum BoundVar {
    /// A producer's output slot, and the doorbell id it rings this consumer with.
    Stream { var: String, producer: Uid, slot: &'static str, event_id: EventId },
    /// A `globals.*` read, resolved and shipped inline — a globals edit re-sends the binding.
    Value { var: String, value: Param },
    /// The graph could not resolve it: an unknown node, a slot that does not exist, an ambiguous
    /// bare `nd()` on a multi-output producer, a global that is not defined.
    Missing { var: String, reason: String },
}

impl BoundVar {
    /// The producer wire a stream variable subscribes, `None` for the other kinds.
    pub fn wire(&self) -> Option<(Uid, &'static str)> {
        match self {
            BoundVar::Stream { producer, slot, .. } => Some((*producer, slot)),
            _ => None,
        }
    }
}

/// Wakes the status drain the moment any engine has something to hand over — the alternative to a
/// poll-to-discover. Every node report notifies; the worker parks on it between paced duties.
#[derive(Default)]
pub struct DrainWaker {
    woke: Mutex<bool>,
    cv: Condvar,
}

impl DrainWaker {
    pub fn notify(&self) {
        *self.woke.lock().unwrap() = true;
        self.cv.notify_one();
    }

    /// Park until a notify or `timeout`, and consume the wake either way.
    pub fn wait_timeout(&self, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        let mut woke = self.woke.lock().unwrap();
        while !*woke {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                break;
            }
            let (guard, _) = self.cv.wait_timeout(woke, left).unwrap();
            woke = guard;
        }
        *woke = false;
    }
}

/// One port-resolved leaf-to-leaf edge, in link order — which IS a multi input's wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub producer: (Uid, &'static str),
    pub consumer: (Uid, &'static str),
}

/// A file's size and mtime — what a rescan diffs. `None` when it could not be stat'd, which
/// compares equal to itself and so reads as "unchanged".
pub type Stamp = (u64, std::time::SystemTime);

/// One node file's outcome from an engine's scan of its folder.
pub struct ScannedType {
    pub type_name: String,
    pub stamp: Option<Stamp>,
    pub outcome: Scanned,
}

pub enum Scanned {
    /// Registered; `replaced` says an earlier file already held the name, and this one now does.
    Registered { isolation: Isolation, replaced: bool },
    /// Not loadable; the palette lists the type greyed with this reason.
    Unavailable(String),
}

/// One binding as settle ships it: the derived state an engine reads, never the authored record.
pub struct BindingView<'a> {
    pub key: &'a ParamKey,
    pub rewritten: &'a str,
    pub vars: &'a [BoundVar],
    pub trigger: bool,
    pub id: Option<BindingId>,
    /// Whether the graph ships it — a disabled or unbindable binding leaves the literal standing.
    pub live: bool,
}

/// One running node as the settled view carries it.
pub struct NodeView<'a> {
    pub engine: &'static str,
    pub name: &'a str,
    pub generation: u64,
    /// Whether this node's engine wakes it by doorbell; a scheduled consumer is never rung.
    pub rings: bool,
    pub manifest: &'static NodeManifest,
    pub params: &'a ParamGroups,
    pub bindings: Vec<BindingView<'a>>,
}

/// The settled graph, as every engine reads it after a batch: the WHOLE graph — engines filter.
pub struct GraphView<'a> {
    pub instance: &'a str,
    pub edges: &'a [Edge],
    pub nodes: HashMap<Uid, NodeView<'a>>,
}

impl GraphView<'_> {
    /// A consumer input slot's desired producers, in wire order.
    pub fn wires_into<'s>(
        &'s self,
        uid: Uid,
        slot: &'s str,
    ) -> impl Iterator<Item = (Uid, &'static str)> + 's {
        self.edges
            .iter()
            .filter(move |e| e.consumer.0 == uid && e.consumer.1 == slot)
            .map(|e| e.producer)
    }
}

/// What [`Engine::editor`] hands back: the open or close itself, run once the graph lock is
/// released, because a plugin's window takes its time to come up and every other op would wait.
pub type EditorAction = Box<dyn FnOnce() -> Result<bool, String> + Send>;

/// One thing a batch of ops changed, recorded by the op path for the settle that follows. The
/// delivery half of a write is deferred so one batch yields ONE decision, from settled state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Touched {
    /// A consumer input whose wire set may have moved.
    Slot(Uid, &'static str),
    /// A param whose value or binding moved and must reach its node.
    Param(Uid, ParamKey),
}

/// One node class an engine advertises: the shared manifest plus the display tier. The engine a
/// type belongs to is WHICH library advertises it — no tag field exists anywhere.
#[derive(Clone, Copy)]
pub struct LibraryEntry {
    pub manifest: &'static NodeManifest,
    /// The display tier's live cell — a Python type's runtime demotion reads through it.
    pub isolation: &'static IsolationCell,
}

/// An engine: the runtime authority for its nodes. The graph applies every op to the MODEL and
/// propagates through these doors; an engine owns instances, health reporting, within-engine
/// transport and its own library, and the graph sees none of them.
pub trait Engine: Send {
    /// The id a registration is keyed by, and the palette's provenance for this library.
    fn id(&self) -> &'static str;
    /// Whether this engine's nodes wake on doorbells. A scheduled engine drains its boundary
    /// subscribers before each tick instead, and a producer facing it rings nothing.
    fn doorbell_driven(&self) -> bool;
    /// Whether this engine's drain marked work only a settle can finish — a Ready it collected,
    /// an ack that completed a phase. What makes the memo rule honest.
    fn dirty(&self) -> bool;
    /// Every node class this engine can build, advertised on request.
    fn library(&self) -> Vec<LibraryEntry>;
    /// Scan ONE folder of this engine's node files and register what loads, a later file taking
    /// a name an earlier one held. A file that cannot load answers why instead.
    fn scan(&mut self, _dir: &Path) -> Vec<ScannedType> {
        Vec::new()
    }
    /// Types the engine finds on its own account — the platform's plugin folders — scanned
    /// after every root, so a name a root holds is known when these register.
    fn scan_own(&mut self) -> Vec<ScannedType> {
        Vec::new()
    }

    /// Forget a type a scan registered; whether this engine held it.
    fn remove_type(&mut self, _type_name: &str) -> bool {
        false
    }
    /// The SDK crate an `.rs` file in this engine's folder is built against, when it takes one.
    fn rust_sdk(&self) -> Option<&'static str> {
        None
    }
    /// The open patch's workspace: where a node's opaque state is kept between two births.
    fn set_workspace(&mut self, _dir: &Path) {}
    /// Write every live node's opaque state where the next birth at its uid will find it.
    fn persist(&mut self) {}
    /// The universal params this engine adds to every one of its nodes — declarations, so the
    /// palette's tooltips and the default-expression seeding read one door. Empty by default.
    fn universal_decls(&self, _manifest: &'static NodeManifest) -> Vec<ParamDecl> {
        Vec::new()
    }
    /// The record a fresh instance of `manifest` starts from: the declared defaults in declared
    /// order — the editor renders in insertion order — with `supplied` folded on top, so a patch
    /// saved before a param existed still gets that param's default; then each universal group
    /// rebuilt LAST in this engine's order, a declared or supplied value winning over the default.
    /// By MANIFEST, not by name, so a type the library no longer answers for can still say what
    /// its live nodes hold. Every engine answers this one way; none overrides it.
    fn normalize_params(&self, manifest: &'static NodeManifest, supplied: Option<ParamGroups>) -> ParamGroups {
        let mut params = ParamGroups::new();
        for (group, entries) in manifest.default_params() {
            params.entry(group).or_default().extend(entries);
        }
        for (group, entries) in supplied.into_iter().flatten() {
            params.entry(group).or_default().extend(entries);
        }
        let mut universal = ParamGroups::new();
        for d in self.universal_decls(manifest) {
            universal.entry(d.group.to_string()).or_default().insert(d.name.to_string(), d.spec.to_param());
        }
        for (group, defaults) in universal {
            let mut held = params.shift_remove(&group).unwrap_or_default();
            let mut page: IndexMap<String, Param> =
                defaults.into_iter().map(|(name, default)| { let value = held.shift_remove(&name).unwrap_or(default); (name, value) }).collect();
            page.extend(held);
            params.insert(group, page);
        }
        if let Some(common) = params.shift_remove("common") {
            params.insert("common".to_string(), common);
        }
        params
    }
    /// Birth at `uid`, with the graph-minted generation. `Some` carries a boot error: the node
    /// then exists holding its place and saying why it is not running.
    fn insert(
        &mut self,
        uid: Uid,
        type_name: &str,
        generation: u64,
        params: &ParamGroups,
    ) -> Option<String>;
    fn remove(&mut self, uid: Uid);
    /// Deliver a settled batch: the touched items, plus whatever the engine's own drain marked
    /// pending. The ONLY place an engine composes messages — drain collects, settle decides.
    fn settle(&mut self, view: &GraphView<'_>, touched: &[Touched]);
    /// Hand over every queued health report. A pull: the caller owns the pace.
    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize;
    /// Re-enumerate a `Str` param's options on the node's own thread — the one imperative
    /// settled state cannot express.
    fn refresh_param(&mut self, uid: Uid, key: ParamKey);
    /// Fire a pulse param on the node's own thread: a request the node acts on and stores nothing of.
    fn pulse_param(&mut self, uid: Uid, key: ParamKey);
    /// Whether a node of this type has an editor window of its own — on the machine goofi runs
    /// on, so a platform with no window host answers false.
    fn has_editor(&self, _type_name: &str) -> bool {
        false
    }
    /// Show or hide `uid`'s editor: the action, answering whether it changed anything.
    fn editor(&mut self, uid: Uid, _show: bool) -> Result<EditorAction, String> {
        Err(format!("{uid} has no editor"))
    }
    /// The patch clock origin moved — a clear reset it. No-op for an engine with no patch time.
    fn reset_clock(&mut self, _origin: Instant) {}
    /// The graph's expression evaluator, shared with every engine that evaluates `nd()` bindings
    /// on its own thread. No-op for an engine that never does.
    fn set_evaluator(&mut self, _evaluator: Arc<dyn ExprEvaluator>) {}
    /// The composition root's door to an engine's CONCRETE surface — a runtime type registry, a
    /// device enumeration. Everything generic stays on this trait.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn shutdown(&mut self);
}

/// One doorbell an output slot rings: the consumer, the id it is rung with, and the subscription
/// the ring is for — a declared input slot, or a binding.
pub struct Ringer<'a> {
    pub consumer: Uid,
    pub event_id: EventId,
    pub via: Via<'a>,
}

pub enum Via<'a> {
    Slot(&'static str),
    Binding(&'a BindingView<'a>),
}

impl GraphView<'_> {
    /// Every doorbell `(producer, slot)` rings, read off the view: wired consumer slots by
    /// manifest position and `nd()` channels by the event id the graph allocated — only for
    /// consumers whose engine wakes on doorbells, and never a slot past the event-id budget. In
    /// one order for one settled state, so a list of them compares.
    pub fn ringers(&self, producer: Uid, slot: &str) -> Vec<Ringer<'_>> {
        let wired = self.edges.iter().filter(|e| e.producer.0 == producer && e.producer.1 == slot).filter_map(|e| {
            let node = self.nodes.get(&e.consumer.0).filter(|n| n.rings)?;
            let at = node.manifest.inputs.iter().position(|s| s.name == e.consumer.1)?;
            (at < 64).then_some(Ringer { consumer: e.consumer.0, event_id: at as EventId + 1, via: Via::Slot(e.consumer.1) })
        });
        let bound = self.nodes.iter().filter(|(_, n)| n.rings).flat_map(|(uid, n)| {
            n.bindings.iter().filter(|b| b.live).flat_map(move |b| {
                b.vars.iter().filter_map(move |v| match v {
                    BoundVar::Stream { producer: p, slot: s, event_id, .. } if *p == producer && *s == slot => {
                        Some(Ringer { consumer: *uid, event_id: *event_id, via: Via::Binding(b) })
                    }
                    _ => None,
                })
            })
        });
        let mut ringers: Vec<Ringer<'_>> = wired.chain(bound).collect();
        ringers.sort_by_key(|r| (r.consumer.0, r.event_id));
        ringers
    }
}
