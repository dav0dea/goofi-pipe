//! The signal engine's author contract: the `Node` trait a signal node implements, its run-time
//! views, the compile-time class inventory, and the universal `common` scheduling group.

use std::fmt;

use goofi_core::{Data, Param};
use goofi_node::{param, NodeManifest, ParamDecl, ParamGroups, ParamKey, Params, ParamSpec};
use goofi_node::{ExprDecl, ExprMode, IsolationCell};
use indexmap::IndexMap;

pub mod discover;

/// A signal node's failure, propagated to the health plane rather than panicking.
#[derive(Debug, Clone)]
pub struct NodeError(pub String);

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NodeError {}
impl From<String> for NodeError {
    fn from(s: String) -> Self {
        NodeError(s)
    }
}
impl From<&str> for NodeError {
    fn from(s: &str) -> Self {
        NodeError(s.to_string())
    }
}

pub type NodeResult = std::result::Result<(), NodeError>;

/// The per-run input view; the two maps are keyed disjointly, so a slot is single XOR multi.
pub struct Inputs<'a> {
    singles: &'a IndexMap<&'static str, Option<Data>>,
    multis: Option<&'a IndexMap<&'static str, Vec<Data>>>,
}

impl<'a> Inputs<'a> {
    pub fn new(singles: &'a IndexMap<&'static str, Option<Data>>) -> Inputs<'a> {
        Inputs { singles, multis: None }
    }
    pub fn with_multi(
        singles: &'a IndexMap<&'static str, Option<Data>>,
        multis: &'a IndexMap<&'static str, Vec<Data>>,
    ) -> Inputs<'a> {
        Inputs { singles, multis: Some(multis) }
    }
    /// The latest frame on a single slot, or the first present frame on a `multi` slot.
    pub fn get(&self, name: &str) -> Option<&Data> {
        if let Some(o) = self.singles.get(name) {
            if let Some(d) = o.as_ref() {
                return Some(d);
            }
        }
        self.multis.and_then(|m| m.get(name)).and_then(|v| v.first())
    }
    /// The present frames on a `multi` slot, or a 0/1-element slice on a single slot.
    pub fn get_multi(&self, name: &str) -> &[Data] {
        if let Some(m) = self.multis {
            if let Some(v) = m.get(name) {
                return v.as_slice();
            }
        }
        match self.singles.get(name) {
            Some(Some(d)) => std::slice::from_ref(d),
            _ => &[],
        }
    }
}

/// A pre-sized output sink; slots left unset emit nothing.
pub struct Outputs<'a> {
    slots: &'a mut IndexMap<&'static str, Option<Data>>,
}

impl<'a> Outputs<'a> {
    pub fn new(slots: &'a mut IndexMap<&'static str, Option<Data>>) -> Outputs<'a> {
        Outputs { slots }
    }
    /// Set an output slot. Writing an unknown slot name is a no-op.
    pub fn set(&mut self, name: &str, data: Data) {
        if let Some(s) = self.slots.get_mut(name) {
            *s = Some(data);
        }
    }
}

/// Per-run engine context handed to a node.
#[derive(Debug, Default, Clone)]
pub struct NodeCtx {
    /// Monotonic seconds since the PATCH began — one clock across every node thread.
    pub now: f64,
}

impl NodeCtx {
    pub fn new() -> NodeCtx {
        NodeCtx::default()
    }
}

/// The two ways a user can author `common.max_frequency`; [`RunPolicy`] normalizes both to Hz.
pub const FREQ_MODE_UPDATES_PER_SECOND: &str = "updates-per-second";
pub const FREQ_MODE_SECONDS_PER_UPDATE: &str = "seconds-per-update";

/// When a node's `process` may run, lifted out of the params.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RunPolicy {
    /// Run whenever the rate cap allows, with no fresh input — a free-running producer.
    pub autotrigger: bool,
    /// Max run rate in Hz; `<= 0` is unbounded. Seconds-per-update is normalized to a rate here.
    pub max_frequency: f64,
}

impl RunPolicy {
    /// The minimum seconds between runs, or `None` when unbounded.
    pub fn period(&self) -> Option<f64> {
        (self.max_frequency > 0.0).then(|| 1.0 / self.max_frequency)
    }

    /// Read the policy from a node's `common` param group, defaulting each absent field.
    pub fn from_params(p: &ParamGroups) -> RunPolicy {
        let autotrigger = param(p, "common", "autotrigger")
            .and_then(Param::as_bool)
            .unwrap_or(false);
        let raw = param(p, "common", "max_frequency")
            .and_then(Param::as_f64)
            .unwrap_or(0.0);
        let seconds_per_update = param(p, "common", "frequency_mode").and_then(Param::as_str)
            == Some(FREQ_MODE_SECONDS_PER_UPDATE);
        let max_frequency = if seconds_per_update && raw > 0.0 { 1.0 / raw } else { raw };
        RunPolicy { autotrigger, max_frequency }
    }
}

/// One universal `common` param, as a function of the manifest it is added to. It may read the
/// manifest's static shape, but never `m.params` for a `common` key — that is a half-built world.
pub type CommonDecl = fn(&NodeManifest) -> ParamDecl;

/// Run on the node's own schedule instead of waiting for an input frame; defaults to `m.producer`.
fn autotrigger(m: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "autotrigger",
        spec: ParamSpec::Bool { default: m.producer },
        expression: None,
        doc: Some(
            "Run on the node's own schedule, instead of waiting for an input frame. \
             Turn this on for sources; leave it off for transforms driven by their input.",
        ),
    }
}

/// The rate cap, carried by every node as a `globals.default_ufreq` expression and live on a
/// producer. `trigger: true` is inert here — a `common.*` arrival never sets `trigger_pending`.
fn max_frequency(m: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "max_frequency",
        spec: ParamSpec::Float { default: 0.0, min: 0.0, max: 100.0 },
        expression: Some(ExprDecl {
            source: "globals.default_ufreq",
            mode: if m.producer { ExprMode::On } else { ExprMode::Off },
            trigger: true,
        }),
        doc: Some(
            "Rate cap for this node, read through `frequency_mode`. 0 means uncapped — the node \
             runs as often as the scheduler and its inputs allow.",
        ),
    }
}

/// How to read [`max_frequency`]: a rate, or a period.
fn frequency_mode(_: &NodeManifest) -> ParamDecl {
    ParamDecl {
        group: "common",
        name: "frequency_mode",
        spec: ParamSpec::Str {
            default: FREQ_MODE_UPDATES_PER_SECOND,
            options: &[FREQ_MODE_UPDATES_PER_SECOND, FREQ_MODE_SECONDS_PER_UPDATE],
            refresh: false,
        },
        expression: None,
        doc: Some(
            "How to read `max_frequency`: as a rate in Hz (updates per second), or as a period \
             in seconds between updates — convenient for very slow nodes.",
        ),
    }
}

/// The universal `common` scheduling group; a fourth param is added here and nowhere else.
pub static COMMON_DECLS: &[CommonDecl] = &[autotrigger, max_frequency, frequency_mode];

pub fn common_decls(m: &NodeManifest) -> impl Iterator<Item = ParamDecl> + '_ {
    COMMON_DECLS.iter().map(move |d| d(m))
}

/// Guarantee a node carries the universal `common` group, keeping any key it declared itself.
pub fn with_common(params: ParamGroups, m: &NodeManifest) -> ParamGroups {
    let mut common = params.get("common").cloned().unwrap_or_default();
    for d in common_decls(m) {
        common.entry(d.name.to_string()).or_insert_with(|| d.spec.to_param());
    }
    let mut merged = ParamGroups::new();
    merged.insert("common".to_string(), common);
    for (k, v) in params {
        if k != "common" {
            merged.insert(k, v);
        }
    }
    merged
}

pub trait Node: Send {
    /// Derived init, after the params have been seeded. An `Err` leaves the node uninitialized and
    /// the next interaction retries the whole initialization on this same instance.
    fn setup(&mut self, _ctx: &mut NodeCtx, _p: &Params<'_>) -> NodeResult {
        Ok(())
    }
    /// The run body: read latest inputs + live params, write outputs.
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        ctx: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult;
    /// React to a param edit; the engine replays it per declared param at initialization.
    fn on_param_changed(&mut self, _key: &ParamKey, _v: &Param) -> NodeResult {
        Ok(())
    }
    /// Re-enumerate a `Str` param's options — the ⟳ button. `p` is the node's LIVE params.
    fn on_param_refreshed(&mut self, _key: &ParamKey, _p: &Params<'_>) -> Option<Vec<String>> {
        None
    }
    // Teardown is `impl Drop`, and does NOT fire between initialization retries: a `setup` that
    // fails partway must release what it acquired before returning `Err`.
}

/// The generic node factory a [`NodeClass`] registers: a default instance, type-erased.
pub fn default_factory<T: Node + Default + 'static>() -> Box<dyn Node> {
    Box::new(T::default())
}

/// A compile-time node registration via `inventory`: the shared manifest plus the build half —
/// the factory and the tier cell — which are the engine's business rather than the manifest's.
pub struct NodeClass {
    pub manifest: NodeManifest,
    pub isolation: &'static IsolationCell,
    /// Build a default instance, type-erased.
    pub factory: fn() -> Box<dyn Node>,
}

inventory::collect!(NodeClass);

pub fn catalog() -> impl Iterator<Item = &'static NodeManifest> {
    classes().map(|c| &c.manifest)
}

pub fn classes() -> impl Iterator<Item = &'static NodeClass> {
    inventory::iter::<NodeClass>()
}

pub fn find(type_name: &str) -> Option<&'static NodeManifest> {
    find_class(type_name).map(|c| &c.manifest)
}

pub fn find_class(type_name: &str) -> Option<&'static NodeClass> {
    inventory::iter::<NodeClass>().find(|c| c.manifest.type_name == type_name)
}
