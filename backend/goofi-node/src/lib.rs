//! The ONE node abstraction, its runtime plumbing, and the native compile-time catalog.

use std::fmt;

use goofi_core::{Data, Param, SlotType};
use indexmap::IndexMap;

pub mod describe;
pub mod seam;
pub use describe::{leak_manifest, type_name_of};
pub use seam::{
    BindingView, BoundVar, DrainWaker, Edge, Engine, EventId, GraphView, LibraryEntry, NodeView,
    Request, Scanned, ScannedType, Stamp, Touched,
};

/// A `u64` internally, a 12-hex string in the `.gfi` and on the wire.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Uid(pub u64);

impl Uid {
    pub fn to_hex(self) -> String {
        format!("{:012x}", self.0)
    }
    /// Exactly 12 hex, nothing wider: bounding the domain is what makes `next_uid`'s `+ 1` total
    /// at every site rather than checked at each one.
    pub fn from_hex(s: &str) -> Option<Uid> {
        if s.len() != 12 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u64::from_str_radix(s, 16).ok().map(Uid)
    }
}

impl std::fmt::Display for Uid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Defines a `pub struct $name(pub String)` error newtype with its Display / Error / `From` impls.
macro_rules! string_error {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $name(pub String);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl std::error::Error for $name {}
        impl From<String> for $name {
            fn from(s: String) -> Self {
                $name(s)
            }
        }
        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                $name(s.to_string())
            }
        }
    };
}

/// Grouped params: `group -> (name -> Param)`, insertion-ordered.
pub type ParamGroups = IndexMap<String, IndexMap<String, Param>>;

/// A `(group, name)` address into a node's params.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct ParamKey {
    pub group: String,
    pub name: String,
}

impl ParamKey {
    pub fn new(group: impl Into<String>, name: impl Into<String>) -> ParamKey {
        ParamKey {
            group: group.into(),
            name: name.into(),
        }
    }
}

pub fn param<'a>(p: &'a ParamGroups, group: &str, name: &str) -> Option<&'a Param> {
    p.get(group)?.get(name)
}

/// A static param descriptor, so a node declares its params as a `static &[ParamDecl]`.
#[derive(Clone, Copy)]
pub struct ParamDecl {
    pub group: &'static str,
    pub name: &'static str,
    pub spec: ParamSpec,
    /// An optional default expression the engine seeds as a binding; `None` ⇒ a literal default.
    pub expression: Option<ExprDecl>,
    /// Help text for the UI's tooltip.
    pub doc: Option<&'static str>,
}

/// A declared param expression: its source, whether it starts live, and whether it wakes the node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExprDecl {
    pub source: &'static str,
    /// Whether this expression starts live; `Off` merely retains it as the param's expression text.
    pub mode: ExprMode,
    /// Whether re-evaluating it also wakes `process()`.
    pub trigger: bool,
}

/// Whether a declared [`ExprDecl`] is live or merely carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExprMode {
    Off,
    On,
}

/// The kind + defaults of a declared param.
#[derive(Clone, Copy)]
pub enum ParamSpec {
    Float { default: f64, min: f64, max: f64 },
    Int { default: i64, min: i64, max: i64 },
    Bool { default: bool },
    Str { default: &'static str, options: &'static [&'static str], refresh: bool },
}

impl ParamSpec {
    pub fn to_param(self) -> Param {
        match self {
            ParamSpec::Float { default, min, max } => Param::float(default, min, max),
            ParamSpec::Int { default, min, max } => Param::int(default, min, max),
            ParamSpec::Bool { default } => Param::boolean(default),
            ParamSpec::Str { default, options, refresh } => Param::Str {
                value: default.to_string(),
                options: (!options.is_empty())
                    .then(|| options.iter().map(|s| s.to_string()).collect()),
                refresh,
            },
        }
    }
}

/// Build a grouped [`ParamGroups`] from a flat, group-tagged declaration list.
pub fn params_from_decls(decls: &[ParamDecl]) -> ParamGroups {
    let mut groups = ParamGroups::new();
    for d in decls {
        groups
            .entry(d.group.to_string())
            .or_default()
            .insert(d.name.to_string(), d.spec.to_param());
    }
    groups
}

/// A typed read-only view of a node's params, for a cold param that is mirrored to no field.
pub struct Params<'a>(&'a ParamGroups);

impl<'a> Params<'a> {
    pub fn new(groups: &'a ParamGroups) -> Params<'a> {
        Params(groups)
    }
    pub fn f64(&self, group: &str, name: &str) -> Option<f64> {
        param(self.0, group, name).and_then(Param::as_f64)
    }
    pub fn i64(&self, group: &str, name: &str) -> Option<i64> {
        param(self.0, group, name).and_then(Param::as_i64)
    }
    pub fn bool(&self, group: &str, name: &str) -> Option<bool> {
        param(self.0, group, name).and_then(Param::as_bool)
    }
    pub fn str(&self, group: &str, name: &str) -> Option<&str> {
        param(self.0, group, name).and_then(Param::as_str)
    }
    pub fn groups(&self) -> &ParamGroups {
        self.0
    }
}

/// An opaque handle to a compiled expression, owned by the evaluator.
pub type BindingId = u64;

string_error!(
    /// A param-expression failure: a compile error, an exception, or an incompatible result.
    ExprError
);

/// The result of compiling an expression: the evaluator's opaque handle.
pub struct Compiled {
    pub id: BindingId,
}

/// One expression variable's value, as the graph resolved it.
#[derive(Clone, Debug)]
pub enum Local {
    Frame(Data),
    Value(Param),
}

/// Per-evaluation context handed to [`ExprEvaluator::eval`].
pub struct EvalCtx<'a> {
    /// The expression's variables, keyed by the generated name the rewrite minted; `None` has not
    /// arrived yet, and the expression sees it as absent.
    pub locals: &'a std::collections::HashMap<String, Option<Local>>,
    /// Engine wall-clock seconds (`NodeCtx::now`) — for time-based (variable-less) expressions.
    pub t: f64,
    /// The param being driven, a type template the evaluator coerces its result to.
    pub target: &'a Param,
}

/// Evaluates param expressions; implemented in `goofi-python` and injected, so the engine core
/// carries no pyo3 dependency.
pub trait ExprEvaluator: Send + Sync {
    fn compile(&self, source: &str) -> Result<Compiled, ExprError>;
    fn eval(&self, id: BindingId, ctx: &EvalCtx<'_>) -> Result<Param, ExprError>;
    fn release(&self, id: BindingId);
}

/// One `nd(..)` call, with both spans its consumers need: the name literal a rename replaces, and
/// the whole term a rewrite replaces.
pub struct NdCall<'a> {
    pub start: usize,
    pub name_start: usize,
    pub name_end: usize,
    /// One past the closing `)`, or `None` when the call does not close cleanly — a rewrite leaves
    /// those verbatim, so the failure shows up as an eval error.
    pub end: Option<usize>,
    pub name: &'a str,
}

/// Scan `source` for `nd('name')` calls, in source order. A lexical scan, not a parse.
pub fn scan_nd_calls(source: &str) -> Vec<NdCall<'_>> {
    let b = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 <= b.len() {
        if &b[i..i + 2] != b"nd" {
            i += 1;
            continue;
        }
        let boundary = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        let mut j = i + 2;
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }
        if boundary && j < b.len() && b[j] == b'(' {
            j += 1;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b'\'' || b[j] == b'"') {
                let q = b[j];
                j += 1;
                let start = j;
                while j < b.len() && b[j] != q {
                    j += 1;
                }
                if j < b.len() {
                    let mut close = j + 1;
                    while close < b.len() && (b[close] as char).is_whitespace() {
                        close += 1;
                    }
                    let end = (b.get(close) == Some(&b')')).then_some(close + 1);
                    out.push(NdCall {
                        start: i,
                        name_start: start,
                        name_end: j,
                        end,
                        name: &source[start..j],
                    });
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 2;
    }
    out
}

/// One `globals.<name>` read [`scan_globals`] found; the span covers the `globals.` prefix too.
pub struct GlobalRead<'a> {
    pub start: usize,
    pub end: usize,
    pub name: &'a str,
}

/// Scan `source` for `globals.<name>` reads, on the same word-boundary rule.
pub fn scan_globals(source: &str) -> Vec<GlobalRead<'_>> {
    const PREFIX: &str = "globals.";
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(pos) = source[i..].find(PREFIX) {
        let start = i + pos;
        i = start + PREFIX.len();
        if start > 0 && is_ident(bytes[start - 1]) {
            continue;
        }
        let name_start = start + PREFIX.len();
        let mut end = name_start;
        while end < bytes.len() && is_ident(bytes[end]) {
            end += 1;
        }
        if end > name_start && !bytes[name_start].is_ascii_digit() {
            out.push(GlobalRead { start, end, name: &source[name_start..end] });
            i = end;
        }
    }
    out
}

/// Where a node type's code actually runs. This is the ONE owner of that fact: the palette shows
/// it, the per-node runtime overlay reports it, and a Python type's factory BUILDS from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isolation {
    /// Compiled into the binary, on the engine's own threads.
    Native,
    /// Python, in the embedded free-threaded interpreter.
    InProcess,
    /// Python, in a subprocess with its own GIL.
    Subprocess,
}

impl Isolation {
    /// The wire name, shared by `inspect_type`'s `tier` and the per-node runtime overlay.
    pub fn wire(self) -> &'static str {
        match self {
            Isolation::Native => "native",
            Isolation::InProcess => "in-process",
            Isolation::Subprocess => "subprocess",
        }
    }
    /// Which language the node is written in — a reading of the tier, never a second field.
    pub fn language(self) -> &'static str {
        match self {
            Isolation::Native => "rust",
            _ => "python",
        }
    }
    fn from_u8(v: u8) -> Isolation {
        match v {
            0 => Isolation::Native,
            1 => Isolation::InProcess,
            _ => Isolation::Subprocess,
        }
    }
}

/// A type's [`Isolation`], shared by its `NodeClass`/`DynType` registration and captured by
/// each running node at birth.
/// It is interior-mutable for one reason: a Python node that re-enables the GIL at RUNTIME is only
/// discovered to be subprocess-bound after its import already passed the probe, and demoting the
/// type is what the next `restart_node` reads.
#[derive(Debug)]
pub struct IsolationCell(std::sync::atomic::AtomicU8);

impl IsolationCell {
    pub const fn new(i: Isolation) -> IsolationCell {
        IsolationCell(std::sync::atomic::AtomicU8::new(i as u8))
    }
    /// A cell of its own, for a manifest built at runtime rather than declared in a `static`.
    pub fn leak(i: Isolation) -> &'static IsolationCell {
        Box::leak(Box::new(IsolationCell::new(i)))
    }
    pub fn get(&self) -> Isolation {
        Isolation::from_u8(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
    /// Returns whether this changed anything, so a caller can report a demotion exactly once.
    pub fn set(&self, i: Isolation) -> bool {
        self.0.swap(i as u8, std::sync::atomic::Ordering::Relaxed) != i as u8
    }
}

/// The cell every compiled-in node points at. Shared because a native node's tier is fixed —
/// only a Python type, whose cell is leaked per type at discovery, is ever written.
pub static NATIVE: IsolationCell = IsolationCell::new(Isolation::Native);

pub struct SlotDecl {
    pub name: &'static str,
    pub kind: SlotType,
    /// Whether fresh data on this slot wakes `process()` (vs. a held reference input).
    pub trigger_process: bool,
    /// A `multi` slot takes any number of wires and delivers them as an ordered `&[Data]`.
    pub multi: bool,
    /// A required slot must hold data when the node runs, so the node may read it unconditionally.
    pub required: bool,
}

pub struct OutputDecl {
    pub name: &'static str,
    pub kind: SlotType,
}

/// Static, declarative node metadata — plain data, shared by every engine.
pub struct NodeManifest {
    pub type_name: &'static str,
    pub category: &'static str,
    pub doc: &'static str,
    pub inputs: &'static [SlotDecl],
    pub outputs: &'static [OutputDecl],
    /// Declared params; the runtime `ParamGroups` is built on demand by [`Self::default_params`].
    pub params: &'static [ParamDecl],
    /// This type is a SOURCE: it makes frames on its own schedule, so `common.autotrigger` and the
    /// carried `globals.default_ufreq` expression both default on.
    pub producer: bool,
}

impl NodeManifest {
    pub fn output_buffer(&self) -> IndexMap<&'static str, Option<Data>> {
        self.outputs.iter().map(|o| (o.name, None)).collect()
    }
    pub fn default_params(&self) -> ParamGroups {
        params_from_decls(self.params)
    }
}

/// Where a node is in its own lifecycle. Two variants rather than the projection's four:
/// `creating` is the GRAPH's and `error` is derived from the fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeStage {
    Setup,
    Ready,
}

impl NodeStage {
    /// The projection the editor draws.
    pub fn as_str(self) -> &'static str {
        match self {
            NodeStage::Setup => "setup",
            NodeStage::Ready => "ready",
        }
    }
}

/// What is wrong with a node. Wall-clock `f64` rather than an `Instant`, because a fault is
/// reported over the wire.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NodeFault {
    Setup { msg: String, since: f64, last_attempt: f64 },
    Process { msg: String, since: f64 },
}

impl NodeFault {
    pub fn msg(&self) -> &str {
        match self {
            NodeFault::Setup { msg, .. } | NodeFault::Process { msg, .. } => msg,
        }
    }
}

/// What a node reports about its own health — the vocabulary every engine shares, and the only
/// thing a status drain hands the graph. Every variant is a TRANSITION, so nothing diffs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Status {
    /// Where the node is in its own lifecycle. The graph's `error` stage is DERIVED from a fault
    /// and is never reported here.
    Stage { stage: NodeStage },
    Fault { fault: Option<NodeFault> },
    /// The node's measured update rate (`meta["ufreq"]`) — a measurement, so it alone is paced.
    Ufreq { hz: f64 },
    /// The answer to a refresh request; `None` when the node implements no hook for it.
    RefreshOptions { key: ParamKey, options: Option<Vec<String>> },
    /// Per-binding errors, `None` where one cleared.
    BindingErrors { errors: Vec<(ParamKey, Option<String>)> },
    /// The evaluated values of the node's bound params — the sparse projection, never the record.
    ParamValues { evaluated: Vec<(ParamKey, Param)> },
}
