//! The graphics engine behind the `Engine` seam: shader nodes on the GPU, one device, one render
//! thread, no window. The engine (this file) owns the library, the plan and the clock; the render
//! thread (`runtime`) owns the textures and draws; a node's control half (`half`) is a thread of
//! its own, parked on the node's door — the same one the audio engine uses.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use goofi_control::{Desired, Handle, Shared, Sub};
use goofi_core::SlotType;
use goofi_node::{
    DrainWaker, Engine, GraphView, LibraryEntry, NodeFault, NodeStage, NodeView, ParamDecl, ParamGroups, ParamKey,
    ParamSpec, ScannedType, Status, Touched, Uid,
};

pub mod gpu;
mod half;
mod plan;
mod runtime;
mod scan;
mod shader;

use gpu::Gpu;
use half::GraphicsHalf;
use runtime::{Runtime, Stats};
use scan::{Class, Compiler};

/// The engine renders at this pace under its own clock — the rate a display refreshes at, and
/// what a viewer can draw.
const PERIOD: Duration = Duration::from_micros(16_667);

/// What drives the ticks: the harness's `render(frames)`, or a clock of the engine's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clock {
    External,
    Timer,
}

/// The timing door: what the engine is doing, for `session status`.
pub struct GraphicsStatus {
    pub adapter: String,
    pub backend: String,
    pub clock: &'static str,
    pub frames: u64,
    pub stages: u64,
    pub tick_max_us: u64,
}

/// One live node: what the plan reads off it, and the cells its two halves share.
pub(crate) struct Instance {
    pub(crate) class: Arc<Class>,
    pub(crate) params: Arc<[AtomicU64]>,
    pub(crate) uploads: Vec<Arc<Mutex<Option<half::Upload>>>>,
    pub(crate) readers: Arc<AtomicBool>,
    pub(crate) tap: Arc<Mutex<Option<goofi_core::Data>>>,
    control: Handle,
    /// What the control half was last told; a settle that changes nothing says nothing.
    last: Option<Desired>,
}

pub struct GraphicsEngine {
    instance: String,
    started: Instant,
    clock: Clock,
    gpu: Arc<Gpu>,
    shared: Arc<Shared>,
    pub(crate) compiler: Compiler,
    pub(crate) classes: HashMap<String, Arc<Class>>,
    live: HashMap<Uid, Instance>,
    runtime: Arc<Mutex<Runtime>>,
    stats: Arc<Stats>,
    ticker: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>,
    faulted: HashMap<Uid, String>,
    pending: Vec<(Uid, Status)>,
    dirty: bool,
    /// Last: every bell onto a control half is built from it, and fields drop in order.
    bells: goofi_transport::IoxNode,
}

/// The universal group every graphics node carries: 0 follows what is wired behind it.
static OUTPUT_DECLS: &[ParamDecl] = &[
    ParamDecl {
        group: "output",
        name: "width",
        spec: ParamSpec::Int { default: 0, min: 0, max: plan::MAX_SIZE as i64 },
        doc: Some("Texture width in pixels; 0 follows the first wired texture input."),
        expression: None,
    },
    ParamDecl {
        group: "output",
        name: "height",
        spec: ParamSpec::Int { default: 0, min: 0, max: plan::MAX_SIZE as i64 },
        doc: Some("Texture height in pixels; 0 follows the first wired texture input."),
        expression: None,
    },
];

impl GraphicsEngine {
    /// Open the device and start the engine, or say why this machine has none.
    pub fn open(
        instance: String,
        started: Instant,
        waker: Arc<DrainWaker>,
        clock: Clock,
    ) -> Result<GraphicsEngine, String> {
        let gpu = Arc::new(Gpu::open()?);
        let shared = Arc::new(Shared::new(waker));
        let stats = Arc::new(Stats::default());
        let runtime = Arc::new(Mutex::new(Runtime::new(gpu.clone(), started, stats.clone())));
        let ticker = (clock == Clock::Timer).then(|| {
            let stop = Arc::new(AtomicBool::new(false));
            let (rt, halt) = (runtime.clone(), stop.clone());
            let thread = std::thread::Builder::new()
                .name("goofi-graphics-clock".into())
                .spawn(move || {
                    let mut next = Instant::now();
                    while !halt.load(Ordering::Relaxed) {
                        if let Ok(mut rt) = rt.lock() {
                            rt.tick();
                        }
                        next += PERIOD;
                        // A tick that overran does not try to catch up: the next one is now.
                        match next.checked_duration_since(Instant::now()) {
                            Some(left) => std::thread::sleep(left),
                            None => next = Instant::now(),
                        }
                    }
                })
                .map_err(|e| format!("could not start the render clock: {e}"))
                .expect("the render clock");
            (stop, thread)
        });
        Ok(GraphicsEngine {
            instance,
            started,
            clock,
            compiler: Compiler::start(gpu.clone(), shared.clone()),
            gpu,
            shared,
            classes: HashMap::new(),
            live: HashMap::new(),
            runtime,
            stats,
            ticker,
            faulted: HashMap::new(),
            pending: Vec::new(),
            dirty: false,
            bells: goofi_transport::iox_node().expect("an iceoryx2 node for the graphics engine's bells"),
        })
    }

    /// The external clock: run `frames` ticks on the caller's thread. The harness's door.
    pub fn render(&mut self, frames: usize) {
        for _ in 0..frames {
            self.runtime.lock().expect("the runtime").tick();
        }
    }

    pub fn status(&self) -> GraphicsStatus {
        GraphicsStatus {
            adapter: self.gpu.adapter.clone(),
            backend: self.gpu.backend.clone(),
            clock: match self.clock {
                Clock::External => "external",
                Clock::Timer => "timer",
            },
            frames: self.stats.frames.load(Ordering::Relaxed),
            stages: self.stats.stages.load(Ordering::Relaxed),
            tick_max_us: self.stats.tick_max_us.load(Ordering::Relaxed),
        }
    }

    /// Everything one node's control half holds, read off the settled view.
    fn desired_of(&self, view: &GraphView<'_>, uid: Uid, nv: &NodeView<'_>) -> Desired {
        let manifest = self.live[&uid].class.manifest;
        let consts = manifest.params.iter().map(|d| goofi_control::param_of(nv.params, d)).collect();
        let mut subs = Vec::new();
        let mut inbox = 0;
        for s in manifest.inputs {
            if s.kind == SlotType::Texture {
                continue;
            }
            let k = inbox;
            inbox += 1;
            if let Some(service) = view.wires_into(uid, s.name).next().and_then(|(p, slot)| goofi_transport::output_of(view, p, slot)) {
                subs.push(Sub::Slot { inbox: k, service });
            }
        }
        for (param, d) in manifest.params.iter().enumerate() {
            let bound = nv.bindings.iter().find(|b| b.live && b.key.group == d.group && b.key.name == d.name);
            let Some(b) = bound else { continue };
            let vars = b.vars.iter().map(|v| goofi_transport::var_of(view, v)).collect();
            subs.push(Sub::Bind { param, key: b.key.clone(), source: b.rewritten.to_string(), id: b.id, vars });
        }
        let targets = manifest
            .outputs
            .iter()
            .map(|o| {
                view.ringers(uid, o.name)
                    .into_iter()
                    .filter_map(|r| Some((goofi_transport::door_of(view, r.consumer)?, r.event_id)))
                    .collect()
            })
            .collect();
        Desired { consts, subs, targets }
    }

    /// How many ARRAY inputs a node has — one upload cell each.
    fn uploads_of(manifest: &goofi_node::NodeManifest) -> usize {
        manifest.inputs.iter().filter(|s| s.kind != SlotType::Texture).count()
    }
}

impl Engine for GraphicsEngine {
    fn id(&self) -> &'static str {
        "graphics"
    }

    /// A control half is woken by a producer, as an audio one is; the render thread is not.
    fn doorbell_driven(&self) -> bool {
        true
    }

    fn dirty(&self) -> bool {
        self.dirty || self.shared.replan.load(Ordering::Acquire)
    }

    fn library(&self) -> Vec<LibraryEntry> {
        self.classes
            .values()
            .map(|c| LibraryEntry { manifest: c.manifest, isolation: &goofi_node::SHADER })
            .collect()
    }

    fn scan(&mut self, dir: &Path) -> Vec<ScannedType> {
        scan::scan(self, dir)
    }

    fn remove_type(&mut self, type_name: &str) -> bool {
        self.classes.remove(type_name).is_some()
    }

    fn universal_decls(&self, _manifest: &'static goofi_node::NodeManifest) -> Vec<ParamDecl> {
        OUTPUT_DECLS.to_vec()
    }

    fn insert(&mut self, uid: Uid, type_name: &str, generation: u64, params: &ParamGroups) -> Option<String> {
        let Some(class) = self.classes.get(type_name).cloned() else {
            return Some(format!("no graphics node type `{type_name}`"));
        };
        let manifest = class.manifest;
        let atomics: Arc<[AtomicU64]> = manifest
            .params
            .iter()
            .map(|d| AtomicU64::new(goofi_control::scalar_of(params, d).to_bits()))
            .collect();
        let uploads: Vec<Arc<Mutex<Option<half::Upload>>>> =
            (0..Self::uploads_of(manifest)).map(|_| Arc::new(Mutex::new(None))).collect();
        let readers = Arc::new(AtomicBool::new(false));
        let tap = Arc::new(Mutex::new(None));
        let spawn = goofi_control::Spawn {
            engine: "graphics",
            uid,
            base: goofi_transport::service_base(&self.instance, uid, generation),
            manifest,
            params: atomics.clone(),
            started: self.started,
        };
        let (cells, flag, out) = (uploads.clone(), readers.clone(), tap.clone());
        let make = move || GraphicsHalf { uploads: cells, readers: flag, tap: out };
        let control = match goofi_control::spawn(spawn, self.shared.clone(), &self.bells, make) {
            Ok(handle) => handle,
            Err(e) => return Some(e),
        };
        self.runtime.lock().expect("the runtime").insert(uid, runtime::params_len(manifest.params));
        self.live.insert(uid, Instance { class, params: atomics, uploads, readers, tap, control, last: None });
        // A synchronous engine is ready the moment its insert answers.
        self.pending.push((uid, Status::Stage { stage: NodeStage::Ready }));
        self.dirty = true;
        self.shared.waker.notify();
        None
    }

    fn remove(&mut self, uid: Uid) {
        if let Some(inst) = self.live.remove(&uid) {
            inst.control.stop();
            self.runtime.lock().expect("the runtime").remove(uid);
            self.faulted.remove(&uid);
            self.pending.retain(|(u, _)| *u != uid);
            self.dirty = true;
        }
    }

    fn settle(&mut self, view: &GraphView<'_>, _touched: &[Touched]) {
        self.dirty = false;
        self.shared.replan.swap(false, Ordering::Acquire);
        for uid in self.live.keys().copied().collect::<Vec<_>>() {
            let Some(nv) = view.nodes.get(&uid) else { continue };
            let desired = self.desired_of(view, uid, nv);
            if self.live[&uid].last.as_ref() == Some(&desired) {
                continue;
            }
            let inst = self.live.get_mut(&uid).expect("live");
            inst.control.send(desired.clone());
            inst.last = Some(desired);
        }
        let (plan, faults) = plan::compile(view, &self.live);
        let since = self.started.elapsed().as_secs_f64();
        let now: HashMap<Uid, String> = faults.into_iter().collect();
        for uid in self.faulted.keys().filter(|u| !now.contains_key(u)) {
            self.pending.push((*uid, Status::Fault { fault: None }));
        }
        for (uid, msg) in &now {
            if self.faulted.get(uid) != Some(msg) {
                self.pending.push((*uid, Status::Fault { fault: Some(NodeFault::Process { msg: msg.clone(), since }) }));
            }
        }
        self.faulted = now;
        self.runtime.lock().expect("the runtime").set_plan(plan);
        if !self.pending.is_empty() {
            self.shared.waker.notify();
        }
    }

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        let mut pending = std::mem::take(&mut self.pending);
        pending.append(&mut self.shared.reports.lock().expect("the reports").clone());
        self.shared.reports.lock().expect("the reports").clear();
        let n = pending.len();
        for (uid, status) in pending {
            apply(uid, status);
        }
        n
    }

    fn refresh_param(&mut self, uid: Uid, key: ParamKey) {
        if let Some(inst) = self.live.get(&uid) {
            inst.control.refresh(key);
        }
    }

    fn pulse_param(&mut self, uid: Uid, key: ParamKey) {
        if let Some(inst) = self.live.get(&uid) {
            inst.control.pulse(key);
        }
    }

    fn reset_clock(&mut self, origin: Instant) {
        self.started = origin;
        self.runtime.lock().expect("the runtime").reset_clock(origin);
    }

    fn set_evaluator(&mut self, evaluator: Arc<dyn goofi_node::ExprEvaluator>) {
        *self.shared.evaluator.lock().expect("the evaluator") = Some(evaluator);
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Stop the clock, then every control half, and WAIT for each to release its shared memory.
    fn shutdown(&mut self) {
        if let Some((stop, thread)) = self.ticker.take() {
            stop.store(true, Ordering::Relaxed);
            let _ = thread.join();
        }
        let halts: Vec<Arc<goofi_transport::Halt>> = self.live.values().map(|i| i.control.halt.clone()).collect();
        for uid in self.live.keys().copied().collect::<Vec<_>>() {
            self.remove(uid);
        }
        goofi_transport::wait_released(halts.iter().map(|h| &**h), goofi_transport::SHUTDOWN_WAIT);
    }
}

impl Drop for GraphicsEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}
