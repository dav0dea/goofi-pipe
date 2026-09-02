//! The audio engine behind the `Engine` seam: synchronous, in-process, one 64-frame block per
//! callback. The control half (this file) owns the library, the slab indices and the plan; the
//! audio half (`runtime`) owns the instances and the arena, and hears from here by message.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use goofi_audio_sdk::AudioNode;
use goofi_core::Param;
use goofi_node::{
    DrainWaker, Engine, GraphView, LibraryEntry, NodeFault, NodeManifest, NodeStage, ParamDecl, ParamGroups, Request,
    Status, Touched, Uid, NATIVE,
};

pub mod nodes;
mod plan;
mod runtime;

use plan::Plan;
use runtime::{Msg, Retired, Runtime, Slot, MAX_PORTS};

/// The rate until a device names one (Step 6 of the audio program).
pub const RATE: f64 = 48_000.0;

struct Instance {
    idx: usize,
    manifest: &'static NodeManifest,
    twin: Box<dyn AudioNode>,
    params: Arc<[AtomicU64]>,
    scalars: Vec<f64>,
}

pub struct AudioEngine {
    started: Instant,
    waker: Arc<DrainWaker>,
    classes: Vec<(&'static NodeManifest, nodes::Make)>,
    runtime: Arc<Mutex<Runtime>>,
    inbox: rtrb::Producer<Msg>,
    outbox: rtrb::Consumer<Retired>,
    free: Vec<usize>,
    slab_len: usize,
    live: HashMap<Uid, Instance>,
    faulted: Vec<Uid>,
    pending: Vec<(Uid, Status)>,
    dirty: bool,
    last: Plan,
}

const SLAB: usize = 64;
const QUEUE: usize = 4096;

impl AudioEngine {
    pub fn new(started: Instant, waker: Arc<DrainWaker>) -> AudioEngine {
        let classes = nodes::SHIPPED
            .iter()
            .map(|(type_name, m, make)| {
                let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
                    type_name,
                    category: m.category,
                    doc: m.doc,
                    inputs: m.inputs,
                    outputs: m.outputs,
                    params: m.params,
                    producer: false,
                }));
                (manifest, *make)
            })
            .collect();
        let (inbox, to_audio) = rtrb::RingBuffer::new(QUEUE);
        let (from_audio, outbox) = rtrb::RingBuffer::new(QUEUE);
        AudioEngine {
            started,
            waker,
            classes,
            runtime: Arc::new(Mutex::new(Runtime::new(SLAB, to_audio, from_audio))),
            inbox,
            outbox,
            free: (0..SLAB).rev().collect(),
            slab_len: SLAB,
            live: HashMap::new(),
            faulted: Vec::new(),
            pending: Vec::new(),
            dirty: false,
            last: Plan::default(),
        }
    }

    /// The external clock: render whole blocks until `frames` are ready, and hand them over
    /// interleaved — exactly what a device callback would receive.
    pub fn drive(&mut self, frames: usize) -> (Vec<f32>, u16) {
        let mut rt = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        rt.fifo.reserve(frames * goofi_audio_sdk::MAX_CHANNELS as usize);
        while rt.fifo.len() < frames * rt.channels as usize {
            rt.render_block();
        }
        let channels = rt.channels;
        let rest = rt.fifo.split_off(frames * channels as usize);
        let out = std::mem::replace(&mut rt.fifo, rest);
        (out, channels)
    }

    /// What the audio thread handed back is dropped here, where dropping may take time.
    fn discard_retired(&mut self) {
        while let Ok(retired) = self.outbox.pop() {
            match retired {
                Retired::Slot(slot) => drop(slot),
                Retired::Plan(plan, arena) => drop((plan, arena)),
                Retired::Slab(slab) => drop(slab),
            }
        }
    }

    fn send(&mut self, msg: Msg) {
        if self.inbox.push(msg).is_err() {
            eprintln!("audio engine: the inbox is full — a block has not been rendered in a long time");
        }
    }

    fn slot_index(&mut self) -> usize {
        if let Some(idx) = self.free.pop() {
            return idx;
        }
        let bigger = self.slab_len * 2;
        self.send(Msg::Grow((0..bigger).map(|_| None).collect()));
        self.free.extend((self.slab_len + 1..bigger).rev());
        let idx = self.slab_len;
        self.slab_len = bigger;
        idx
    }
}

/// A param's scalar as the audio thread reads it: a number as itself, a bool as 0/1, an option
/// as its index, free text as 0.
fn scalar(p: &Param) -> f64 {
    match p {
        Param::Float { value, .. } => *value,
        Param::Int { value, .. } => *value as f64,
        Param::Bool { value } => f64::from(u8::from(*value)),
        Param::Str { value, options: Some(options), .. } => {
            options.iter().position(|o| o == value).map_or(0.0, |i| i as f64)
        }
        Param::Str { .. } => 0.0,
    }
}

fn scalar_of(params: &ParamGroups, d: &ParamDecl) -> f64 {
    match params.get(d.group).and_then(|g| g.get(d.name)) {
        Some(p) => scalar(p),
        None => scalar(&d.spec.to_param()),
    }
}

impl Engine for AudioEngine {
    fn id(&self) -> &'static str {
        "audio"
    }

    fn doorbell_driven(&self) -> bool {
        false
    }

    fn dirty(&self) -> bool {
        self.dirty
    }

    fn library(&self) -> Vec<LibraryEntry> {
        self.classes.iter().map(|(manifest, _)| LibraryEntry { manifest, isolation: &NATIVE }).collect()
    }

    fn rust_sdk(&self) -> Option<&'static str> {
        Some("goofi-audio-sdk")
    }

    fn normalize_params(&self, type_name: &str, supplied: Option<ParamGroups>) -> Result<ParamGroups, String> {
        let (manifest, _) = self
            .classes
            .iter()
            .find(|(m, _)| m.type_name == type_name)
            .ok_or_else(|| format!("no node type `{type_name}` in the audio library"))?;
        let mut params = manifest.default_params();
        for (group, entries) in supplied.into_iter().flatten() {
            params.entry(group).or_default().extend(entries);
        }
        Ok(params)
    }

    fn insert(&mut self, uid: Uid, type_name: &str, _generation: u64, params: &ParamGroups) -> Option<String> {
        let Some((manifest, make)) = self.classes.iter().find(|(m, _)| m.type_name == type_name).copied() else {
            return Some(format!("no audio node type `{type_name}`"));
        };
        let widest = manifest.params.len().max(manifest.inputs.len()).max(manifest.outputs.len());
        if widest > MAX_PORTS {
            return Some(format!("`{type_name}` declares more than {MAX_PORTS} ports"));
        }
        let mut node = make();
        node.prepare(RATE);
        let scalars: Vec<f64> = manifest.params.iter().map(|d| scalar_of(params, d)).collect();
        let atomics: Arc<[AtomicU64]> = scalars.iter().map(|v| AtomicU64::new(v.to_bits())).collect();
        let idx = self.slot_index();
        self.send(Msg::Insert {
            idx,
            slot: Slot { node, params: atomics.clone(), last: vec![f64::NAN; scalars.len()] },
        });
        self.live.insert(uid, Instance { idx, manifest, twin: make(), params: atomics, scalars });
        self.pending.push((uid, Status::Stage { stage: NodeStage::Ready }));
        self.dirty = true;
        self.waker.notify();
        None
    }

    fn remove(&mut self, uid: Uid) {
        if let Some(inst) = self.live.remove(&uid) {
            self.send(Msg::Remove(inst.idx));
            self.free.push(inst.idx);
            self.faulted.retain(|u| *u != uid);
            self.pending.retain(|(u, _)| *u != uid);
            self.dirty = true;
        }
    }

    fn settle(&mut self, view: &GraphView<'_>, _touched: &[Touched]) {
        self.dirty = false;
        for (uid, nv) in &view.nodes {
            if nv.engine != "audio" {
                continue;
            }
            let Some(inst) = self.live.get_mut(uid) else { continue };
            for (i, d) in inst.manifest.params.iter().enumerate() {
                let v = scalar_of(nv.params, d);
                inst.scalars[i] = v;
                inst.params[i].store(v.to_bits(), Ordering::Relaxed);
            }
        }
        let nodes: Vec<plan::Node<'_>> = view
            .nodes
            .iter()
            .filter(|(_, nv)| nv.engine == "audio")
            .filter_map(|(uid, _)| {
                self.live.get(uid).map(|i| plan::Node {
                    uid: *uid,
                    idx: i.idx,
                    manifest: i.manifest,
                    twin: i.twin.as_ref(),
                    scalars: &i.scalars,
                })
            })
            .collect();
        let (plan, faults) = plan::compile(view, &nodes);
        let since = self.started.elapsed().as_secs_f64();
        let now_faulted: Vec<Uid> = faults.iter().map(|(u, _)| *u).collect();
        for uid in self.faulted.iter().filter(|u| !now_faulted.contains(u)) {
            self.pending.push((*uid, Status::Fault { fault: None }));
        }
        for (uid, msg) in faults {
            if !self.faulted.contains(&uid) {
                self.pending.push((uid, Status::Fault { fault: Some(NodeFault::Process { msg, since }) }));
            }
        }
        self.faulted = now_faulted;
        if plan != self.last {
            let arena = vec![0.0; plan.arena_len];
            self.send(Msg::Plan { plan: plan.clone(), arena });
            self.last = plan;
        }
        if !self.pending.is_empty() {
            self.waker.notify();
        }
    }

    fn drain(&mut self, apply: &mut dyn FnMut(Uid, Status)) -> usize {
        self.discard_retired();
        let pending = std::mem::take(&mut self.pending);
        let n = pending.len();
        for (uid, status) in pending {
            apply(uid, status);
        }
        n
    }

    fn request(&mut self, _uid: Uid, _request: Request) {}

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn shutdown(&mut self) {
        for uid in self.live.keys().copied().collect::<Vec<_>>() {
            self.remove(uid);
        }
        if let Ok(mut rt) = self.runtime.lock() {
            rt.render_block();
        }
        self.discard_retired();
    }
}
