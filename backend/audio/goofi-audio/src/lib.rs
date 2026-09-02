//! The audio engine behind the `Engine` seam: synchronous, in-process, one 64-frame block per
//! callback. The control half (this file) owns the library, the slab indices and the plan; the
//! audio half (`runtime`) owns the instances and the arena, and hears from here by message.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use goofi_audio_sdk::AudioNode;
use goofi_node::{
    DrainWaker, Engine, GraphView, LibraryEntry, NodeFault, NodeManifest, NodeStage, ParamGroups, Request, Status,
    Touched, Uid, NATIVE,
};

pub mod nodes;
mod plan;
mod runtime;

use plan::Plan;
use runtime::{Msg, Retired, Runtime, Slot, MAX_PORTS};

/// The rate until a device names one (Step 6 of the audio program).
pub const RATE: f64 = 48_000.0;

pub(crate) struct Instance {
    pub(crate) idx: usize,
    pub(crate) manifest: &'static NodeManifest,
    /// Answers `channels` on the control thread; the box that processes never leaves the audio
    /// thread.
    pub(crate) twin: Box<dyn AudioNode>,
    params: Arc<[AtomicU64]>,
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
        while rt.fifo.len() < frames * rt.plan.heard().1 as usize {
            rt.render_block();
        }
        let channels = rt.plan.heard().1;
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

    /// A message always lands. A full ring means the audio thread has not run for a long time, so
    /// taking its lock to apply the backlog costs no block.
    fn send(&mut self, mut msg: Msg) {
        while let Err(rtrb::PushError::Full(back)) = self.inbox.push(msg) {
            self.runtime.lock().unwrap_or_else(|e| e.into_inner()).apply_pending();
            msg = back;
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
        let atomics: Arc<[AtomicU64]> =
            manifest.params.iter().map(|d| AtomicU64::new(plan::scalar_of(params, d).to_bits())).collect();
        let idx = self.slot_index();
        self.send(Msg::Insert { idx, slot: Slot { node, params: atomics.clone() } });
        self.live.insert(uid, Instance { idx, manifest, twin: make(), params: atomics });
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
        for (uid, inst) in &self.live {
            let Some(nv) = view.nodes.get(uid) else { continue };
            for (i, d) in inst.manifest.params.iter().enumerate() {
                inst.params[i].store(plan::scalar_of(nv.params, d).to_bits(), Ordering::Relaxed);
            }
        }
        let (plan, faults) = plan::compile(view, &self.live);
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
