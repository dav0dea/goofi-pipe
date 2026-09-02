//! What the audio thread owns, and one block of it. Nothing here allocates, locks or blocks:
//! every message is a pointer move, every port is a view into the arena the plan laid out.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use goofi_audio_sdk::{AudioNode, Block, Port, PortMut, BLOCK, MAX_CHANNELS};

use crate::nodes::audio_out;
use crate::plan::{Plan, Source, SILENCE};

/// The most inputs, outputs or params a node may declare: a block's ports are stack arrays.
pub const MAX_PORTS: usize = 16;

pub struct Slot {
    pub node: Box<dyn AudioNode>,
    /// The scalar per param, `f64` bits, written by the node's control half.
    pub params: Arc<[AtomicU64]>,
    /// One per Array input, in declaration order.
    pub inboxes: Vec<Inbox>,
    /// One per output: what the control half publishes to whoever subscribes.
    pub taps: Vec<rtrb::Producer<f32>>,
}

/// The audio thread's end of an Array input: chunks of interleaved samples, each headed by its
/// channel count and length, read one sample per sample.
pub struct Inbox {
    ring: rtrb::Consumer<f32>,
    chans: usize,
    left: usize,
    last: [f32; MAX_CHANNELS as usize],
}

impl Inbox {
    pub fn new(ring: rtrb::Consumer<f32>) -> Inbox {
        Inbox { ring, chans: 0, left: 0, last: [0.0; MAX_CHANNELS as usize] }
    }

    /// Empty the ring and forget the chunk in hand — what the previous producer left.
    fn flush(&mut self) {
        if let Ok(chunk) = self.ring.read_chunk(self.ring.slots()) {
            chunk.commit_all();
        }
        self.chans = 0;
        self.left = 0;
        self.last = [0.0; MAX_CHANNELS as usize];
    }

    /// One block: per channel the next sample entered, or the last one held.
    pub fn fill(&mut self, out: &mut PortMut<'_>) {
        let channels = out.channels();
        for i in 0..BLOCK {
            if self.left == 0 {
                if let Ok(head) = self.ring.read_chunk(2) {
                    let mut head = head.into_iter();
                    self.chans = head.next().unwrap_or(0.0) as usize;
                    self.left = head.next().unwrap_or(0.0) as usize;
                }
            }
            if self.left > 0 {
                match self.ring.read_chunk(self.chans) {
                    Ok(sample) => {
                        for (c, v) in sample.into_iter().enumerate() {
                            self.last[c] = v;
                        }
                        self.left -= 1;
                    }
                    Err(_) => self.left = 0,
                }
            }
            for c in 0..channels as usize {
                out.chan_mut(c)[i] = match (self.chans, c) {
                    (1, _) => self.last[0],
                    (n, c) if c < n => self.last[c],
                    _ => 0.0,
                };
            }
        }
    }
}

pub enum Msg {
    Insert { idx: usize, slot: Slot },
    Remove(usize),
    Plan { plan: Plan, arena: Vec<f32> },
    Grow(Vec<Option<Slot>>),
}

/// What comes back to be dropped off the audio thread.
pub enum Retired {
    Slot(Slot),
    Plan(Plan, Vec<f32>),
    Slab(Vec<Option<Slot>>),
}

pub struct Runtime {
    pub slab: Vec<Option<Slot>>,
    pub plan: Plan,
    pub arena: Vec<f32>,
    pub inbox: rtrb::Consumer<Msg>,
    pub outbox: rtrb::Producer<Retired>,
    /// Rendered output not yet handed to the device, interleaved at `channels()`.
    pub fifo: Vec<f32>,
    /// The device's width while a device is the clock; the summed output's own otherwise.
    device: Option<u16>,
}

impl Runtime {
    pub fn new(slab: usize, inbox: rtrb::Consumer<Msg>, outbox: rtrb::Producer<Retired>) -> Runtime {
        Runtime {
            slab: (0..slab).map(|_| None).collect(),
            plan: Plan::default(),
            arena: vec![0.0; BLOCK],
            inbox,
            outbox,
            fifo: Vec::new(),
            device: None,
        }
    }

    /// The width the FIFO is interleaved at.
    pub fn channels(&self) -> u16 {
        self.device.unwrap_or(self.plan.output.1)
    }

    pub fn set_device(&mut self, channels: Option<u16>) {
        self.device = channels;
        self.fifo.clear();
    }

    /// Fill one device buffer: whole blocks until enough is rendered, the surplus carried.
    pub fn render_into(&mut self, out: &mut [f32]) {
        while self.fifo.len() < out.len() {
            self.render_block();
        }
        let rest = self.fifo.split_off(out.len());
        out.copy_from_slice(&self.fifo);
        self.fifo = rest;
    }

    fn apply(&mut self, msg: Msg) {
        let retired = match msg {
            Msg::Insert { idx, slot } => self.slab[idx].replace(slot).map(Retired::Slot),
            Msg::Remove(idx) => self.slab[idx].take().map(Retired::Slot),
            Msg::Plan { plan, arena } => {
                let old = std::mem::replace(&mut self.plan, plan);
                let old_arena = std::mem::replace(&mut self.arena, arena);
                // An inbox the new plan no longer reads is flushed, or what its last producer
                // left would play first when the input is wired again.
                for stage in &old.stages {
                    for src in &stage.ins {
                        let Source::Inbox { inbox, .. } = src else { continue };
                        if !self.plan.reads_inbox(stage.idx, *inbox) {
                            if let Some(slot) = self.slab[stage.idx].as_mut() {
                                slot.inboxes[*inbox].flush();
                            }
                        }
                    }
                }
                Some(Retired::Plan(old, old_arena))
            }
            Msg::Grow(mut bigger) => {
                for (i, s) in self.slab.iter_mut().enumerate() {
                    bigger[i] = s.take();
                }
                Some(Retired::Slab(std::mem::replace(&mut self.slab, bigger)))
            }
        };
        if let Some(r) = retired {
            let _ = self.outbox.push(r);
        }
    }

    /// Take every message the control half sent; a pointer move each.
    pub fn apply_pending(&mut self) {
        while let Ok(msg) = self.inbox.pop() {
            self.apply(msg);
        }
    }

    /// One block: drain the inbox, run every stage in plan order, append the output to the fifo.
    pub fn render_block(&mut self) {
        self.apply_pending();
        let base = self.arena.as_mut_ptr();
        let len = self.arena.len();
        for stage in &self.plan.stages {
            let Some(slot) = self.slab[stage.idx].as_mut() else { continue };
            for src in stage.params.iter().chain(&stage.ins) {
                match src {
                    Source::Scalar { at, param } => {
                        let v = f64::from_bits(slot.params[*param].load(Ordering::Relaxed)) as f32;
                        let region = unsafe { region_mut(base, len, *at, 1) };
                        if region[0] != v {
                            region.fill(v);
                        }
                    }
                    Source::Sum { at, channels, parts } => {
                        let dst = unsafe { region_mut(base, len, *at, *channels) };
                        dst.fill(0.0);
                        for (part, pc) in parts {
                            let src = Port::new(unsafe { region(base, len, *part, *pc) }, *pc, true);
                            for c in 0..*channels as usize {
                                let from = src.chan(c);
                                for i in 0..BLOCK {
                                    dst[c * BLOCK + i] += from[i];
                                }
                            }
                        }
                    }
                    Source::Inbox { at, channels, inbox } => {
                        let region = unsafe { region_mut(base, len, *at, *channels) };
                        slot.inboxes[*inbox].fill(&mut PortMut::new(region, *channels));
                    }
                    Source::Silence | Source::Region { .. } => {}
                }
            }
            {
                let ins: [Port<'_>; MAX_PORTS] = std::array::from_fn(|i| match stage.ins.get(i) {
                    Some(s) => unsafe { port(base, len, s) },
                    None => Port::new(&[], 0, false),
                });
                let params: [Port<'_>; MAX_PORTS] = std::array::from_fn(|i| match stage.params.get(i) {
                    Some(s) => unsafe { port(base, len, s) },
                    None => Port::new(&[], 0, false),
                });
                let mut outs: [PortMut<'_>; MAX_PORTS] = std::array::from_fn(|i| match stage.outs.get(i) {
                    Some((at, channels)) => PortMut::new(unsafe { region_mut(base, len, *at, *channels) }, *channels),
                    None => PortMut::new(&mut [], 0),
                });
                slot.node.process(&mut Block {
                    ins: &ins[..stage.ins.len()],
                    outs: &mut outs[..stage.outs.len()],
                    params: &params[..stage.params.len()],
                });
            }
            for (k, (at, channels)) in stage.outs.iter().enumerate() {
                let Some(tap) = slot.taps.get_mut(k) else { continue };
                let out = unsafe { region(base, len, *at, *channels) };
                if let Ok(chunk) = tap.write_chunk_uninit(1 + out.len()) {
                    chunk.fill_from_iter(std::iter::once(*channels as f32).chain(out.iter().copied()));
                }
            }
        }
        let (at, channels) = self.plan.output;
        {
            let dst = unsafe { region_mut(base, len, at, channels) };
            dst.fill(0.0);
            for stage in self.plan.sinks.iter().map(|i| &self.plan.stages[*i]) {
                let (Some(input), Some(gain)) = (stage.ins.first(), stage.params.get(audio_out::P::GAIN)) else { continue };
                let (input, gain) = unsafe { (port(base, len, input), port(base, len, gain)) };
                for c in 0..channels as usize {
                    let (x, g) = (input.chan(c), gain.chan(c));
                    for i in 0..BLOCK {
                        dst[c * BLOCK + i] += x[i] * g[i];
                    }
                }
            }
        }
        let out = Port::new(unsafe { region(base, len, at, channels) }, channels, true);
        let width = self.channels() as usize;
        for i in 0..BLOCK {
            for c in 0..width {
                self.fifo.push(out.chan(c)[i]);
            }
        }
    }
}

/// # Safety
/// `at .. at + channels * BLOCK` lies inside the arena, and no live `region_mut` overlaps it —
/// the plan lays every region out disjoint.
unsafe fn region<'a>(base: *mut f32, len: usize, at: usize, channels: u16) -> &'a [f32] {
    let n = channels as usize * BLOCK;
    debug_assert!(at + n <= len);
    std::slice::from_raw_parts(base.add(at), n)
}

/// # Safety
/// As [`region`], and nothing else views this region while the slice lives.
unsafe fn region_mut<'a>(base: *mut f32, len: usize, at: usize, channels: u16) -> &'a mut [f32] {
    let n = channels as usize * BLOCK;
    debug_assert!(at + n <= len);
    std::slice::from_raw_parts_mut(base.add(at), n)
}

/// # Safety
/// As [`region`].
unsafe fn port<'a>(base: *mut f32, len: usize, src: &Source) -> Port<'a> {
    match src {
        Source::Silence => Port::new(region(base, len, SILENCE, 1), 1, false),
        Source::Region { at, channels } | Source::Sum { at, channels, .. } | Source::Inbox { at, channels, .. } => {
            Port::new(region(base, len, *at, *channels), *channels, true)
        }
        Source::Scalar { at, .. } => Port::new(region(base, len, *at, 1), 1, true),
    }
}
