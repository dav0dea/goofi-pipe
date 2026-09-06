//! The plan: what the render thread runs each tick, compiled at settle from the settled view —
//! an order, each stage's inputs and size, and the cells the tick reads demand from and writes
//! its readback into.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use goofi_core::{Data, SlotType};
use goofi_node::{GraphView, ParamDecl, Uid};

use crate::half::Upload;
use crate::scan::Built;
use crate::Instance;

/// A generator's size, and what a chain that can follow nothing falls back to.
pub const GENERATOR: u32 = 512;
/// The widest a node may ask for on either axis.
pub const MAX_SIZE: u32 = 8192;

pub enum Input {
    /// Another stage's output, by index into `Plan::stages`.
    Stage(usize),
    /// The k-th ARRAY input's upload cell.
    Upload(usize),
    /// Nothing is wired: the shared transparent texture.
    None,
}

pub struct Stage {
    pub uid: Uid,
    pub pipeline: Built,
    pub inputs: Vec<Input>,
    pub size: (u32, u32),
    pub decls: &'static [ParamDecl],
    pub params: Arc<[AtomicU64]>,
    pub uploads: Vec<Arc<Mutex<Option<Upload>>>>,
    /// Whether anyone drinks from this output right now — the half writes it each tick.
    pub readers: Arc<AtomicBool>,
    /// Where the tick leaves the frame it read back, for the half to publish.
    pub tap: Arc<Mutex<Option<Data>>>,
}

#[derive(Default)]
pub struct Plan {
    pub stages: Vec<Stage>,
}

impl Plan {
    /// Which stages render this tick: every stage a reader reaches backwards over the edges. A
    /// node nobody reads costs nothing, which is what makes a big idle patch free.
    pub fn demanded(&self) -> Vec<bool> {
        let mut want = vec![false; self.stages.len()];
        let mut stack: Vec<usize> =
            (0..self.stages.len()).filter(|&i| self.stages[i].readers.load(Ordering::Relaxed)).collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut want[i], true) {
                continue;
            }
            for input in &self.stages[i].inputs {
                if let Input::Stage(j) = input {
                    stack.push(*j);
                }
            }
        }
        want
    }
}

/// The order, the sizes and the wiring, from settled state. A node whose class answers `feedback`
/// ignores its in-edges and runs first, reading its producer's texture as the last tick left it;
/// a loop with no such node is excluded and named.
pub fn compile(view: &GraphView<'_>, live: &HashMap<Uid, Instance>) -> (Plan, Vec<(Uid, String)>) {
    let mut wires: HashMap<(Uid, &str), Uid> = HashMap::new();
    for e in view.edges {
        if live.contains_key(&e.consumer.0) && live.contains_key(&e.producer.0) {
            wires.insert(e.consumer, e.producer.0);
        }
    }
    let feeds = |consumer: Uid| -> Vec<Uid> {
        let inst = &live[&consumer];
        inst.class
            .manifest
            .inputs
            .iter()
            .filter(|s| s.kind == SlotType::Texture)
            .filter_map(|s| wires.get(&(consumer, s.name)).copied())
            .collect()
    };
    let inbound: HashMap<Uid, Vec<Uid>> =
        live.iter().map(|(uid, i)| (*uid, if i.class.feedback { Vec::new() } else { feeds(*uid) })).collect();
    let (order, stuck) = kahn(live, &inbound, &HashSet::new());
    // A node Kahn could not place is IN a loop when it reaches itself; the rest are only fed by one.
    let members: HashSet<Uid> = stuck.iter().copied().filter(|u| reaches_itself(*u, &inbound, &stuck)).collect();
    let (order, _) = if members.is_empty() { (order, stuck) } else { kahn(live, &inbound, &members) };
    let mut faults: Vec<(Uid, String)> = members
        .iter()
        .map(|u| (*u, "in a loop with no feedback node, so it does not render".to_string()))
        .collect();

    let mut sizes: HashMap<Uid, (u32, u32)> = HashMap::new();
    for uid in &order {
        size_of(*uid, view, live, &wires, &mut sizes, &mut Vec::new());
    }
    let at: HashMap<Uid, usize> = order.iter().enumerate().map(|(i, u)| (*u, i)).collect();
    let mut stages = Vec::with_capacity(order.len());
    for uid in &order {
        let inst = &live[uid];
        if let Some(Err(why)) = inst.class.pipeline.get() {
            faults.push((*uid, format!("shader: {why}")));
        }
        let mut upload = 0;
        let inputs = inst
            .class
            .manifest
            .inputs
            .iter()
            .map(|s| match s.kind {
                SlotType::Texture => wires.get(&(*uid, s.name)).and_then(|p| at.get(p)).map_or(Input::None, |i| Input::Stage(*i)),
                _ => {
                    let k = upload;
                    upload += 1;
                    // From the settled view, not from an unlink event: an unwired ARRAY input is
                    // transparent black, and the texture its last frame made is not the answer.
                    match view.wires_into(*uid, s.name).next() {
                        Some(_) => Input::Upload(k),
                        None => Input::None,
                    }
                }
            })
            .collect();
        stages.push(Stage {
            uid: *uid,
            pipeline: inst.class.pipeline.clone(),
            inputs,
            size: sizes[uid],
            decls: inst.class.manifest.params,
            params: inst.params.clone(),
            uploads: inst.uploads.clone(),
            readers: inst.readers.clone(),
            tap: inst.tap.clone(),
        });
    }
    (Plan { stages }, faults)
}

/// A node's size: what `output/width` and `output/height` say, and for a zero on an axis the
/// first wired texture input's size on that axis. A chain that follows itself, or one that
/// follows nothing, is a generator.
fn size_of(
    uid: Uid,
    view: &GraphView<'_>,
    live: &HashMap<Uid, Instance>,
    wires: &HashMap<(Uid, &str), Uid>,
    sizes: &mut HashMap<Uid, (u32, u32)>,
    visiting: &mut Vec<Uid>,
) -> (u32, u32) {
    if let Some(known) = sizes.get(&uid) {
        return *known;
    }
    let asked = |name: &str| -> u32 {
        view.nodes
            .get(&uid)
            .and_then(|nv| goofi_node::param(nv.params, "output", name))
            .and_then(|p| p.as_f64())
            .map_or(0, |v| v.max(0.0).min(MAX_SIZE as f64) as u32)
    };
    let (w, h) = (asked("width"), asked("height"));
    let mut answer = (w, h);
    // A chain that follows ITSELF cannot answer; what it asked for on either axis still stands.
    if visiting.contains(&uid) {
        return (if w == 0 { GENERATOR } else { w }, if h == 0 { GENERATOR } else { h });
    }
    visiting.push(uid);
    if w == 0 || h == 0 {
        let behind = live.get(&uid).and_then(|inst| {
            inst.class
                .manifest
                .inputs
                .iter()
                .filter(|s| s.kind == SlotType::Texture)
                .find_map(|s| wires.get(&(uid, s.name)).copied())
        });
        let (fw, fh) = match behind {
            Some(p) => size_of(p, view, live, wires, sizes, visiting),
            None => (GENERATOR, GENERATOR),
        };
        answer = (if w == 0 { fw } else { w }, if h == 0 { fh } else { h });
    }
    visiting.pop();
    sizes.insert(uid, answer);
    answer
}

/// The order Kahn finds — feedback nodes first, then by uid — and what it could not place.
fn kahn(
    live: &HashMap<Uid, Instance>,
    inbound: &HashMap<Uid, Vec<Uid>>,
    dropped: &HashSet<Uid>,
) -> (Vec<Uid>, Vec<Uid>) {
    let mut indegree: HashMap<Uid, usize> = HashMap::new();
    let mut successors: HashMap<Uid, Vec<Uid>> = HashMap::new();
    for (uid, from) in inbound {
        if dropped.contains(uid) {
            continue;
        }
        let from: Vec<Uid> = from.iter().copied().filter(|p| !dropped.contains(p)).collect();
        indegree.insert(*uid, from.len());
        for p in from {
            successors.entry(p).or_default().push(*uid);
        }
    }
    let mut order: Vec<Uid> = Vec::with_capacity(live.len());
    let mut ready: Vec<Uid> = indegree.iter().filter(|(_, d)| **d == 0).map(|(u, _)| *u).collect();
    while !ready.is_empty() {
        ready.sort_by_key(|u| std::cmp::Reverse((!live[u].class.feedback, u.0)));
        let u = ready.pop().expect("not empty");
        order.push(u);
        for s in successors.get(&u).into_iter().flatten() {
            let d = indegree.get_mut(s).expect("a successor is in the graph");
            *d -= 1;
            if *d == 0 {
                ready.push(*s);
            }
        }
    }
    let stuck: Vec<Uid> = indegree.keys().filter(|u| !order.contains(u)).copied().collect();
    (order, stuck)
}

fn reaches_itself(start: Uid, inbound: &HashMap<Uid, Vec<Uid>>, within: &[Uid]) -> bool {
    let mut seen: HashSet<Uid> = HashSet::new();
    let mut stack: Vec<Uid> =
        inbound.get(&start).into_iter().flatten().copied().filter(|p| within.contains(p)).collect();
    while let Some(u) = stack.pop() {
        if u == start {
            return true;
        }
        if seen.insert(u) {
            stack.extend(inbound.get(&u).into_iter().flatten().copied().filter(|p| within.contains(p)));
        }
    }
    false
}
