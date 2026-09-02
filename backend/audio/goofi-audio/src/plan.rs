//! The plan: what the audio thread runs each block, compiled on the control thread from the
//! settled view — an order, the arena's regions, and where every port and param reads.

use std::collections::{HashMap, HashSet};

use goofi_audio_sdk::{BLOCK, MAX_CHANNELS};
use goofi_core::Param;
use goofi_node::{GraphView, ParamDecl, ParamGroups, Uid};

use crate::Instance;

/// An offset into the arena, in floats; a region is `channels * BLOCK` of them.
pub type Region = usize;

/// The silence every unwired input reads: one channel at the arena's start.
pub const SILENCE: Region = 0;

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Silence,
    Region { at: Region, channels: u16 },
    /// A multi input: its wires summed into a scratch region, each read through `Port::chan`'s
    /// one rule for a narrower part.
    Sum { at: Region, channels: u16, parts: Vec<(Region, u16)> },
    /// A scalar-sourced param: its own one-channel region, refilled when its atomic moves.
    Scalar { at: Region, param: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub idx: usize,
    pub ins: Vec<Source>,
    pub params: Vec<Source>,
    pub outs: Vec<(Region, u16)>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Plan {
    pub stages: Vec<Stage>,
    pub arena_len: usize,
    /// What the device hears: the lowest-uid `AudioOut`'s input.
    pub output: Option<Source>,
}

impl Plan {
    /// The region the device reads and its channel count; silence when nothing is heard.
    pub fn heard(&self) -> (Region, u16) {
        match &self.output {
            Some(Source::Region { at, channels }) | Some(Source::Sum { at, channels, .. }) => (*at, *channels),
            _ => (SILENCE, 1),
        }
    }
}

/// A param's scalar as the audio thread reads it: a number as itself, a bool as 0/1, an option
/// as its index, free text as 0.
pub(crate) fn scalar(p: &Param) -> f64 {
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

pub(crate) fn scalar_of(params: &ParamGroups, d: &ParamDecl) -> f64 {
    match params.get(d.group).and_then(|g| g.get(d.name)) {
        Some(p) => scalar(p),
        None => scalar(&d.spec.to_param()),
    }
}

/// Kahn over port edges and same-engine references, ties by uid. A node whose type answers
/// `feedback()` ignores its in-edges and runs before every other root, reading its producers'
/// regions as the previous block left them. A loop with no such node is excluded and named; what
/// the loop feeds still runs, reading silence at that jack.
pub fn compile(view: &GraphView<'_>, live: &HashMap<Uid, Instance>) -> (Plan, Vec<(Uid, String)>) {
    let mut wires: HashMap<(Uid, &str), Vec<(Uid, &'static str)>> = HashMap::new();
    for e in view.edges {
        if live.contains_key(&e.consumer.0) && live.contains_key(&e.producer.0) {
            wires.entry(e.consumer).or_default().push(e.producer);
        }
    }
    let mut refs: HashMap<(Uid, usize), (Uid, &'static str)> = HashMap::new();
    for (uid, inst) in live {
        let Some(nv) = view.nodes.get(uid) else { continue };
        for (i, d) in inst.manifest.params.iter().enumerate() {
            let bound = nv.bindings.iter().find(|b| b.key.group == d.group && b.key.name == d.name);
            let Some(b) = bound else { continue };
            if !b.live || b.id.is_some() || b.vars.len() != 1 {
                continue;
            }
            if let Some(p) = b.vars[0].wire().filter(|(uid, _)| live.contains_key(uid)) {
                refs.insert((*uid, i), p);
            }
        }
    }
    let feeds = |consumer: Uid| -> Vec<Uid> {
        let inst = &live[&consumer];
        let mut from: Vec<Uid> = inst
            .manifest
            .inputs
            .iter()
            .flat_map(|s| wires.get(&(consumer, s.name)).into_iter().flatten().map(|p| p.0))
            .collect();
        from.extend((0..inst.manifest.params.len()).filter_map(|i| refs.get(&(consumer, i)).map(|p| p.0)));
        from
    };
    let inbound: HashMap<Uid, Vec<Uid>> = live
        .iter()
        .map(|(uid, inst)| (*uid, if inst.twin.feedback() { Vec::new() } else { feeds(*uid) }))
        .collect();
    let (order, stuck) = kahn(live, &inbound, &HashSet::new());
    // A node Kahn could not order is IN a loop when it reaches itself; the rest are only fed by one.
    let members: HashSet<Uid> = stuck.iter().copied().filter(|u| reaches_itself(*u, &inbound, &stuck)).collect();
    let (order, _) = if members.is_empty() { (order, stuck) } else { kahn(live, &inbound, &members) };
    let faults: Vec<(Uid, String)> =
        members.iter().map(|u| (*u, "in a loop with no feedback node, so it does not run".to_string())).collect();

    let mut plan = Plan { stages: Vec::new(), arena_len: BLOCK, output: None };
    let alloc = |channels: u16, len: &mut usize| -> Region {
        let at = *len;
        *len += channels as usize * BLOCK;
        at
    };
    let parts_of = |uid: Uid, slot: &str, outs_of: &HashMap<(Uid, &'static str), (Region, u16)>| -> Vec<(Region, u16)> {
        wires.get(&(uid, slot)).into_iter().flatten().filter_map(|p| outs_of.get(p).copied()).collect()
    };
    // Pass one lays out every output, so a feedback node's loop in-edge finds its producer's
    // region in pass two; a producer not yet laid out counts as one channel.
    let mut outs_of: HashMap<(Uid, &'static str), (Region, u16)> = HashMap::new();
    for uid in &order {
        let inst = &live[uid];
        let nv = &view.nodes[uid];
        let mut counts: Vec<u16> = inst
            .manifest
            .inputs
            .iter()
            .map(|s| parts_of(*uid, s.name, &outs_of).iter().map(|p| p.1).max().unwrap_or(1))
            .collect();
        counts.extend((0..inst.manifest.params.len()).filter_map(|i| refs.get(&(*uid, i)).map(|p| outs_of.get(p).map_or(1, |o| o.1))));
        let scalars: Vec<f64> = inst.manifest.params.iter().map(|d| scalar_of(nv.params, d)).collect();
        let wanted = inst.twin.channels(&counts, &scalars, inst.manifest.outputs.len());
        for (i, o) in inst.manifest.outputs.iter().enumerate() {
            let channels = wanted.get(i).copied().unwrap_or(1).clamp(1, MAX_CHANNELS);
            outs_of.insert((*uid, o.name), (alloc(channels, &mut plan.arena_len), channels));
        }
    }
    let mut heard: Option<(Uid, Source)> = None;
    for uid in &order {
        let inst = &live[uid];
        let ins: Vec<Source> = inst
            .manifest
            .inputs
            .iter()
            .map(|s| {
                let parts = parts_of(*uid, s.name, &outs_of);
                match parts.as_slice() {
                    [] => Source::Silence,
                    [(at, channels)] => Source::Region { at: *at, channels: *channels },
                    _ => {
                        let channels = parts.iter().map(|p| p.1).max().unwrap_or(1);
                        Source::Sum { at: alloc(channels, &mut plan.arena_len), channels, parts }
                    }
                }
            })
            .collect();
        let params: Vec<Source> = (0..inst.manifest.params.len())
            .map(|i| match refs.get(&(*uid, i)).and_then(|p| outs_of.get(p)) {
                Some((at, channels)) => Source::Region { at: *at, channels: *channels },
                None => Source::Scalar { at: alloc(1, &mut plan.arena_len), param: i },
            })
            .collect();
        let outs: Vec<(Region, u16)> = inst.manifest.outputs.iter().map(|o| outs_of[&(*uid, o.name)]).collect();
        if inst.manifest.type_name == "AudioOut" && heard.as_ref().is_none_or(|(h, _)| uid.0 < h.0) {
            heard = ins.first().map(|s| (*uid, s.clone()));
        }
        plan.stages.push(Stage { idx: inst.idx, ins, params, outs });
    }
    plan.output = heard.map(|(_, s)| s);
    (plan, faults)
}

/// The order Kahn finds — feedback nodes first, then by uid — and the nodes it could not place.
/// Edges out of `dropped` nodes do not count, so what a loop feeds is placed on silence.
fn kahn(live: &HashMap<Uid, Instance>, inbound: &HashMap<Uid, Vec<Uid>>, dropped: &HashSet<Uid>) -> (Vec<Uid>, Vec<Uid>) {
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
        ready.sort_by_key(|u| std::cmp::Reverse((!live[u].twin.feedback(), u.0)));
        let u = ready.pop().unwrap();
        order.push(u);
        for s in successors.get(&u).into_iter().flatten() {
            let d = indegree.get_mut(s).unwrap();
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
    let mut stack: Vec<Uid> = inbound.get(&start).into_iter().flatten().copied().filter(|p| within.contains(p)).collect();
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
