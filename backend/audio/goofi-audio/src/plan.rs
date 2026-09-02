//! The plan: what the audio thread runs each block, compiled on the control thread from the
//! settled view — an order, the arena's regions, and where every port and param reads.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use goofi_audio_sdk::{BLOCK, MAX_CHANNELS};
use goofi_core::{Param, SlotType};
use goofi_node::{BindingView, GraphView, NodeManifest, ParamDecl, ParamGroups, Uid};

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
    /// An Array input: filled from the node's inbox each block, one sample per sample entered.
    Inbox { at: Region, channels: u16, inbox: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub idx: usize,
    pub ins: Vec<Source>,
    pub params: Vec<Source>,
    pub outs: Vec<(Region, u16)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub stages: Vec<Stage>,
    pub arena_len: usize,
    /// What the device hears: every agreeing `AudioOut`'s input times its gain, summed here.
    pub output: (Region, u16),
    /// The `AudioOut` stages that sum into `output`.
    pub sinks: Vec<usize>,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { stages: Vec::new(), arena_len: BLOCK, output: (SILENCE, 1), sinks: Vec::new() }
    }
}

impl Plan {
    pub fn reads_inbox(&self, idx: usize, inbox: usize) -> bool {
        self.stages
            .iter()
            .any(|s| s.idx == idx && s.ins.iter().any(|i| matches!(i, Source::Inbox { inbox: n, .. } if *n == inbox)))
    }
}

/// A param's scalar as the audio thread reads it: a number as itself, a bool as 0/1, an option
/// as its index, free text as 0.
pub(crate) fn scalar(p: &Param) -> f64 {
    p.as_f64().unwrap_or_else(|| match p {
        Param::Str { value, options: Some(options), .. } => {
            options.iter().position(|o| o == value).map_or(0.0, |i| i as f64)
        }
        _ => 0.0,
    })
}

/// The record's value for one declared param, the declared default where the record has none.
pub(crate) fn param_of(params: &ParamGroups, d: &ParamDecl) -> Param {
    goofi_node::param(params, d.group, d.name).cloned().unwrap_or_else(|| d.spec.to_param())
}

pub(crate) fn scalar_of(params: &ParamGroups, d: &ParamDecl) -> f64 {
    scalar(&param_of(params, d))
}

/// The inbox an Array input reads — its index among the node's Array inputs — and `None` for an
/// audio one.
pub(crate) fn inbox_of(manifest: &NodeManifest, input: usize) -> Option<usize> {
    let array = |s: &goofi_node::SlotDecl| s.kind != SlotType::Audio;
    array(&manifest.inputs[input]).then(|| manifest.inputs[..input].iter().filter(|s| array(s)).count())
}

fn alloc(channels: u16, len: &mut usize) -> Region {
    let at = *len;
    *len += channels as usize * BLOCK;
    at
}

/// One jack's source: silence, its one producer's region, or a sum — which one part also takes
/// when it is this node's own output, so a self-loop never reads and writes one region at once.
fn source_of(parts: Vec<(Region, u16)>, own: &[Region], len: &mut usize) -> Source {
    match parts.as_slice() {
        [] => Source::Silence,
        [(at, channels)] if !own.contains(at) => Source::Region { at: *at, channels: *channels },
        _ => {
            let channels = parts.iter().map(|p| p.1).max().unwrap_or(1);
            Source::Sum { at: alloc(channels, len), channels, parts }
        }
    }
}

/// Whether a binding is a plan edge: a bare reference to a live audio output — a same-engine
/// stream, so the control half never sees it.
pub(crate) fn is_edge(b: &BindingView<'_>, live: &HashMap<Uid, Instance>) -> bool {
    b.live && b.id.is_none() && b.vars.len() == 1 && b.vars[0].wire().is_some_and(|(p, _)| live.contains_key(&p))
}

/// Kahn over port edges and same-engine references, ties by uid. A node whose type answers
/// `feedback()` ignores its in-edges and runs before every other root, reading its producers'
/// regions as the previous block left them. A loop with no such node is excluded and named; what
/// the loop feeds still runs, reading silence at that jack. A `silent` `AudioOut` runs but does
/// not sum.
pub fn compile(view: &GraphView<'_>, live: &HashMap<Uid, Instance>, silent: &[Uid]) -> (Plan, Vec<(Uid, String)>) {
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
            if let Some(b) = bound.filter(|b| is_edge(b, live)) {
                refs.insert((*uid, i), b.vars[0].wire().expect("an edge"));
            }
        }
    }
    let feeds = |consumer: Uid| -> Vec<Uid> {
        let inst = &live[&consumer];
        let mut from: Vec<Uid> = inst
            .manifest
            .inputs
            .iter()
            .filter(|s| s.kind == SlotType::Audio)
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

    let mut plan = Plan::default();
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
            .enumerate()
            .map(|(i, s)| match inbox_of(inst.manifest, i) {
                Some(inbox) => inst.control.chans[inbox].load(Ordering::Relaxed),
                None => parts_of(*uid, s.name, &outs_of).iter().map(|p| p.1).max().unwrap_or(1),
            })
            .collect();
        counts.extend((0..inst.manifest.params.len()).filter_map(|i| refs.get(&(*uid, i)).map(|p| outs_of.get(p).map_or(1, |o| o.1))));
        let scalars: Vec<f64> = inst.manifest.params.iter().map(|d| scalar_of(nv.params, d)).collect();
        let wanted = inst.twin.channels(&counts, &scalars, inst.manifest.outputs.len());
        for (i, o) in inst.manifest.outputs.iter().enumerate() {
            let channels = wanted.get(i).copied().unwrap_or(1).clamp(1, MAX_CHANNELS);
            outs_of.insert((*uid, o.name), (alloc(channels, &mut plan.arena_len), channels));
        }
    }
    for uid in &order {
        let inst = &live[uid];
        let outs: Vec<(Region, u16)> = inst.manifest.outputs.iter().map(|o| outs_of[&(*uid, o.name)]).collect();
        let own: Vec<Region> = outs.iter().map(|o| o.0).collect();
        let ins: Vec<Source> = inst
            .manifest
            .inputs
            .iter()
            .enumerate()
            .map(|(i, s)| match inbox_of(inst.manifest, i) {
                Some(_) if view.wires_into(*uid, s.name).next().is_none() => Source::Silence,
                Some(inbox) => {
                    let channels = inst.control.chans[inbox].load(Ordering::Relaxed);
                    Source::Inbox { at: alloc(channels, &mut plan.arena_len), channels, inbox }
                }
                None => source_of(parts_of(*uid, s.name, &outs_of), &own, &mut plan.arena_len),
            })
            .collect();
        let params: Vec<Source> = (0..inst.manifest.params.len())
            .map(|i| match refs.get(&(*uid, i)).and_then(|p| outs_of.get(p)) {
                Some(part) => source_of(vec![*part], &own, &mut plan.arena_len),
                None => Source::Scalar { at: alloc(1, &mut plan.arena_len), param: i },
            })
            .collect();
        if inst.manifest.type_name == crate::nodes::audio_out::TYPE && !silent.contains(uid) {
            plan.sinks.push(plan.stages.len());
        }
        plan.stages.push(Stage { idx: inst.idx, ins, params, outs });
    }
    let width = plan
        .sinks
        .iter()
        .filter_map(|i| plan.stages[*i].ins.first())
        .map(|s| match s {
            Source::Region { channels, .. } | Source::Sum { channels, .. } | Source::Inbox { channels, .. } => *channels,
            Source::Silence | Source::Scalar { .. } => 1,
        })
        .max()
        .unwrap_or(1);
    plan.output = (alloc(width, &mut plan.arena_len), width);
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
