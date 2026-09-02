//! The plan: what the audio thread runs each block, compiled on the control thread from the
//! settled view — an order, the arena's regions, and where every port and param reads.

use std::collections::HashMap;

use goofi_audio_sdk::{AudioNode, BLOCK};
use goofi_node::{GraphView, NodeManifest, Uid};

/// An offset into the arena, in floats; a region is `channels * BLOCK` of them.
pub type Region = usize;

/// The silence every unwired input reads: one channel at the arena's start.
pub const SILENCE: Region = 0;

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Silence,
    Region { at: Region, channels: u16 },
    /// A multi input: its wires summed into a scratch region — a one-channel wire on every
    /// channel, a narrower one padded with silence.
    Sum { at: Region, channels: u16, parts: Vec<(Region, u16)> },
    /// A scalar-sourced param: its own one-channel region, refilled when its atomic moves.
    Scalar { at: Region, param: usize },
}

impl Source {
    pub fn channels(&self) -> u16 {
        match self {
            Source::Silence | Source::Scalar { .. } => 1,
            Source::Region { channels, .. } | Source::Sum { channels, .. } => *channels,
        }
    }
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
    /// What the device hears: the first `AudioOut`'s input, by uid.
    pub output: Option<Source>,
}

/// One live audio node as the compiler sees it.
pub struct Node<'a> {
    pub uid: Uid,
    pub idx: usize,
    pub manifest: &'static NodeManifest,
    /// Answers `channels` on the control thread; the box that processes never leaves the audio
    /// thread.
    pub twin: &'a dyn AudioNode,
    pub scalars: &'a [f64],
}

/// Kahn over port edges and same-engine references, ties by uid. A node whose type answers
/// `feedback()` ignores its in-edges and runs first; a loop with no such node is excluded and
/// named.
pub fn compile(view: &GraphView<'_>, nodes: &[Node<'_>]) -> (Plan, Vec<(Uid, String)>) {
    let by_uid: HashMap<Uid, &Node<'_>> = nodes.iter().map(|n| (n.uid, n)).collect();
    let mut wires: HashMap<(Uid, &str), Vec<(Uid, &'static str)>> = HashMap::new();
    for e in view.edges {
        if by_uid.contains_key(&e.consumer.0) && by_uid.contains_key(&e.producer.0) {
            wires.entry(e.consumer).or_default().push(e.producer);
        }
    }
    let mut refs: HashMap<(Uid, usize), (Uid, &'static str)> = HashMap::new();
    for n in nodes {
        let Some(nv) = view.nodes.get(&n.uid) else { continue };
        for (i, d) in n.manifest.params.iter().enumerate() {
            let bound = nv.bindings.iter().find(|b| b.key.group == d.group && b.key.name == d.name);
            let Some(b) = bound else { continue };
            if b.id.is_some() || b.vars.len() != 1 {
                continue;
            }
            if let Some(p) = b.vars[0].wire().filter(|(uid, _)| by_uid.contains_key(uid)) {
                refs.insert((n.uid, i), p);
            }
        }
    }
    let feeds = |consumer: Uid| -> Vec<Uid> {
        let n = by_uid[&consumer];
        let mut from: Vec<Uid> = n.manifest.inputs.iter().flat_map(|s| wires.get(&(consumer, s.name)).into_iter().flatten().map(|p| p.0)).collect();
        from.extend((0..n.manifest.params.len()).filter_map(|i| refs.get(&(consumer, i)).map(|p| p.0)));
        from
    };
    let mut indegree: HashMap<Uid, usize> = HashMap::new();
    let mut successors: HashMap<Uid, Vec<Uid>> = HashMap::new();
    for n in nodes {
        let inbound = if n.twin.feedback() { Vec::new() } else { feeds(n.uid) };
        indegree.insert(n.uid, inbound.len());
        for p in inbound {
            successors.entry(p).or_default().push(n.uid);
        }
    }
    let mut order: Vec<Uid> = Vec::with_capacity(nodes.len());
    let mut ready: Vec<Uid> = indegree.iter().filter(|(_, d)| **d == 0).map(|(u, _)| *u).collect();
    while !ready.is_empty() {
        ready.sort_by_key(|u| std::cmp::Reverse(u.0));
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
    let faults: Vec<(Uid, String)> = nodes
        .iter()
        .filter(|n| !order.contains(&n.uid))
        .map(|n| (n.uid, "in a loop with no feedback node, so it does not run".to_string()))
        .collect();

    let mut plan = Plan { stages: Vec::new(), arena_len: BLOCK, output: None };
    let alloc = |channels: u16, len: &mut usize| -> Region {
        let at = *len;
        *len += channels as usize * BLOCK;
        at
    };
    let mut outs_of: HashMap<(Uid, &'static str), (Region, u16)> = HashMap::new();
    for uid in order {
        let n = by_uid[&uid];
        let ins: Vec<Source> = n
            .manifest
            .inputs
            .iter()
            .map(|s| {
                let parts: Vec<(Region, u16)> = wires
                    .get(&(uid, s.name))
                    .into_iter()
                    .flatten()
                    .filter_map(|p| outs_of.get(p).copied())
                    .collect();
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
        let params: Vec<Source> = (0..n.manifest.params.len())
            .map(|i| match refs.get(&(uid, i)).and_then(|p| outs_of.get(p)) {
                Some((at, channels)) => Source::Region { at: *at, channels: *channels },
                None => Source::Scalar { at: alloc(1, &mut plan.arena_len), param: i },
            })
            .collect();
        let mut counts: Vec<u16> = ins.iter().map(Source::channels).collect();
        counts.extend(params.iter().filter(|p| matches!(p, Source::Region { .. })).map(Source::channels));
        let wanted = n.twin.channels(&counts, n.scalars, n.manifest.outputs.len());
        let outs: Vec<(Region, u16)> = n
            .manifest
            .outputs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let channels = wanted.get(i).copied().unwrap_or(1).clamp(1, goofi_audio_sdk::MAX_CHANNELS);
                let at = alloc(channels, &mut plan.arena_len);
                outs_of.insert((uid, o.name), (at, channels));
                (at, channels)
            })
            .collect();
        if n.manifest.type_name == "AudioOut" && plan.output.is_none() {
            plan.output = ins.first().cloned();
        }
        plan.stages.push(Stage { idx: n.idx, ins, params, outs });
    }
    (plan, faults)
}
