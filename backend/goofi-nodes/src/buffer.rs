//! Buffer — a rolling window over a float32 stream. Appends each incoming frame's
//! samples and emits the most recent `size` of them. The reference stress patch
//! uses eight of these. Length-changing, so only `sfreq` is carried through.

use goofi_core::SlotType;
use goofi_core::{Data, Meta, Value};
use goofi_node::{
    default_factory, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl,
    Outputs, ParamDecl, ParamSpec, Params, SlotDecl,
};

/// `size` is a cold param — read live from `p` each tick, so the node holds only the
/// rolling window as state (no `size` field, no `on_param_changed`).
#[derive(Default)]
struct Buffer {
    ring: Vec<f32>,
}

impl Node for Buffer {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx, p: &Params<'_>) -> NodeResult {
        let size = p.i64("buffer", "size").unwrap_or(1000).max(1) as usize;
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        for chunk in store.as_bytes().chunks_exact(4) {
            self.ring.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        if self.ring.len() > size {
            let excess = self.ring.len() - size;
            self.ring.drain(0..excess);
        }

        let mut buf = Vec::with_capacity(self.ring.len() * 4);
        for v in &self.ring {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let meta = Meta::new().with_sfreq(d.meta().sfreq());
        let data = Data::array_f32(vec![self.ring.len()], buf, meta)
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[ParamDecl {
    group: "buffer",
    name: "size",
    spec: ParamSpec::Int { default: 1000, min: 1, max: 10_000_000 },
    expression: None,
    doc: Some("How many of the most recent samples to keep along the time axis."),
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Buffer",
        category: "signal",
        doc: "Rolling window over a float32 stream; emits the most recent `size` samples.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        params: PARAMS,
        isolation: Isolation::InProcess,
        producer: false,
        factory: default_factory::<Buffer>,
    }
}
