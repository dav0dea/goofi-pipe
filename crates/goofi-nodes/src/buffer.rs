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
        let meta = Meta {
            sfreq: d.meta().sfreq,
            ..Default::default()
        };
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
    default_expr: None,
}];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
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
        factory: default_factory::<Buffer>,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamGroups, Params};
    use indexmap::IndexMap;

    fn f32_frame(vals: &[f32]) -> Data {
        let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::array_f32(vec![vals.len()], buf, Meta::empty()).unwrap()
    }

    /// Params with `buffer.size` = `size` (cold-read live by `process`).
    fn params_with_size(size: i64) -> ParamGroups {
        let mut g = IndexMap::new();
        g.insert("size".to_string(), Param::int(size, 1, 100));
        let mut groups = ParamGroups::new();
        groups.insert("buffer".to_string(), g);
        groups
    }

    fn run(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, params: &ParamGroups, frame: Data) -> Data {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx, &Params::new(params)).unwrap();
        }
        outbuf.get("out").unwrap().as_ref().unwrap().clone()
    }

    fn to_vec(d: &Data) -> Vec<f32> {
        if let Value::Array(s) = d.value() {
            s.as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            panic!("not array")
        }
    }

    #[test]
    fn buffer_rolls_to_window_size() {
        let m = goofi_node::find("Buffer").unwrap();
        let mut node = (m.factory)();
        // `size` is read live from params — a size of 3, no on_param_changed needed.
        let params = params_with_size(3);

        let o1 = run(&mut node, m, &params, f32_frame(&[1.0, 2.0]));
        assert_eq!(to_vec(&o1), vec![1.0, 2.0]);

        let o2 = run(&mut node, m, &params, f32_frame(&[3.0, 4.0]));
        // last 3 of [1,2,3,4]
        assert_eq!(to_vec(&o2), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn default_params_declare_size() {
        let m = goofi_node::find("Buffer").unwrap();
        let p = m.default_params();
        assert_eq!(p["buffer"]["size"].as_i64(), Some(1000));
    }
}
