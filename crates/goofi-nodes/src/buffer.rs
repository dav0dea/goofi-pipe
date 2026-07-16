//! Buffer — a rolling window over a float32 stream. Appends each incoming frame's
//! samples and emits the most recent `size` of them. The reference stress patch
//! uses eight of these. Length-changing, so only `sfreq` is carried through.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Meta, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct Buffer {
    size: usize,
    ring: Vec<f32>,
}

impl Node for Buffer {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(()); // native Buffer handles float32 for now
        }
        for chunk in store.as_bytes().chunks_exact(4) {
            self.ring.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        if self.ring.len() > self.size {
            let excess = self.ring.len() - self.size;
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
        let data = Data::from_array_bytes(DType::F32, vec![self.ring.len()], buf, meta)
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "buffer" && key.name == "size" {
            if let Some(x) = v.as_i64() {
                self.size = x.max(1) as usize;
                if self.ring.len() > self.size {
                    let excess = self.ring.len() - self.size;
                    self.ring.drain(0..excess);
                }
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("size".to_string(), Param::int(1000, 1, 10_000_000));
    let mut groups = ParamGroups::new();
    groups.insert("buffer".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let size = param(p, "buffer", "size")
        .and_then(Param::as_i64)
        .unwrap_or(1000)
        .max(1) as usize;
    Box::new(Buffer {
        size,
        ring: Vec::new(),
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
    length_preserving: false,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Buffer",
        category: "signal",
        doc: "Rolling window over a float32 stream; emits the most recent `size` samples.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, DType, Meta, Param, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;

    fn f32_frame(vals: &[f32]) -> Data {
        let buf: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        Data::from_array_bytes(DType::F32, vec![vals.len()], buf, Meta::empty()).unwrap()
    }

    fn run(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, frame: Data) -> Data {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
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
        let params = (m.default_params)();
        let mut node = (m.make)(&params);
        node.on_param_changed(&ParamKey::new("buffer", "size"), &Param::int(3, 1, 100))
            .unwrap();

        let o1 = run(&mut node, m, f32_frame(&[1.0, 2.0]));
        assert_eq!(to_vec(&o1), vec![1.0, 2.0]);

        let o2 = run(&mut node, m, f32_frame(&[3.0, 4.0]));
        // last 3 of [1,2,3,4]
        assert_eq!(to_vec(&o2), vec![2.0, 3.0, 4.0]);
    }
}
