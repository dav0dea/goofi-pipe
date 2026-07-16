//! Math — element-wise binary op between a float32 array and a scalar constant.
//! Length-preserving, so the full input meta (sfreq/channels/index) carries through.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy)]
enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Op {
    fn parse(s: &str) -> Op {
        match s {
            "subtract" => Op::Subtract,
            "multiply" => Op::Multiply,
            "divide" => Op::Divide,
            _ => Op::Add,
        }
    }
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Op::Add => a + b,
            Op::Subtract => a - b,
            Op::Multiply => a * b,
            Op::Divide => a / b,
        }
    }
}

struct Math {
    op: Op,
    value: f32,
}

impl Node for Math {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(store.as_bytes().len());
        for chunk in store.as_bytes().chunks_exact(4) {
            let a = f32::from_le_bytes(chunk.try_into().unwrap());
            buf.extend_from_slice(&self.op.apply(a, self.value).to_le_bytes());
        }
        // Length-preserving: the input meta stays valid (shape unchanged).
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        match (key.group.as_str(), key.name.as_str()) {
            ("math", "operation") => {
                if let Some(s) = v.as_str() {
                    self.op = Op::parse(s);
                }
            }
            ("math", "value") => {
                if let Some(x) = v.as_f64() {
                    self.value = x as f32;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert(
        "operation".to_string(),
        Param::Str {
            value: "add".to_string(),
            options: Some(vec![
                "add".into(),
                "subtract".into(),
                "multiply".into(),
                "divide".into(),
            ]),
            refresh: None,
        },
    );
    g.insert("value".to_string(), Param::float(0.0, -1.0e9, 1.0e9));
    let mut groups = ParamGroups::new();
    groups.insert("math".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let op = param(p, "math", "operation")
        .and_then(Param::as_str)
        .map(Op::parse)
        .unwrap_or(Op::Add);
    let value = param(p, "math", "value").and_then(Param::as_f64).unwrap_or(0.0) as f32;
    Box::new(Math { op, value })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Math",
        category: "array",
        doc: "Element-wise op (add/subtract/multiply/divide) with a scalar constant.",
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

    #[test]
    fn math_multiplies_elementwise() {
        let m = goofi_node::find("Math").unwrap();
        let params = (m.default_params)();
        let mut node = (m.make)(&params);
        node.on_param_changed(
            &ParamKey::new("math", "operation"),
            &Param::Str {
                value: "multiply".into(),
                options: None,
                refresh: None,
            },
        )
        .unwrap();
        node.on_param_changed(&ParamKey::new("math", "value"), &Param::float(3.0, -1e9, 1e9))
            .unwrap();

        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![3], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        if let Value::Array(s) = d.value() {
            let v: Vec<f32> = s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(v, vec![3.0, 6.0, 9.0]);
        } else {
            panic!("expected array");
        }
    }
}
