//! Function — apply an element-wise math function (abs/sqrt/log/exp/trig) to a
//! float32 array. Length-preserving, so the full input meta (sfreq/channels/index)
//! carries through unchanged. Ported from the Python `nodes/array/function.py`
//! (`log` is the natural logarithm, matching numpy).

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy)]
enum Func {
    Abs,
    Sqrt,
    Log,
    Exp,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
}

impl Func {
    fn parse(s: &str) -> Func {
        match s {
            "sqrt" => Func::Sqrt,
            "log" => Func::Log,
            "exp" => Func::Exp,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "arcsin" => Func::Arcsin,
            "arccos" => Func::Arccos,
            "arctan" => Func::Arctan,
            _ => Func::Abs,
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Func::Abs => x.abs(),
            Func::Sqrt => x.sqrt(),
            Func::Log => x.ln(), // numpy `log` is the natural log
            Func::Exp => x.exp(),
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Arcsin => x.asin(),
            Func::Arccos => x.acos(),
            Func::Arctan => x.atan(),
        }
    }
}

struct Function {
    func: Func,
}

impl Node for Function {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("array") else {
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
            let x = f32::from_le_bytes(chunk.try_into().unwrap());
            buf.extend_from_slice(&self.func.apply(x).to_le_bytes());
        }
        // Length-preserving: shape and the input meta stay valid.
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("out", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "function" && key.name == "function" {
            if let Some(s) = v.as_str() {
                self.func = Func::parse(s);
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert(
        "function".to_string(),
        Param::Str {
            value: "abs".to_string(),
            options: Some(
                ["abs", "sqrt", "log", "exp", "sin", "cos", "tan", "arcsin", "arccos", "arctan"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("function".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let func = param(p, "function", "function").and_then(Param::as_str).map(Func::parse).unwrap_or(Func::Abs);
    Box::new(Function { func })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "array",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Function",
        category: "array",
        doc: "Apply an element-wise function (abs/sqrt/log/exp/trig) to the array.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        default_params,
        isolation: Isolation::InProcess,
        make,
    }
}

#[cfg(test)]
mod tests {
    use goofi_core::{Data, DType, Meta, Value};
    use goofi_node::{Inputs, NodeCtx, Outputs, ParamKey};
    use indexmap::IndexMap;
    use std::f32::consts::PI;

    fn apply(func: &str, input: &[f32]) -> Vec<f32> {
        let m = goofi_node::find("Function").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        node.on_param_changed(&ParamKey::new("function", "function"), &goofi_core::Param::str_free(func)).unwrap();
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![input.len()], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        match outbuf.get("out").unwrap().as_ref().unwrap().value() {
            Value::Array(s) => s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn abs_and_sqrt() {
        assert_eq!(apply("abs", &[-2.0, -0.5, 3.0]), vec![2.0, 0.5, 3.0]);
        assert_eq!(apply("sqrt", &[4.0, 9.0, 0.0]), vec![2.0, 3.0, 0.0]);
    }

    #[test]
    fn log_exp_are_inverse() {
        let e = apply("exp", &[0.0, 1.0]);
        assert!((e[0] - 1.0).abs() < 1e-6 && (e[1] - std::f32::consts::E).abs() < 1e-5);
        let l = apply("log", &[1.0, std::f32::consts::E]);
        assert!(l[0].abs() < 1e-6 && (l[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trig_values() {
        let s = apply("sin", &[0.0, PI / 2.0]);
        assert!(s[0].abs() < 1e-6 && (s[1] - 1.0).abs() < 1e-6);
        // arctan(1) = pi/4.
        assert!((apply("arctan", &[1.0])[0] - PI / 4.0).abs() < 1e-6);
    }

    #[test]
    fn length_preserving_shape_and_meta() {
        let m = goofi_node::find("Function").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut meta = Meta::empty();
        meta.sfreq = Some(250.0);
        let buf: Vec<u8> = [-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![2, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("array", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("out").unwrap().as_ref().unwrap();
        match d.value() {
            Value::Array(s) => assert_eq!(s.shape(), &[2, 3], "shape preserved"),
            _ => panic!("expected array"),
        }
        assert_eq!(d.meta().sfreq, Some(250.0), "meta carried through");
    }
}
