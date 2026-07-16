//! Normalization — z-score or min-max normalize a float32 array. Length-preserving,
//! so the full input meta carries through.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

#[derive(Clone, Copy)]
enum Mode {
    ZScore,
    MinMax,
}

impl Mode {
    fn parse(s: &str) -> Mode {
        match s {
            "minmax" => Mode::MinMax,
            _ => Mode::ZScore,
        }
    }
    fn apply(self, x: &[f32]) -> Vec<f32> {
        const EPS: f32 = 1e-12;
        match self {
            Mode::ZScore => {
                let n = x.len() as f32;
                let mean = x.iter().sum::<f32>() / n;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                let std = var.sqrt() + EPS;
                x.iter().map(|v| (v - mean) / std).collect()
            }
            Mode::MinMax => {
                let mn = x.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = (mx - mn) + EPS;
                x.iter().map(|v| (v - mn) / range).collect()
            }
        }
    }
}

struct Normalization {
    mode: Mode,
}

impl Node for Normalization {
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
        let x: Vec<f32> = store
            .as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if x.is_empty() {
            return Ok(());
        }
        let y = self.mode.apply(&x);
        let buf: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("normalized", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "normalization" && key.name == "mode" {
            if let Some(s) = v.as_str() {
                self.mode = Mode::parse(s);
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert(
        "mode".to_string(),
        Param::Str {
            value: "zscore".to_string(),
            options: Some(vec!["zscore".into(), "minmax".into()]),
            refresh: None,
        },
    );
    let mut groups = ParamGroups::new();
    groups.insert("normalization".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let mode = param(p, "normalization", "mode")
        .and_then(Param::as_str)
        .map(Mode::parse)
        .unwrap_or(Mode::ZScore);
    Box::new(Normalization { mode })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "normalized",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Normalization",
        category: "signal",
        doc: "Z-score or min-max normalize a float32 array.",
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
    use goofi_node::{Inputs, NodeCtx, Outputs};
    use indexmap::IndexMap;

    fn run(type_params: &str, input: &[f32]) -> Vec<f32> {
        let m = goofi_node::find("Normalization").unwrap();
        let mut params = (m.default_params)();
        params["normalization"].insert(
            "mode".into(),
            goofi_core::Param::Str {
                value: type_params.into(),
                options: None,
                refresh: None,
            },
        );
        let mut node = (m.make)(&params);
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![input.len()], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outbuf);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let d = outbuf.get("normalized").unwrap().as_ref().unwrap();
        if let Value::Array(s) = d.value() {
            s.as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            panic!("expected array")
        }
    }

    #[test]
    fn zscore_has_zero_mean_unit_std() {
        let y = run("zscore", &[1.0, 2.0, 3.0]);
        let mean = y.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5, "mean ~0, got {mean}");
        assert!((y[0] + 1.2247).abs() < 1e-3 && (y[2] - 1.2247).abs() < 1e-3);
    }

    #[test]
    fn minmax_maps_to_unit_interval() {
        let y = run("minmax", &[2.0, 4.0, 6.0]);
        assert!((y[0]).abs() < 1e-5 && (y[2] - 1.0).abs() < 1e-5 && (y[1] - 0.5).abs() < 1e-5);
    }
}
