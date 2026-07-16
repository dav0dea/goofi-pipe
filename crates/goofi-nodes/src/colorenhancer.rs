//! ColorEnhancer — elementwise contrast/brightness/gamma correction plus an
//! optional RGB color boost for a float32 image normalized to `[0, 1]`.
//! Length-preserving, so the full input meta carries through unchanged. Ported
//! from `nodes/misc/colorenhancer.py`.

use goofi_core::SlotType;
use goofi_core::{Data, DType, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

struct ColorEnhancer {
    contrast: f32,
    brightness: f32,
    gamma: f32,
    color_boost: f32,
}

impl Node for ColorEnhancer {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("image") else {
            return Ok(());
        };
        let Value::Array(store) = d.value() else {
            return Ok(());
        };
        if store.dtype() != DType::F32 {
            return Ok(());
        }
        // `enhanced *= color_boost` in numpy scales the WHOLE array; it's gated on
        // the last axis being RGB (shape[-1] == 3), grayscale is left untouched.
        let boost = store.shape().last() == Some(&3);
        let mut buf = Vec::with_capacity(store.as_bytes().len());
        for chunk in store.as_bytes().chunks_exact(4) {
            let x = f32::from_le_bytes(chunk.try_into().unwrap());
            // Contrast pivots around the [0,1] midpoint, then brightness offsets.
            let mut e = (x - 0.5) * self.contrast + 0.5 + self.brightness;
            // np.power of a negative base with a non-integer gamma is NaN; f32::powf
            // matches np.power faithfully (both defer to C powf, incl. exact results
            // for integer exponents), and NaN flows through .clamp() unchanged just
            // as np.clip propagates it — so no special-casing.
            e = e.powf(self.gamma).clamp(0.0, 1.0);
            if boost {
                e = (e * self.color_boost).clamp(0.0, 1.0);
            }
            buf.extend_from_slice(&e.to_le_bytes());
        }
        // Length-preserving: the input shape/meta stay valid.
        let data = Data::from_array_bytes(DType::F32, store.shape().to_vec(), buf, d.meta().clone())
            .map_err(|e| e.to_string())?;
        out.set("enhanced_image", data);
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        let Some(x) = v.as_f64().map(|x| x as f32) else {
            return Ok(());
        };
        match (key.group.as_str(), key.name.as_str()) {
            ("enhancement", "contrast") => self.contrast = x,
            ("enhancement", "brightness") => self.brightness = x,
            ("enhancement", "gamma") => self.gamma = x,
            ("enhancement", "color_boost") => self.color_boost = x,
            _ => {}
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("contrast".to_string(), Param::float(1.0, 0.1, 10.0));
    g.insert("brightness".to_string(), Param::float(0.0, -0.9, 0.9));
    g.insert("gamma".to_string(), Param::float(1.0, 0.1, 5.0));
    g.insert("color_boost".to_string(), Param::float(1.0, 0.5, 3.0));
    let mut groups = ParamGroups::new();
    groups.insert("enhancement".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let get = |name: &str, dflt: f32| {
        param(p, "enhancement", name)
            .and_then(Param::as_f64)
            .unwrap_or(dflt as f64) as f32
    };
    Box::new(ColorEnhancer {
        contrast: get("contrast", 1.0),
        brightness: get("brightness", 0.0),
        gamma: get("gamma", 1.0),
        color_boost: get("color_boost", 1.0),
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "image",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "enhanced_image",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "ColorEnhancer",
        category: "misc",
        doc: "Enhance a [0,1] image with contrast/brightness/gamma (+ RGB color boost).",
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

    fn enhance(
        contrast: f32,
        brightness: f32,
        gamma: f32,
        color_boost: f32,
        shape: Vec<usize>,
        input: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        let m = goofi_node::find("ColorEnhancer").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        for (name, val) in [
            ("contrast", contrast),
            ("brightness", brightness),
            ("gamma", gamma),
            ("color_boost", color_boost),
        ] {
            node.on_param_changed(&ParamKey::new("enhancement", name), &Param::float(val as f64, -1e9, 1e9))
                .unwrap();
        }
        let buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, shape, buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("image", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("enhanced_image").unwrap().as_ref().unwrap().clone();
        match d.value() {
            Value::Array(s) => (
                s.shape().to_vec(),
                s.as_bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            ),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn color_image_applies_contrast_brightness_and_boost() {
        // shape [1,1,3] -> last axis 3 -> color boost applies.
        // contrast=2, brightness=0.1, gamma=1, boost=1.5:
        //   0.5 -> (0.0)*2+0.6 = 0.6 -> *1.5 = 0.9
        //   0.7 -> (0.2)*2+0.6 = 1.0 (clip) -> *1.5 = 1.0 (clip)
        //   0.3 -> (-0.2)*2+0.6 = 0.2 -> *1.5 = 0.3
        let (shape, out) = enhance(2.0, 0.1, 1.0, 1.5, vec![1, 1, 3], &[0.5, 0.7, 0.3]);
        assert_eq!(shape, vec![1, 1, 3]);
        assert!((out[0] - 0.9).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "got {}", out[1]);
        assert!((out[2] - 0.3).abs() < 1e-5, "got {}", out[2]);
    }

    #[test]
    fn grayscale_skips_color_boost() {
        // shape [2,2] -> last axis 2 != 3 -> boost ignored (else 0.25*3 would clip).
        // Identity contrast/brightness/gamma leaves in-range values unchanged.
        let (shape, out) = enhance(1.0, 0.0, 1.0, 3.0, vec![2, 2], &[0.25, 0.5, 0.75, 0.5]);
        assert_eq!(shape, vec![2, 2]);
        for (a, b) in out.iter().zip([0.25, 0.5, 0.75, 0.5]) {
            assert!((a - b).abs() < 1e-6, "got {a} expected {b}");
        }
    }

    #[test]
    fn negative_base_noninteger_gamma_is_nan_like_numpy() {
        // brightness drives e negative; a fractional gamma on a negative base is NaN
        // (np.power), and np.clip(nan) stays nan. shape [1] -> last axis 1, no boost.
        //   0.0 -> (0-0.5)+0.5-0.9 = -0.9; (-0.9)^0.5 = NaN
        let (_, out) = enhance(1.0, -0.9, 0.5, 1.0, vec![1], &[0.0]);
        assert!(out[0].is_nan(), "expected NaN, got {}", out[0]);
    }

    #[test]
    fn clips_result_into_unit_range() {
        // shape [2] (no boost). contrast=2, brightness=-0.5, gamma=1:
        //   0.0 -> (-0.5)*2+0.5-0.5 = -1.0 -> clamp 0.0
        //   1.0 -> ( 0.5)*2+0.5-0.5 =  1.0 -> clamp 1.0
        let (_, out) = enhance(2.0, -0.5, 1.0, 1.0, vec![2], &[0.0, 1.0]);
        assert!((out[0] - 0.0).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-6, "got {}", out[1]);
    }

    #[test]
    fn absent_input_does_not_emit() {
        let m = goofi_node::find("ColorEnhancer").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("image", None);
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("enhanced_image").unwrap().is_none(), "no input -> no emit");
    }

    #[test]
    fn non_f32_dtype_does_not_emit() {
        // ARRAY nodes are f32-only; a u8 image must be ignored, not misread.
        let m = goofi_node::find("ColorEnhancer").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let frame = Data::from_array_bytes(DType::U8, vec![3], vec![1u8, 2, 3], Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("image", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        assert!(outbuf.get("enhanced_image").unwrap().is_none(), "non-f32 -> no emit");
    }

    #[test]
    fn carries_input_meta_unchanged() {
        let m = goofi_node::find("ColorEnhancer").unwrap();
        let mut node = (m.make)(&(m.default_params)());
        let mut meta = Meta::empty();
        meta.sfreq = Some(30.0);
        meta.index = Some(7);
        let buf: Vec<u8> = [0.5f32, 0.5, 0.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Data::from_array_bytes(DType::F32, vec![1, 1, 3], buf, meta).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("image", Some(frame));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx::new()).unwrap();
        let d = outbuf.get("enhanced_image").unwrap().as_ref().unwrap();
        assert_eq!(d.meta().sfreq, Some(30.0));
        assert_eq!(d.meta().index, Some(7));
    }
}
