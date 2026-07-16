//! Delay — forward each frame after a fixed wall-clock delay. The Python
//! `signal/delay.py` sleeps in `process()`, which is impossible in the single-
//! process engine (it would stall the shared tick loop for every node). This is
//! the architecturally-correct redesign: a NON-blocking queue keyed on `ctx.now`
//! — each fresh frame is buffered with a release time, and the oldest due frame is
//! emitted per tick. Steady streams get a clean `delay`-second time shift.

use std::collections::VecDeque;

use goofi_core::SlotType;
use goofi_core::{Data, Param, Value};
use goofi_node::{
    param, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl, Outputs,
    ParamGroups, ParamKey, SlotDecl,
};
use indexmap::IndexMap;

/// Sub-nanosecond slack so f64 wall-clock accumulation noise (e.g. 0.05 + 0.1 !=
/// 0.15 exactly) can't hold a frame an extra tick at an exact release boundary.
const DUE_EPS: f64 = 1e-9;

struct Delay {
    delay: f64, // seconds
    queue: VecDeque<(f64, Data)>, // (release_time, frame), oldest at the front
}

impl Node for Delay {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, c: &mut NodeCtx) -> NodeResult {
        // "data" is a triggering input, so process runs once per fresh frame:
        // buffer it with the instant it becomes due.
        if let Some(d) = inp.get("data") {
            if matches!(d.value(), Value::Array(_) | Value::Str(_) | Value::Table(_)) {
                self.queue.push_back((c.now + self.delay.max(0.0), d.clone()));
            }
        }
        // Emit the oldest frame whose release time has arrived (one per tick — a
        // steady one-in/one-out stream stays one-in/one-out, just time-shifted).
        if let Some(&(release, _)) = self.queue.front() {
            if release <= c.now + DUE_EPS {
                let (_, frame) = self.queue.pop_front().unwrap();
                out.set("output", frame);
            }
        }
        Ok(())
    }

    fn on_param_changed(&mut self, key: &ParamKey, v: &Param) -> NodeResult {
        if key.group == "delay" && key.name == "time" {
            if let Some(x) = v.as_f64() {
                self.delay = x;
            }
        }
        Ok(())
    }
}

fn default_params() -> ParamGroups {
    let mut g = IndexMap::new();
    g.insert("time".to_string(), Param::float(0.1, 0.0, 100.0));
    let mut groups = ParamGroups::new();
    groups.insert("delay".to_string(), g);
    groups
}

fn make(p: &ParamGroups) -> Box<dyn Node> {
    let delay = param(p, "delay", "time").and_then(Param::as_f64).unwrap_or(0.1);
    Box::new(Delay {
        delay,
        queue: VecDeque::new(),
    })
}

static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "output",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Delay",
        category: "signal",
        doc: "Forward each frame after a fixed wall-clock delay (non-blocking).",
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

    fn frame(id: f32) -> Data {
        Data::from_array_bytes(DType::F32, vec![1], id.to_le_bytes().to_vec(), Meta::empty()).unwrap()
    }
    fn val(d: &Data) -> f32 {
        match d.value() {
            Value::Array(s) => f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()),
            _ => panic!(),
        }
    }

    // Feed `frame` at wall-clock `now`; return the emitted frame's id, if any.
    fn tick(node: &mut Box<dyn goofi_node::Node>, m: &goofi_node::NodeManifest, id: f32, now: f64) -> Option<f32> {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(frame(id)));
        let inp = Inputs::new(&inmap);
        let mut outbuf = m.output_buffer();
        node.process(&inp, &mut Outputs::new(&mut outbuf), &mut NodeCtx { tick: 0, now }).unwrap();
        outbuf.get("output").unwrap().as_ref().map(val)
    }

    #[test]
    fn frames_emerge_after_the_delay() {
        // 0.1 s delay. Feed A,B,C,D at 0, 0.05, 0.1, 0.15 s.
        let m = goofi_node::find("Delay").unwrap();
        let mut node = (m.make)(&(m.default_params)()); // default 0.1 s
        assert_eq!(tick(&mut node, m, 1.0, 0.0), None, "A held (0 < 0.1)");
        assert_eq!(tick(&mut node, m, 2.0, 0.05), None, "B held");
        assert_eq!(tick(&mut node, m, 3.0, 0.10), Some(1.0), "A due at 0.10 -> emits A");
        assert_eq!(tick(&mut node, m, 4.0, 0.15), Some(2.0), "B due at 0.15 -> emits B");
    }

    #[test]
    fn zero_delay_is_passthrough() {
        let m = goofi_node::find("Delay").unwrap();
        let mut params = (m.default_params)();
        params["delay"].insert("time".into(), goofi_core::Param::float(0.0, 0.0, 100.0));
        let mut node = (m.make)(&params);
        assert_eq!(tick(&mut node, m, 7.0, 0.0), Some(7.0), "delay 0 forwards immediately");
        assert_eq!(tick(&mut node, m, 8.0, 0.01), Some(8.0));
    }
}
