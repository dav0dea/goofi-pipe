//! Buffer — a rolling window along one axis.
//!
//! Keeps the most recent `size` entries along `axis` and emits them. **The rank never changes**:
//! a `[C, T]` frame buffered on the time axis stays `[C, size]`, and a scalar — which is `[1]`
//! here, since a 0-d array is promoted on construction — becomes `[size]`. A node adds or drops a
//! dimension only when its computation demands one, and rolling a window demands none.
//!
//! Each position along the OTHER axes gets its own window, so buffering `[C, T]` on time keeps C
//! independent channel histories rather than interleaving them into one stream.

use goofi_core::{resolve_axis, Axis, Data, SlotType};
use goofi_node::{
    default_factory, Inputs, Isolation, Node, NodeCtx, NodeManifest, NodeResult, OutputDecl,
    Outputs, ParamDecl, ParamSpec, Params, SlotDecl,
};

#[derive(Default)]
struct Buffer {
    /// One window per position along the axes OUTSIDE the buffered one — `windows[o]` holds that
    /// block's kept entries back to back, as raw f32-LE bytes, `inner * 4` per entry.
    windows: Vec<Vec<u8>>,
    /// The shape the windows were filled from, with the buffered axis set to 0. A frame that
    /// disagrees describes a different signal, so the windows are dropped rather than mixed.
    layout: Vec<usize>,
}

impl Node for Buffer {
    fn process(
        &mut self,
        inp: &Inputs<'_>,
        out: &mut Outputs<'_>,
        _c: &mut NodeCtx,
        p: &Params<'_>,
    ) -> NodeResult {
        let Some(d) = inp.get("data") else { return Ok(()) };
        let size = p.i64("buffer", "size").unwrap_or(1000).max(1) as usize;
        let a = d.assert_ndims().at_least(1)?;
        let axis = resolve_axis(p.i64("buffer", "axis").unwrap_or(-1), a.ndim())?;

        let shape = a.shape();
        // Bytes per entry along the buffered axis, and how many independent windows the axes
        // before it demand.
        let stride: usize = shape[axis + 1..].iter().product::<usize>() * 4;
        let outer: usize = shape[..axis].iter().product();
        let block = shape[axis] * stride;

        let mut layout = shape.to_vec();
        layout[axis] = 0;
        if layout != self.layout {
            self.windows = vec![Vec::new(); outer];
            self.layout = layout;
        }

        let cap = size * stride;
        let src = a.as_bytes();
        for (o, window) in self.windows.iter_mut().enumerate() {
            window.extend_from_slice(&src[o * block..(o + 1) * block]);
            if window.len() > cap {
                window.drain(..window.len() - cap);
            }
        }

        let kept = self.windows.first().map_or(0, |w| w.len() / stride);
        let mut buf = Vec::with_capacity(outer * kept * stride);
        for window in &self.windows {
            buf.extend_from_slice(window);
        }

        let mut shape_out = shape.to_vec();
        shape_out[axis] = kept;
        // Everything the frame carried stays, EXCEPT the buffered axis' own labels: they name
        // entries that have since rolled past. The other axes are untouched — a channel name is
        // still that channel's.
        let mut meta = d.meta().clone();
        if meta.channels().get(axis).is_some_and(|x| !x.is_empty()) {
            let axes = meta.channels().clone().with(axis, Axis::default());
            meta.set_channels(axes);
        }
        out.set("out", Data::array_f32(shape_out, buf, meta).map_err(|e| e.to_string())?);
        Ok(())
    }
}

static PARAMS: &[ParamDecl] = &[
    ParamDecl {
        group: "buffer",
        name: "size",
        spec: ParamSpec::Int { default: 1000, min: 1, max: 10_000_000 },
        expression: None,
        doc: Some("How many of the most recent entries to keep along the chosen axis."),
    },
    ParamDecl {
        group: "buffer",
        name: "axis",
        spec: ParamSpec::Int { default: -1, min: -8, max: 7 },
        expression: None,
        doc: Some(
            "Which axis to roll along, negative from the end. -1 is time, the default and the \
             one a signal is usually buffered over; -2 is channels.",
        ),
    },
];
static INPUTS: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: SlotType::Array,
    trigger_process: true,
    multi: false,
    required: true,
}];
static OUTPUTS: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: SlotType::Array,
}];

inventory::submit! {
    NodeManifest {
        type_name: "Buffer",
        category: "signal",
        doc: "Rolling window along one axis: keeps the most recent `size` entries, rank unchanged.",
        inputs: INPUTS,
        outputs: OUTPUTS,
        params: PARAMS,
        isolation: Isolation::InProcess,
        producer: false,
        factory: default_factory::<Buffer>,
    }
}
