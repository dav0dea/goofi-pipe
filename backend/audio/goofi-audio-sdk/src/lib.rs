//! The audio node's author contract: the `AudioNode` trait, the block it is handed, and the
//! conventions every signal in the engine follows — shared by the engine that runs a node and
//! the file that authors one, so the two halves cannot drift.

pub use goofi_core;
pub use goofi_node::{ExprDecl, ExprMode, OutputDecl, ParamDecl, ParamKey, ParamSpec, SlotDecl};

/// Every block is exactly this many frames; the engine carries any surplus to the next callback.
pub const BLOCK: usize = 64;
/// What a node is prepared for; a port never carries more channels than this.
pub const MAX_CHANNELS: u16 = 16;

/// What a node file declares: a `NodeManifest` less the type name, which is the FILE's. The
/// signal-only slot flags are ignored; `multi: true` on an input sums its wires at the jack.
pub struct Manifest {
    pub category: &'static str,
    pub doc: &'static str,
    pub inputs: &'static [SlotDecl],
    pub outputs: &'static [OutputDecl],
    pub params: &'static [ParamDecl],
}

/// The params a node declares, as ONE list that is both the manifest's slice and the indices a
/// node reads them by: `params! { CUTOFF = ParamDecl { … }, GAIN = ParamDecl { … } }` yields
/// `PARAMS` and `P::CUTOFF == 0`, `P::GAIN == 1`.
#[macro_export]
macro_rules! params {
    ($($name:ident = $decl:expr),* $(,)?) => {
        pub static PARAMS: &[$crate::ParamDecl] = &[$($decl),*];
        pub mod P {
            $crate::params!(@index 0usize; $($name)*);
        }
    };
    (@index $i:expr; $name:ident $($rest:ident)*) => {
        pub const $name: usize = $i;
        $crate::params!(@index $i + 1usize; $($rest)*);
    };
    (@index $i:expr;) => {};
}

/// One input or param for one block: planar channels of `BLOCK` frames each. A param's region
/// holds its one source, settled for this block.
pub struct Port<'a> {
    channels: u16,
    wired: bool,
    data: &'a [f32],
}

impl<'a> Port<'a> {
    /// `data` holds `channels` planar blocks; `wired` is false for the shared silence an unwired
    /// input reads.
    pub fn new(data: &'a [f32], channels: u16, wired: bool) -> Port<'a> {
        debug_assert_eq!(data.len(), channels as usize * BLOCK);
        Port { channels, wired, data }
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// `false` when nothing is wired: the data is silence and `channels` is 1.
    pub fn wired(&self) -> bool {
        self.wired
    }

    /// Channel `c` of a port that may be narrower than the block it feeds: a one-channel port is
    /// on every channel, a channel past a wider port's count is silence.
    pub fn chan(&self, c: usize) -> &[f32; BLOCK] {
        let c = if self.channels == 1 { 0 } else { c };
        if c >= self.channels as usize {
            return &SILENT;
        }
        self.data[c * BLOCK..(c + 1) * BLOCK].try_into().expect("a channel is BLOCK frames")
    }
}

static SILENT: [f32; BLOCK] = [0.0; BLOCK];

/// One output for one block, with the channel count `channels()` answered for this plan.
pub struct PortMut<'a> {
    channels: u16,
    data: &'a mut [f32],
}

impl<'a> PortMut<'a> {
    pub fn new(data: &'a mut [f32], channels: u16) -> PortMut<'a> {
        debug_assert_eq!(data.len(), channels as usize * BLOCK);
        PortMut { channels, data }
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn chan_mut(&mut self, c: usize) -> &mut [f32; BLOCK] {
        (&mut self.data[c * BLOCK..(c + 1) * BLOCK]).try_into().expect("a channel is BLOCK frames")
    }
}

/// One block: every declared input, output and param, in declaration order — wires already
/// summed and coerced, a param's region holding its one source.
pub struct Block<'a> {
    pub ins: &'a [Port<'a>],
    pub outs: &'a mut [PortMut<'a>],
    pub params: &'a [Port<'a>],
}

/// The DSP half of an audio node. It moves to the audio thread inside the plan and owns
/// arithmetic only: no allocation, no lock, no syscall in `process`.
pub trait AudioNode: Send {
    /// Per-output channel counts for these per-input counts — ports first, then referenced
    /// params, in declaration order — and the settled scalar params. Pure; evaluated at plan
    /// compile, on the control thread. The default is `max(ins).max(1)` for each of the `outs`.
    fn channels(&self, ins: &[u16], _params: &[f64], outs: usize) -> Vec<u16> {
        vec![ins.iter().copied().max().unwrap_or(1).max(1); outs]
    }
    /// Once on the control thread before the first block, again only when the rate changes.
    /// Allocate here, for `MAX_CHANNELS` and `BLOCK` frames.
    fn prepare(&mut self, rate: f64);
    /// One block, on the audio thread.
    fn process(&mut self, b: &mut Block<'_>);
    /// `true` for a type whose outputs come from the PREVIOUS block's inputs — the one way a
    /// loop closes. The sort ignores its in-edges; it runs first each block on last block's
    /// regions.
    fn feedback(&self) -> bool {
        false
    }
    /// State beyond the params, as opaque bytes. A param value is never in it, and a node that
    /// returns nothing leaves nothing behind.
    fn save(&self) -> Vec<u8> {
        Vec::new()
    }
    fn load(&mut self, _bytes: &[u8]) {}
}

/// The conventions, stated once: a bipolar signal lives in `[-1, 1]` and `1` is full scale; a
/// unipolar one in `[0, 1]`; a gate is HIGH at `>= GATE_HIGH`; pitch is volts per octave, zero
/// at C4, so transposition is an addition.
pub const GATE_HIGH: f32 = 0.5;
pub const C4_HZ: f32 = 261.63;

pub fn hz_of(pitch: f32) -> f32 {
    C4_HZ * 2f32.powf(pitch)
}

pub fn pitch_of(hz: f32) -> f32 {
    (hz / C4_HZ).log2()
}

/// A rising-edge detector over a gate: `true` on the sample the gate goes HIGH.
#[derive(Default)]
pub struct Edge {
    high: bool,
}

impl Edge {
    pub fn rising(&mut self, v: f32) -> bool {
        let high = v >= GATE_HIGH;
        let rose = high && !self.high;
        self.high = high;
        rose
    }
}
