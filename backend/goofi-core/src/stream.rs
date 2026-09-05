//! `Stream` — the recent past of one node's input, so a stateful transform keeps no state of its
//! own and every chunking of one signal gives one answer.

/// One input's recent past, held time-major so a step is `stride` contiguous bytes.
#[derive(Default)]
pub struct Stream {
    stride: usize,
    data: Vec<u8>,
}

/// The dimensions outside `dim`, `dim` itself, and the dimensions inside it.
fn split(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    (
        shape[..dim].iter().product(),
        shape.get(dim).copied().unwrap_or(0),
        shape[dim + 1..].iter().product(),
    )
}

fn reorder(shape: &[usize], dim: usize, src: &[u8], to_time_major: bool) -> Vec<u8> {
    let (outer, steps, inner) = split(shape, dim);
    let ib = inner * 4;
    if outer <= 1 {
        return src.to_vec();
    }
    let mut out = vec![0u8; src.len()];
    for t in 0..steps {
        for o in 0..outer {
            let (a, b) = (((o * steps) + t) * ib, ((t * outer) + o) * ib);
            let (from, to) = if to_time_major { (a, b) } else { (b, a) };
            out[to..to + ib].copy_from_slice(&src[from..from + ib]);
        }
    }
    out
}

fn step(s: &[u8], i: usize, stride: usize) -> &[u8] {
    &s[i * stride..(i + 1) * stride]
}

/// The longest prefix of `frame` that is also a suffix of `past`, in steps.
fn overlap(past: &[u8], frame: &[u8], stride: usize) -> usize {
    let (p, f) = (past.len() / stride, frame.len() / stride);
    if p == 0 || f == 0 {
        return 0;
    }
    // A frame delivered twice matches whole, which the automaton below cannot report: its full
    // match is a proper prefix by construction.
    if f <= p && past[(p - f) * stride..] == *frame {
        return f;
    }
    let mut pi = vec![0usize; f];
    let mut k = 0;
    for i in 1..f {
        while k > 0 && step(frame, i, stride) != step(frame, k, stride) {
            k = pi[k - 1];
        }
        if step(frame, i, stride) == step(frame, k, stride) {
            k += 1;
        }
        pi[i] = k;
    }
    let mut k = 0;
    for i in 0..p {
        while k > 0 && step(past, i, stride) != step(frame, k, stride) {
            k = pi[k - 1];
        }
        if step(past, i, stride) == step(frame, k, stride) {
            k += 1;
        }
        if k == f {
            k = pi[k - 1];
        }
    }
    k
}

impl Stream {
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Forget the past — what a `reset` pulse clears.
    pub fn reset(&mut self) {
        self.data.clear();
        self.stride = 0;
    }

    /// The steps held right now.
    pub fn steps(&self) -> usize {
        self.data.len().checked_div(self.stride).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.steps() == 0
    }

    /// Fold one frame in along `dim`, over at least `history` steps of past. Answers the stitched
    /// frame in the caller's own layout, and the step the frame starts at along `dim`.
    pub fn push(
        &mut self,
        shape: &[usize],
        dim: usize,
        frame: &[u8],
        history: usize,
    ) -> (Vec<usize>, Vec<u8>, usize) {
        let (_, steps, _) = split(shape, dim);
        let stride = frame.len().checked_div(steps).unwrap_or(0);
        if stride == 0 {
            return (shape.to_vec(), frame.to_vec(), 0);
        }
        if stride != self.stride {
            self.data.clear();
            self.stride = stride;
        }
        // One whole frame of past is the floor: a rolling window overlaps its predecessor by all
        // but its newest steps, and a shorter past cannot express that match.
        let keep = history.max(steps).min(self.steps());
        let cut = self.data.len() - keep * stride;
        self.data.drain(..cut);

        let tm = reorder(shape, dim, frame, true);
        let k = overlap(&self.data, &tm, stride);
        let offset = self.steps() - k;
        self.data.extend_from_slice(&tm[k * stride..]);

        let mut out_shape = shape.to_vec();
        out_shape[dim] = self.steps();
        (out_shape.clone(), reorder(&out_shape, dim, &self.data, false), offset)
    }
}
