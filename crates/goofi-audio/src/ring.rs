//! `AudioRing` — a bounded lock-free SPSC float32 ring buffer over interleaved
//! `[frames, channels]` blocks. Ported from Python `audio/ring.py`.
//!
//! Single producer (`write`, the generator thread) / single consumer
//! (`read_into`, the audio callback). Overflow is reject-newest: a `write` accepts
//! only what currently fits and drops the remainder (bumping `overruns` once). A
//! short `read_into` copies what's available and leaves the rest of `out`
//! untouched (bumping `underruns` once); the caller zero-fills. No per-call
//! allocation.
//!
//! Lock-free: the producer owns `w` (total frames written), the consumer owns `r`
//! (total frames read); both increase monotonically and occupancy is DERIVED as
//! `w - r`. No field is written by both threads, so there is no shared
//! read-modify-write. The buffer is a slice of per-element `UnsafeCell<f32>`;
//! producer and consumer only ever touch DISJOINT frame ranges (bounded by
//! occupancy) via raw pointers — never overlapping references — and the
//! Release/Acquire pairing on `w`/`r` publishes each payload before its index is
//! observable. Correct even without a GIL. Counters never wrap (u64).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AudioRing {
    buf: Box<[UnsafeCell<f32>]>,
    cap: usize,      // frames
    channels: usize, // interleaved width
    w: AtomicU64,    // producer-owned, monotonic frame count
    r: AtomicU64,    // consumer-owned, monotonic frame count
    overruns: AtomicU64,
    underruns: AtomicU64,
}

// SAFETY: single-producer / single-consumer. The producer writes only frames in
// `[w, w+take)` and publishes `w` after; the consumer reads only frames in
// `[r, r+take)` and publishes `r` after. Occupancy (`w - r <= cap`) guarantees
// those ranges are disjoint, so the two threads never access the same cell, and
// the Release store / Acquire load ordering makes each payload visible before its
// index. No overlapping references to `buf` are ever formed (raw-pointer copies).
unsafe impl Send for AudioRing {}
unsafe impl Sync for AudioRing {}

impl AudioRing {
    pub fn new(capacity_frames: usize, channels: usize) -> AudioRing {
        assert!(capacity_frames > 0 && channels > 0, "ring needs frames and channels");
        let buf = (0..capacity_frames * channels)
            .map(|_| UnsafeCell::new(0.0f32))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        AudioRing {
            buf,
            cap: capacity_frames,
            channels,
            w: AtomicU64::new(0),
            r: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
        }
    }

    fn base(&self) -> *mut f32 {
        // UnsafeCell<f32> is `repr(transparent)` over f32, so the slice's pointer
        // aliases a contiguous `[f32]` of the same length.
        self.buf.as_ptr() as *mut f32
    }

    /// PRODUCER ONLY. Write as many whole frames of `block` (interleaved,
    /// `channels` wide) as currently fit; drop the rest. Returns frames written.
    pub fn write(&self, block: &[f32]) -> usize {
        let n = block.len() / self.channels;
        let w = self.w.load(Ordering::Relaxed);
        let r = self.r.load(Ordering::Acquire);
        let free = self.cap - (w - r) as usize;
        let take = n.min(free);
        if take < n {
            self.overruns.fetch_add(1, Ordering::Relaxed);
        }
        if take > 0 {
            let start = (w as usize % self.cap) * self.channels;
            let ncols = take * self.channels;
            let cap_samples = self.cap * self.channels;
            let base = self.base();
            let src = block.as_ptr();
            // SAFETY: the target cells are free (not read by the consumer, bounded
            // by occupancy); ranges stay in `[0, cap_samples)`; single producer.
            unsafe {
                if start + ncols <= cap_samples {
                    std::ptr::copy_nonoverlapping(src, base.add(start), ncols);
                } else {
                    let first = cap_samples - start;
                    std::ptr::copy_nonoverlapping(src, base.add(start), first);
                    std::ptr::copy_nonoverlapping(src.add(first), base, ncols - first);
                }
            }
            self.w.store(w + take as u64, Ordering::Release);
        }
        take
    }

    /// CONSUMER ONLY. Fill up to `out.len()/channels` frames of `out` from the
    /// ring; a short read leaves the tail of `out` untouched. Returns frames read.
    pub fn read_into(&self, out: &mut [f32]) -> usize {
        let n = out.len() / self.channels;
        let r = self.r.load(Ordering::Relaxed);
        let w = self.w.load(Ordering::Acquire);
        let take = ((w - r) as usize).min(n);
        if take > 0 {
            let start = (r as usize % self.cap) * self.channels;
            let ncols = take * self.channels;
            let cap_samples = self.cap * self.channels;
            let base: *const f32 = self.base();
            let dst = out.as_mut_ptr();
            // SAFETY: the source cells are filled (not written by the producer,
            // bounded by occupancy); ranges stay in `[0, cap_samples)`; single
            // consumer; `out` is exclusively borrowed.
            unsafe {
                if start + ncols <= cap_samples {
                    std::ptr::copy_nonoverlapping(base.add(start), dst, ncols);
                } else {
                    let first = cap_samples - start;
                    std::ptr::copy_nonoverlapping(base.add(start), dst, first);
                    std::ptr::copy_nonoverlapping(base, dst.add(first), ncols - first);
                }
            }
            self.r.store(r + take as u64, Ordering::Release);
        }
        if take < n {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
        take
    }

    pub fn fill(&self) -> usize {
        (self.w.load(Ordering::Acquire) - self.r.load(Ordering::Acquire)) as usize
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn overruns(&self) -> u64 {
        self.overruns.load(Ordering::Relaxed)
    }
    pub fn underruns(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn write_read_roundtrip_and_fill() {
        let ring = AudioRing::new(4, 2); // 4 frames, 2 channels
        assert_eq!(ring.write(&[1.0, 1.5, 2.0, 2.5]), 2); // 2 frames
        assert_eq!(ring.fill(), 2);
        let mut out = [0.0f32; 4];
        assert_eq!(ring.read_into(&mut out), 2);
        assert_eq!(out, [1.0, 1.5, 2.0, 2.5]);
        assert_eq!(ring.fill(), 0);
    }

    #[test]
    fn overflow_rejects_newest_and_counts() {
        let ring = AudioRing::new(2, 1);
        assert_eq!(ring.write(&[1.0, 2.0, 3.0, 4.0]), 2, "only 2 frames fit");
        assert_eq!(ring.overruns(), 1);
        let mut out = [0.0f32; 4];
        assert_eq!(ring.read_into(&mut out), 2);
        assert_eq!(&out[..2], &[1.0, 2.0]); // newest (3,4) were dropped
    }

    #[test]
    fn underflow_short_read_leaves_tail_and_counts() {
        let ring = AudioRing::new(4, 1);
        ring.write(&[7.0]);
        let mut out = [9.0f32; 3];
        assert_eq!(ring.read_into(&mut out), 1);
        assert_eq!(out, [7.0, 9.0, 9.0]); // tail untouched
        assert_eq!(ring.underruns(), 1);
    }

    #[test]
    fn wraps_around_capacity() {
        let ring = AudioRing::new(3, 1);
        ring.write(&[1.0, 2.0]);
        let mut o = [0.0f32; 2];
        ring.read_into(&mut o); // r now at 2
        ring.write(&[3.0, 4.0, 5.0]); // wraps: writes into slots 2, 0, 1
        assert_eq!(ring.fill(), 3);
        let mut out = [0.0f32; 3];
        assert_eq!(ring.read_into(&mut out), 3);
        assert_eq!(out, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn concurrent_spsc_preserves_order_without_corruption() {
        // Producer writes frames 0..N (frame k = [k, k]); consumer reads them all,
        // each thread spinning on full/empty so nothing is dropped. The consumer
        // must observe exactly 0,1,2,...,N-1 in order — any race would corrupt or
        // reorder a value.
        const N: u64 = 200_000;
        let ring = Arc::new(AudioRing::new(64, 2));
        let prod = {
            let ring = ring.clone();
            std::thread::spawn(move || {
                let mut k = 0u64;
                while k < N {
                    let f = k as f32;
                    if ring.write(&[f, f]) == 1 {
                        k += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            })
        };
        let cons = std::thread::spawn(move || {
            let mut got = 0u64;
            let mut out = [0.0f32; 2];
            while got < N {
                if ring.read_into(&mut out) == 1 {
                    assert_eq!(out[0], got as f32, "frame {got} corrupted/reordered");
                    assert_eq!(out[1], got as f32);
                    got += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
        prod.join().unwrap();
        cons.join().unwrap();
    }
}
