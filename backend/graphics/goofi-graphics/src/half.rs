//! The graphics engine's half of a node's control thread: an arrival becomes texels the render
//! thread uploads, and the frame that thread read back goes out while anyone drinks from it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use goofi_control::{Cx, Half, Ticked};
use goofi_core::{Data, Value};

/// One arrival as the render thread takes it: `width * height * 4` f16 texels, row 0 the top.
pub struct Upload {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<u16>,
}

impl Upload {
    /// A frame as RGBA texels, unclamped. `[N]` is one row; `[H, W]` is gray; `[H, W, C]` fills
    /// the channels it has, with alpha 1 where it has none.
    pub fn of(frame: &Data) -> Option<Upload> {
        let Value::Array(a) = frame.value() else { return None };
        let (h, w, c) = match *a.shape() {
            [n] => (1, n, 1),
            [h, w] => (h, w, 1),
            [h, w, c] if (1..=4).contains(&c) => (h, w, c),
            _ => return None,
        };
        if h == 0 || w == 0 {
            return None;
        }
        let x: Vec<f32> =
            a.as_bytes().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
        let mut texels = Vec::with_capacity(h * w * 4);
        for i in 0..h * w {
            let s = &x[i * c..(i + 1) * c];
            let rgba = match c {
                1 => [s[0], s[0], s[0], 1.0],
                2 => [s[0], s[0], s[0], s[1]],
                3 => [s[0], s[1], s[2], 1.0],
                _ => [s[0], s[1], s[2], s[3]],
            };
            texels.extend(rgba.iter().map(|v| half::f16::from_f32(if v.is_finite() { *v } else { 0.0 }).to_bits()));
        }
        Some(Upload { width: w as u32, height: h as u32, texels })
    }
}

pub struct GraphicsHalf {
    pub uploads: Vec<Arc<Mutex<Option<Upload>>>>,
    pub readers: Arc<AtomicBool>,
    pub tap: Arc<Mutex<Option<Data>>>,
}

impl Half for GraphicsHalf {
    /// An arrival replaces whatever the render thread has not taken yet: latest wins, as every
    /// crossing into a scheduled engine is.
    fn arrive(&mut self, inbox: usize, frame: &Data) -> bool {
        let Some(cell) = self.uploads.get(inbox) else { return false };
        if let Some(up) = Upload::of(frame) {
            *cell.lock().unwrap() = Some(up);
        }
        false
    }

    fn tick(&mut self, cx: &Cx<'_>, publish: &mut dyn FnMut(usize, &[u8])) -> Ticked {
        // What the render thread reads to decide whether this node runs at all.
        let readers = cx.readers.first().copied().unwrap_or(false);
        self.readers.store(readers, Ordering::Relaxed);
        if readers {
            if let Some(frame) = self.tap.lock().unwrap().take() {
                publish(0, &goofi_codec::encode(&frame));
            }
        }
        Ticked::default()
    }
}
