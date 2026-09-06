//! WAV in both directions: a 32-bit float writer whose size fields stay right while it grows, and
//! a reader over the kinds a recording or a sample arrives in.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Bytes before the first sample: RIFF, `fmt ` in the 18-byte form a float file needs, `fact`,
/// and `data`.
const HEADER: u64 = 58;
/// What a RIFF size field holds, less the header: the point a take takes its next part.
const CEILING: u64 = u32::MAX as u64 - HEADER;

fn why(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{}: {e}", path.display())
}

pub struct Writer {
    file: BufWriter<File>,
    pub path: PathBuf,
    pub rate: u32,
    pub channels: u16,
    bytes: u64,
    synced: u64,
    scratch: Vec<u8>,
}

impl Writer {
    pub fn create(path: &Path, rate: u32, channels: u16) -> Result<Writer, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| why(parent, e))?;
        }
        let mut file = BufWriter::new(File::create(path).map_err(|e| why(path, e))?);
        let block = u32::from(channels) * 4;
        let mut h: Vec<u8> = Vec::with_capacity(HEADER as usize);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(HEADER as u32 - 8).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&18u32.to_le_bytes());
        h.extend_from_slice(&3u16.to_le_bytes());
        h.extend_from_slice(&channels.to_le_bytes());
        h.extend_from_slice(&rate.to_le_bytes());
        h.extend_from_slice(&rate.saturating_mul(block).to_le_bytes());
        h.extend_from_slice(&(block as u16).to_le_bytes());
        h.extend_from_slice(&32u16.to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(b"fact");
        h.extend_from_slice(&4u32.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&0u32.to_le_bytes());
        file.write_all(&h).map_err(|e| why(path, e))?;
        let scratch = Vec::new();
        Ok(Writer { file, path: path.to_path_buf(), rate, channels, bytes: 0, synced: 0, scratch })
    }

    /// One planar `[C, T]` chunk, interleaved into the file. `false` says it did not fit under the
    /// RIFF ceiling, and the take carries on in its next part.
    pub fn write(&mut self, planar: &[f32], frames: usize) -> Result<bool, String> {
        let c = self.channels as usize;
        let size = (frames * c * 4) as u64;
        if self.bytes + size > CEILING {
            return Ok(false);
        }
        self.scratch.clear();
        self.scratch.reserve(size as usize);
        for i in 0..frames {
            for ch in 0..c {
                self.scratch.extend_from_slice(&planar[ch * frames + i].to_le_bytes());
            }
        }
        self.file.write_all(&self.scratch).map_err(|e| why(&self.path, e))?;
        self.bytes += size;
        Ok(true)
    }

    pub fn frames(&self) -> u64 {
        self.bytes / (self.channels as u64 * 4)
    }

    /// Patch the three size fields. Called as the take grows too, so a goofi that dies leaves a
    /// file that still plays.
    pub fn sync(&mut self) -> Result<(), String> {
        if self.synced == self.bytes {
            return Ok(());
        }
        self.file.flush().map_err(|e| why(&self.path, e))?;
        let frames = self.frames() as u32;
        let file = self.file.get_mut();
        for (at, v) in [(4u64, (HEADER - 8 + self.bytes) as u32), (46, frames), (54, self.bytes as u32)] {
            file.seek(SeekFrom::Start(at)).map_err(|e| why(&self.path, e))?;
            file.write_all(&v.to_le_bytes()).map_err(|e| why(&self.path, e))?;
        }
        file.seek(SeekFrom::End(0)).map_err(|e| why(&self.path, e))?;
        self.synced = self.bytes;
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

/// What a `fmt ` chunk says one sample is.
#[derive(Clone, Copy)]
enum Kind {
    Int(u16),
    Float,
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::Int(bits) => bits.div_ceil(8) as usize,
            Kind::Float => 4,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Kind::Int(8) => (b[0] as f32 - 128.0) / 128.0,
            Kind::Int(16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            Kind::Int(24) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
            Kind::Int(_) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            Kind::Float => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }
}

pub struct Reader {
    file: BufReader<File>,
    pub path: PathBuf,
    pub rate: u32,
    pub channels: u16,
    pub frames: u64,
    kind: Kind,
    data_at: u64,
    frame: u64,
    scratch: Vec<u8>,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Reader, String> {
        let mut file = BufReader::new(File::open(path).map_err(|e| why(path, e))?);
        let mut riff = [0u8; 12];
        file.read_exact(&mut riff).map_err(|e| why(path, e))?;
        if &riff[..4] != b"RIFF" || &riff[8..] != b"WAVE" {
            return Err(format!("{} is not a WAV file", path.display()));
        }
        let (mut format, mut fmt) = (0u16, None);
        let mut at = 12u64;
        loop {
            let mut head = [0u8; 8];
            file.read_exact(&mut head).map_err(|_| format!("{} ends before its data", path.display()))?;
            let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as u64;
            at += 8;
            if &head[..4] == b"fmt " {
                let mut body = vec![0u8; size as usize];
                file.read_exact(&mut body).map_err(|e| why(path, e))?;
                if body.len() < 16 {
                    return Err(format!("{} has a short fmt chunk", path.display()));
                }
                format = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                // Extensible names the real format in the first two bytes of its sub-format GUID.
                if format == 0xFFFE && body.len() >= 26 {
                    format = u16::from_le_bytes([body[24], body[25]]);
                }
                fmt = Some((channels, rate, bits));
            } else if &head[..4] == b"data" {
                let Some((channels, rate, bits)) = fmt else {
                    return Err(format!("{} has data before its fmt chunk", path.display()));
                };
                let kind = match (format, bits) {
                    (3, 32) => Kind::Float,
                    (1, 8 | 16 | 24 | 32) => Kind::Int(bits),
                    _ => return Err(format!("{} is {bits}-bit of format {format}, which goofi does not read", path.display())),
                };
                if channels == 0 || rate == 0 {
                    return Err(format!("{} names no channels or no rate", path.display()));
                }
                let block = channels as u64 * kind.bytes() as u64;
                return Ok(Reader {
                    file,
                    path: path.to_path_buf(),
                    rate,
                    channels,
                    frames: size / block,
                    kind,
                    data_at: at,
                    frame: 0,
                    scratch: Vec::new(),
                });
            } else {
                file.seek(SeekFrom::Current(size as i64)).map_err(|e| why(path, e))?;
            }
            at += size + (size & 1);
            if size & 1 == 1 {
                file.seek(SeekFrom::Current(1)).map_err(|e| why(path, e))?;
            }
        }
    }

    fn block(&self) -> u64 {
        self.channels as u64 * self.kind.bytes() as u64
    }

    pub fn at(&self) -> u64 {
        self.frame
    }

    pub fn seek(&mut self, frame: u64) -> Result<(), String> {
        let frame = frame.min(self.frames);
        self.file.seek(SeekFrom::Start(self.data_at + frame * self.block())).map_err(|e| why(&self.path, e))?;
        self.frame = frame;
        Ok(())
    }

    /// Up to `frames` frames as one planar `[C, T]` buffer, and how many it held; fewer at the end.
    pub fn read(&mut self, frames: usize) -> Result<(usize, Vec<f32>), String> {
        let want = (frames as u64).min(self.frames - self.frame) as usize;
        if want == 0 {
            return Ok((0, Vec::new()));
        }
        let (c, size) = (self.channels as usize, self.kind.bytes());
        self.scratch.resize(want * c * size, 0);
        let read = self.file.read_exact(&mut self.scratch);
        read.map_err(|e| why(&self.path, e))?;
        let mut planar = vec![0.0f32; want * c];
        for i in 0..want {
            for ch in 0..c {
                let b = (i * c + ch) * size;
                planar[ch * want + i] = self.kind.read(&self.scratch[b..b + size]);
            }
        }
        self.frame += want as u64;
        Ok((want, planar))
    }
}
