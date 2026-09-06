//! What the render thread owns: the plan, one GPU state per live node, and the tick that draws
//! every demanded stage once and reads back the ones somebody is watching.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use goofi_core::{Data, Meta};
use goofi_node::Uid;

use crate::gpu::{padded_row, target, Gpu};
use crate::half::Upload;
use crate::plan::{Input, Plan};
use crate::shader;

#[derive(Default)]
pub struct Stats {
    pub frames: AtomicU64,
    pub stages: AtomicU64,
    pub tick_max_us: AtomicU64,
}

/// One texture the engine owns, with the size it was made for.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// One node's GPU state, kept across plans so a topology edit costs no allocation.
struct State {
    out: Option<Target>,
    /// The readback, sized with `out`; only a stage somebody watches has one.
    staging: Option<wgpu::Buffer>,
    uploads: Vec<Option<Target>>,
    time: wgpu::Buffer,
    resolution: wgpu::Buffer,
    params: Option<wgpu::Buffer>,
}

pub struct Runtime {
    plan: Plan,
    states: HashMap<Uid, State>,
    started: Instant,
    stats: Arc<Stats>,
    /// Last, for the reason [`Gpu`] states.
    gpu: Arc<Gpu>,
}

impl Runtime {
    pub fn new(gpu: Arc<Gpu>, started: Instant, stats: Arc<Stats>) -> Runtime {
        Runtime { gpu, plan: Plan::default(), states: HashMap::new(), started, stats }
    }

    /// A birth's GPU state. `params` is the byte length of its uniform block, zero for a node
    /// that declares none.
    pub fn insert(&mut self, uid: Uid, params: usize) {
        let _gate = crate::gpu::gate();
        let uniform = |label: &str, size: u64| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let state = State {
            out: None,
            staging: None,
            uploads: Vec::new(),
            time: uniform("time", 4),
            resolution: uniform("resolution", 8),
            params: (params > 0).then(|| uniform("params", params as u64)),
        };
        self.states.insert(uid, state);
    }

    /// A node leaves, and the WHOLE plan goes with it: `Input::Stage` is an index into it. The
    /// settle that ends the batch builds the next one, and until it does this engine draws nothing.
    pub fn remove(&mut self, uid: Uid) {
        let _gate = crate::gpu::gate();
        self.states.remove(&uid);
        self.plan = Plan::default();
    }

    /// Every GPU object this engine holds, given back at once.
    pub fn clear(&mut self) {
        let _gate = crate::gpu::gate();
        self.plan = Plan::default();
        self.states.clear();
    }

    pub fn set_plan(&mut self, plan: Plan) {
        let _gate = crate::gpu::gate();
        self.plan = plan;
    }

    pub fn reset_clock(&mut self, origin: Instant) {
        self.started = origin;
    }

    /// One tick: upload what arrived, write the uniforms, draw every demanded stage, and read
    /// back the ones with a reader.
    pub fn tick(&mut self) {
        let began = Instant::now();
        let t = self.started.elapsed().as_secs_f32();
        let want = self.plan.demanded();
        if !want.contains(&true) {
            self.stats.frames.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let _gate = crate::gpu::gate();
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let mut readbacks: Vec<usize> = Vec::new();
        for (i, drawn) in want.iter().enumerate() {
            if !drawn {
                continue;
            }
            let stage = &self.plan.stages[i];
            let Some(Ok(pipeline)) = stage.pipeline.get() else { continue };
            let Some(state) = self.states.get_mut(&stage.uid) else { continue };
            state.ensure_out(&self.gpu, stage.size, stage.readers.load(Ordering::Relaxed));
            for (k, cell) in stage.uploads.iter().enumerate() {
                if let Some(up) = cell.lock().unwrap().take() {
                    state.upload(&self.gpu, k, &up);
                }
            }
            self.gpu.queue.write_buffer(&state.time, 0, &t.to_le_bytes());
            let res = [(stage.size.0 as f32).to_le_bytes(), (stage.size.1 as f32).to_le_bytes()].concat();
            self.gpu.queue.write_buffer(&state.resolution, 0, &res);
            if let Some(buf) = &state.params {
                self.gpu.queue.write_buffer(buf, 0, &shader::uniform_bytes(stage.decls, &stage.params));
            }
            // Cloned handles, so reading another stage's output ends the borrow of `states`.
            let views: Vec<wgpu::TextureView> = stage
                .inputs
                .iter()
                .map(|input| match input {
                    Input::Stage(j) => self
                        .states
                        .get(&self.plan.stages[*j].uid)
                        .and_then(|s| s.out.as_ref())
                        .map_or(&self.gpu.blank, |t| &t.view),
                    Input::Upload(k) => self.states[&stage.uid]
                        .uploads
                        .get(*k)
                        .and_then(|u| u.as_ref())
                        .map_or(&self.gpu.blank, |t| &t.view),
                    Input::None => &self.gpu.blank,
                })
                .cloned()
                .collect();
            let state = &self.states[&stage.uid];
            let group0 = state.group0(&self.gpu);
            let group1 = state.group1(&self.gpu, &views);
            let out = state.out.as_ref().expect("ensure_out made it");
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &out.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group0, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.draw(0..3, 0..1);
            }
            self.stats.stages.fetch_add(1, Ordering::Relaxed);
            if let Some(buffer) = &state.staging {
                let (w, h) = out.size;
                encoder.copy_texture_to_buffer(
                    out.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(padded_row(w)),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                readbacks.push(i);
            }
        }
        self.gpu.queue.submit([encoder.finish()]);
        for &i in &readbacks {
            let uid = self.plan.stages[i].uid;
            if let Some(b) = self.states[&uid].staging.as_ref() {
                b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            }
        }
        let _ = self.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        for &i in &readbacks {
            let stage = &self.plan.stages[i];
            let state = &self.states[&stage.uid];
            let (Some(buffer), Some(out)) = (state.staging.as_ref(), state.out.as_ref()) else { continue };
            let frame = read_back(buffer, out.size);
            buffer.unmap();
            if let Some(frame) = frame {
                *stage.tap.lock().unwrap() = Some(frame);
            }
        }
        self.stats.frames.fetch_add(1, Ordering::Relaxed);
        self.stats.tick_max_us.fetch_max(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// The mapped rows as one `[H, W, 4]` f32 frame, the padding each row carries dropped.
fn read_back(buffer: &wgpu::Buffer, (w, h): (u32, u32)) -> Option<Data> {
    let mapped = buffer.slice(..).get_mapped_range().ok()?;
    let pitch = padded_row(w) as usize;
    let row = w as usize * 8;
    let mut bytes = Vec::with_capacity(w as usize * h as usize * 16);
    for y in 0..h as usize {
        let line = mapped.get(y * pitch..y * pitch + row)?;
        for texel in line.chunks_exact(2) {
            let v = half::f16::from_bits(u16::from_le_bytes([texel[0], texel[1]])).to_f32();
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    drop(mapped);
    Data::array_f32(vec![h as usize, w as usize, 4], bytes, Meta::new()).ok()
}

impl State {
    /// The output texture at `size`, and its staging buffer while somebody reads it. Both are
    /// remade when the size moves, which is what loses a feedback chain its history.
    fn ensure_out(&mut self, gpu: &Gpu, size: (u32, u32), readers: bool) {
        if self.out.as_ref().is_none_or(|t| t.size != size) {
            let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC;
            let texture = target(gpu, "out", size, usage);
            let view = texture.create_view(&Default::default());
            self.out = Some(Target { texture, view, size });
            self.staging = None;
        }
        match (readers, self.staging.is_some()) {
            (true, false) => {
                self.staging = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: u64::from(padded_row(size.0)) * u64::from(size.1),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            (false, true) => self.staging = None,
            _ => {}
        }
    }

    /// One arrival into the texture its input samples.
    fn upload(&mut self, gpu: &Gpu, k: usize, up: &Upload) {
        if self.uploads.len() <= k {
            self.uploads.resize_with(k + 1, || None);
        }
        let size = (up.width, up.height);
        if self.uploads[k].as_ref().is_none_or(|t| t.size != size) {
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            let texture = target(gpu, "upload", size, usage);
            let view = texture.create_view(&Default::default());
            self.uploads[k] = Some(Target { texture, view, size });
        }
        let held = self.uploads[k].as_ref().expect("just made");
        let bytes: Vec<u8> = up.texels.iter().flat_map(|t| t.to_le_bytes()).collect();
        gpu.queue.write_texture(
            held.texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(up.width * 8), rows_per_image: None },
            wgpu::Extent3d { width: up.width, height: up.height, depth_or_array_layers: 1 },
        );
    }

    fn group0(&self, gpu: &Gpu) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.time.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.resolution.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
        ];
        if let Some(p) = &self.params {
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: gpu.group0(self.params.is_some()),
            entries: &entries,
        })
    }

    fn group1(&self, gpu: &Gpu, views: &[wgpu::TextureView]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(v) })
            .collect();
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &gpu.group1(views.len()),
            entries: &entries,
        })
    }
}

/// A stage's uniform block length, measured by the writer so there is one layout.
pub fn params_len(decls: &[goofi_node::ParamDecl]) -> usize {
    let zeros: Vec<AtomicU64> = decls.iter().map(|_| AtomicU64::new(0)).collect();
    shader::uniform_bytes(decls, &zeros).len()
}
