//! One device for the whole engine: the adapter that answered, the sampler and the bind group
//! layouts every pipeline shares, and the 1x1 transparent texture an unwired input samples.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Every texture in the engine.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct Gpu {
    pub adapter: String,
    pub backend: String,
    pub sampler: wgpu::Sampler,
    /// What an unwired texture input reads: present, transparent, never an error.
    pub blank: wgpu::TextureView,
    group0: [wgpu::BindGroupLayout; 2],
    group1: Mutex<HashMap<usize, Arc<wgpu::BindGroupLayout>>>,
    layouts: Mutex<HashMap<(bool, usize), Arc<wgpu::PipelineLayout>>>,
    pub queue: wgpu::Queue,
    /// LAST, with the queue before it: fields drop in declaration order, and a resource outliving
    /// the device it was made on is a driver crash rather than an error.
    pub device: wgpu::Device,
}

/// The ONE device this process renders on, opened at the first ask. Not one per engine: an
/// engine's teardown would unload the Vulkan driver under a sibling engine still inside it, and
/// the suite runs many engines at once. Nothing destroys it, so nothing races on its death.
pub fn shared() -> Result<Arc<Gpu>, String> {
    static ONE: OnceLock<Result<Arc<Gpu>, String>> = OnceLock::new();
    ONE.get_or_init(|| Gpu::open().map(Arc::new)).clone()
}

/// Held for EVERY operation on that device: a compile, a tick, a birth, a teardown. The driver
/// on the machine this was written on crashed inside its own shader compiler whenever one thread
/// built a pipeline while another encoded, submitted or freed on the same device, so the engine
/// takes the device one caller at a time. Never taken while the runtime lock is free to be taken
/// after it: the order is runtime, then this.
pub fn gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE: Mutex<()> = Mutex::new(());
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Give a value that owns GPU objects back, behind the gate. A pipeline destroyed while the
/// compile thread builds one is the same collision as any other.
pub fn give_back<T>(x: T) {
    let _gate = gate();
    drop(x);
}

impl Gpu {
    fn open() -> Result<Gpu, String> {
        // No display handle: this engine never opens a window, and asking for one would refuse
        // the device on a headless machine — a server, a CI runner — that has a GPU regardless.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter answered: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("goofi-graphics"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("`{}` refused a device: {e}", info.name))?;
        device.on_uncaptured_error(Arc::new(|e| eprintln!("graphics: {e}")));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("goofi-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blank = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blank"),
            size: one_texel(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            blank.as_image_copy(),
            &[0u8; 8],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None },
            one_texel(),
        );
        let blank = blank.create_view(&Default::default());

        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let layout = |entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries })
        };
        let group0 = [
            layout(&[uniform(0), uniform(1), sampler_entry]),
            layout(&[uniform(0), uniform(1), sampler_entry, uniform(3)]),
        ];
        Ok(Gpu {
            device,
            queue,
            adapter: info.name.clone(),
            backend: info.backend.to_string(),
            sampler,
            blank,
            group0,
            group1: Mutex::new(HashMap::new()),
            layouts: Mutex::new(HashMap::new()),
        })
    }

    /// Group 0 is the frame's own: time, resolution, the sampler, and the params when there are any.
    pub fn group0(&self, params: bool) -> &wgpu::BindGroupLayout {
        &self.group0[usize::from(params)]
    }

    /// Group 1 is the input textures, one layout per count; zero inputs is an empty group, so
    /// every stage binds both groups.
    pub fn group1(&self, inputs: usize) -> Arc<wgpu::BindGroupLayout> {
        self.group1
            .lock()
            .unwrap()
            .entry(inputs)
            .or_insert_with(|| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..inputs as u32)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    })
                    .collect();
                Arc::new(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                }))
            })
            .clone()
    }

    pub fn layout(&self, params: bool, inputs: usize) -> Arc<wgpu::PipelineLayout> {
        if let Some(held) = self.layouts.lock().unwrap().get(&(params, inputs)) {
            return held.clone();
        }
        let group1 = self.group1(inputs);
        let made = Arc::new(self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(self.group0(params)), Some(&group1)],
            immediate_size: 0,
        }));
        self.layouts.lock().unwrap().insert((params, inputs), made.clone());
        made
    }
}

fn one_texel() -> wgpu::Extent3d {
    wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }
}

/// A texture the engine renders into and reads back from.
pub fn target(gpu: &Gpu, label: &str, (w, h): (u32, u32), usage: wgpu::TextureUsages) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    })
}

/// A `copy_texture_to_buffer` row pitch: the GPU pads every row to 256 bytes.
pub fn padded_row(width: u32) -> u32 {
    (width * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}
