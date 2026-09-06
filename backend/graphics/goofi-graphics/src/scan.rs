//! The engine's scan of one `nodes_graphics/` folder, and the thread that builds a pipeline for
//! what it registered. A driver takes its time over a compile, so no op waits on one: the class
//! holds a cell, and the plan picks it up at the tick after it is filled.

use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};

use goofi_node::{Isolation, NodeManifest, Scanned, ScannedType};

use crate::gpu::Gpu;
use crate::{shader, GraphicsEngine};

/// A pipeline once the compile thread answers, or why the device refused it.
pub type Built = Arc<OnceLock<Result<Arc<wgpu::RenderPipeline>, String>>>;

/// One `.wgsl` file as the engine holds it.
pub struct Class {
    pub manifest: &'static NodeManifest,
    pub feedback: bool,
    pub pipeline: Built,
}

pub(crate) fn scan(engine: &mut GraphicsEngine, dir: &Path) -> Vec<ScannedType> {
    let mut out = Vec::new();
    for (path, type_name, stamp) in goofi_node::node_files(dir, "graphics") {
        let outcome = match engine.register(&path, &type_name) {
            Ok(replaced) => Scanned::Registered { isolation: Isolation::Shader, replaced },
            Err(reason) => {
                // A file that no longer loads displaces its registration, so the palette greys the
                // type rather than offering one nothing can build.
                crate::gpu::give_back(engine.classes.remove(type_name.as_str()));
                Scanned::Unavailable(reason)
            }
        };
        out.push(ScannedType { type_name, stamp, outcome });
    }
    out
}

impl GraphicsEngine {
    /// One file: its header is the manifest, its text plus the prelude is what naga judges, and
    /// only then does a pipeline get asked for.
    pub(crate) fn register(&mut self, path: &Path, type_name: &str) -> Result<bool, String> {
        let source = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let intro = shader::header(&source)?;
        if let Some(reason) = goofi_node::illegal_slot(&intro) {
            return Err(reason);
        }
        let manifest = goofi_node::leak_manifest(type_name.to_string(), &intro)?;
        let full = format!("{source}{}", shader::prelude(manifest));
        shader::validate(&full)?;
        let pipeline = self.compiler.build(full, !manifest.params.is_empty(), manifest.inputs.len());
        let class = Arc::new(Class { manifest, feedback: intro.feedback, pipeline });
        let displaced = self.classes.insert(type_name.to_string(), class);
        let replaced = displaced.is_some();
        crate::gpu::give_back(displaced);
        Ok(replaced)
    }
}

struct Job {
    source: String,
    params: bool,
    inputs: usize,
    cell: Built,
    shared: Arc<goofi_control::Shared>,
}

/// Where every pipeline in the process is built: ONE thread, beside the one device, so no op
/// waits on a compile.
fn compiler() -> Option<&'static mpsc::Sender<Job>> {
    static ONE: OnceLock<Option<mpsc::Sender<Job>>> = OnceLock::new();
    ONE.get_or_init(|| {
        let gpu = crate::gpu::shared().ok()?;
        let (jobs, take) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("goofi-graphics-compile".into())
            .spawn(move || {
                while let Ok(job) = take.recv() {
                    let _ = job.cell.set(compile(&gpu, &job));
                    // The tick picks the cell up by itself; the settle is for a refusal, which
                    // only a plan can turn into the node's standing error.
                    job.shared.ask_settle();
                    // The job may hold the last handle on the pipeline it just built.
                    crate::gpu::give_back(job);
                }
            })
            .ok()?;
        Some(jobs)
    })
    .as_ref()
}

/// One engine's end of that queue: the shared state a finished compile must wake.
pub struct Compiler(pub Arc<goofi_control::Shared>);

impl Compiler {
    pub fn build(&self, source: String, params: bool, inputs: usize) -> Built {
        let cell: Built = Arc::new(OnceLock::new());
        let job = Job { source, params, inputs, cell: cell.clone(), shared: self.0.clone() };
        match compiler() {
            Some(jobs) => {
                let _ = jobs.send(job);
            }
            None => {
                let _ = cell.set(Err("no compile thread".into()));
            }
        }
        cell
    }
}

fn compile(gpu: &Gpu, job: &Job) -> Result<Arc<wgpu::RenderPipeline>, String> {
    let _gate = crate::gpu::gate();
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(job.source.as_str().into()),
    });
    let layout = gpu.layout(job.params, job.inputs);
    let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: crate::gpu::FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    match pollster::block_on(scope.pop()) {
        Some(e) => Err(format!("the device refused the pipeline: {e}")),
        None => Ok(Arc::new(pipeline)),
    }
}
