//! goofi-subproc — the subprocess node tier (Pathway C).
//!
//! A [`RemoteNode`] runs a Python node in an isolated **GIL** interpreter, one
//! process per node. It exists for two reasons: (1) deps that aren't
//! free-threading-safe can't run in the in-process pyo3 host; (2) per the latency
//! finding, a *separate interpreter* has its own object ownership, so parallel
//! heavy-Python compute avoids free-threaded CPython's biased-refcount penalty.
//!
//! Each tick is one request/response: the input `Data` is GOOF-encoded
//! ([`goofi_codec::encode`]), length-prefixed, and written to the child's stdin;
//! the child runs the user `process(x)` and writes back a GOOF frame we decode
//! ([`goofi_codec::decode`]). Transport is pipes in this first cut; iceoryx2 SHM
//! is the intended production transport (the protocol is transport-agnostic).
//!
//! The node implements the same [`Node`] trait as native and in-process Python
//! nodes, so the scheduler never branches on backend and the engine hosts it
//! through the ordinary `register_dyn_type` seam.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use goofi_node::{Inputs, Node, NodeCtx, NodeResult, Outputs};

/// The Python worker: a self-contained GOOF-array codec (meta passed through
/// opaquely — the Rust decoder re-derives shape/dtype from the body) that runs
/// the user's `process(x)` supplied via the `GOOFI_USER_SRC` env var. Needs only
/// numpy (no msgpack — meta bytes are never parsed here).
const WORKER_SRC: &str = r#"
import sys, os, struct
import numpy as np

MAGIC = b'GOOF'; VER = 2

def read_exact(f, n):
    b = b''
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b

def decode_array(frame):
    assert frame[:4] == MAGIC and frame[4] == VER and frame[5] == 0, "not a GOOF array frame"
    ml = struct.unpack('<I', frame[6:10])[0]
    bl = struct.unpack('<I', frame[10:14])[0]
    meta = frame[14:14 + ml]
    body = frame[14 + ml:14 + ml + bl]
    ndim = body[0]; dl = body[1]; off = 2
    dtype = body[off:off + dl].decode(); off += dl
    shape = [struct.unpack('<I', body[off + 4 * i:off + 4 * i + 4])[0] for i in range(ndim)]
    off += 4 * ndim
    arr = np.frombuffer(body[off:], dtype=np.dtype(dtype)).reshape(shape).copy()
    return arr, meta

def encode_array(arr, meta):
    arr = np.ascontiguousarray(arr)
    dtype = arr.dtype.str.encode()
    body = bytes([arr.ndim, len(dtype)]) + dtype
    for d in arr.shape:
        body += struct.pack('<I', d)
    body += arr.tobytes()
    hdr = MAGIC + bytes([VER, 0]) + struct.pack('<I', len(meta)) + struct.pack('<I', len(body))
    return hdr + meta + body

ns = {}
exec(os.environ['GOOFI_USER_SRC'], ns)
process = ns['process']

inp = sys.stdin.buffer
outp = sys.stdout.buffer
while True:
    hdr = read_exact(inp, 4)
    if hdr is None:
        break
    n = struct.unpack('<I', hdr)[0]
    frame = read_exact(inp, n)
    arr, meta = decode_array(frame)
    res = np.ascontiguousarray(np.asarray(process(arr), dtype=arr.dtype)).ravel()
    out = encode_array(res, meta)  # opaque meta reused; decoder re-derives shape/dtype
    outp.write(struct.pack('<I', len(out)))
    outp.write(out)
    outp.flush()
"#;

/// The spawned child + its pipes.
struct Running {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

/// A Python node running in an isolated GIL subprocess. Construction is cheap and
/// infallible ([`RemoteNode::new`]); the subprocess is spawned lazily on the first
/// `process` (so a discovery factory never panics, and a spawn failure surfaces on
/// the node's error channel instead of crashing the graph).
pub struct RemoteNode {
    python: String,
    source: String,
    proc: Option<Running>,
}

impl RemoteNode {
    /// A remote node backed by `python` running `source` (defining
    /// `process(x) -> array-like`). No process spawns until the first tick.
    pub fn new(python: impl Into<String>, source: impl Into<String>) -> RemoteNode {
        RemoteNode {
            python: python.into(),
            source: source.into(),
            proc: None,
        }
    }

    /// Eagerly spawn (convenience for direct use / tests). Returns the spawn error.
    pub fn spawn(python: &str, source: &str) -> std::io::Result<RemoteNode> {
        let mut node = RemoteNode::new(python, source);
        node.ensure().map_err(std::io::Error::other)?;
        Ok(node)
    }

    fn ensure(&mut self) -> std::result::Result<&mut Running, String> {
        if self.proc.is_none() {
            let mut child = Command::new(&self.python)
                .arg("-c")
                .arg(WORKER_SRC)
                .env("GOOFI_USER_SRC", &self.source)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("spawn `{}`: {e}", self.python))?;
            let stdin = child.stdin.take().expect("piped stdin");
            let stdout = child.stdout.take().expect("piped stdout");
            self.proc = Some(Running { child, stdin, stdout });
        }
        Ok(self.proc.as_mut().unwrap())
    }
}

impl Running {
    /// Send one length-prefixed frame and read the length-prefixed response.
    fn roundtrip(&mut self, frame: &[u8]) -> std::io::Result<Vec<u8>> {
        self.stdin.write_all(&(frame.len() as u32).to_le_bytes())?;
        self.stdin.write_all(frame)?;
        self.stdin.flush()?;
        let mut lenb = [0u8; 4];
        self.stdout.read_exact(&mut lenb)?;
        let n = u32::from_le_bytes(lenb) as usize;
        let mut buf = vec![0u8; n];
        self.stdout.read_exact(&mut buf)?;
        Ok(buf)
    }
}

impl Node for RemoteNode {
    fn process(&mut self, inp: &Inputs<'_>, out: &mut Outputs<'_>, _c: &mut NodeCtx) -> NodeResult {
        let Some(d) = inp.get("data") else {
            return Ok(());
        };
        let frame = goofi_codec::encode(d);
        let running = self.ensure()?;
        let resp = running
            .roundtrip(&frame)
            .map_err(|e| format!("subprocess io: {e}"))?;
        let data = goofi_codec::decode(&resp)?;
        out.set("out", data);
        Ok(())
    }

    fn terminate(&mut self) {
        if let Some(mut p) = self.proc.take() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}

impl Drop for RemoteNode {
    fn drop(&mut self) {
        if let Some(p) = self.proc.as_mut() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery — turn a directory of `process(x)` files into subprocess node types
// the engine hosts via `register_dyn_type` (mirrors `goofi_py::discover`, but the
// factory spawns a RemoteNode instead of an in-process PyNode).
// ---------------------------------------------------------------------------

use goofi_node::{Isolation, NodeManifest, OutputDecl, ParamGroups, SlotDecl};

static PY_IN: &[SlotDecl] = &[SlotDecl {
    name: "data",
    kind: goofi_core::SlotType::Array,
    trigger_process: true,
}];
static PY_OUT: &[OutputDecl] = &[OutputDecl {
    name: "out",
    kind: goofi_core::SlotType::Array,
    length_preserving: true,
}];
fn sp_params() -> ParamGroups {
    ParamGroups::new()
}
fn sp_stub_make(_: &ParamGroups) -> Box<dyn Node> {
    unreachable!("a discovered subprocess node is built by its factory")
}

/// Builds a node instance (spawns lazily on first tick).
pub type SubprocFactory = Box<dyn Fn(&ParamGroups) -> Box<dyn Node> + Send + Sync>;

/// A discovered subprocess node type, ready to `register_dyn_type` into a Graph.
pub struct SubprocNodeType {
    pub manifest: &'static NodeManifest,
    pub factory: SubprocFactory,
}

/// `snake_case` file stem -> `CamelCase` type name (matches `goofi_py::discover`
/// so the same file yields the same type name whichever backend hosts it).
fn camel(stem: &str) -> String {
    stem.split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Scan `dir` for `*.py` node files (skipping `_`-prefixed) that define
/// `process`, returning subprocess-backed types that run on `python`.
pub fn discover(dir: &std::path::Path, python: &str) -> std::io::Result<Vec<SubprocNodeType>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    let mut out = Vec::new();
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.starts_with('_') {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Cheap guard: must plausibly define `process` (a missing one would only
        // fail once the subprocess is spawned and asked to run).
        if !source.contains("def process") {
            continue;
        }

        let type_name: &'static str = Box::leak(camel(stem).into_boxed_str());
        let doc: &'static str =
            Box::leak(format!("Subprocess Python node from {}", path.display()).into_boxed_str());
        let manifest: &'static NodeManifest = Box::leak(Box::new(NodeManifest {
            type_name,
            category: "subprocess",
            doc,
            inputs: PY_IN,
            outputs: PY_OUT,
            default_params: sp_params,
            isolation: Isolation::Subprocess,
            make: sp_stub_make,
        }));
        let python = python.to_string();
        let factory: SubprocFactory =
            Box::new(move |_p| Box::new(RemoteNode::new(&python, &source)) as Box<dyn Node>);
        out.push(SubprocNodeType { manifest, factory });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use goofi_core::{DType, Data, Meta, Value};
    use indexmap::IndexMap;

    /// A python3 with numpy, or None (the subprocess tier test is skipped then).
    fn usable_python() -> Option<String> {
        for cand in ["python3", "python"] {
            if let Ok(out) = Command::new(cand)
                .arg("-c")
                .arg("import numpy")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
            {
                if out.success() {
                    return Some(cand.to_string());
                }
            }
        }
        None
    }

    fn run(node: &mut RemoteNode, d: Data) -> Data {
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx).expect("remote process");
        }
        outmap.get("out").unwrap().clone().expect("output frame")
    }

    fn floats(d: &Data) -> Vec<f32> {
        match d.value() {
            Value::Array(s) => s
                .as_bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => panic!("not array"),
        }
    }

    #[test]
    fn remote_node_runs_a_python_node_in_a_subprocess() {
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy available");
            return;
        };
        let mut node =
            RemoteNode::spawn(&py, "def process(x):\n    return x * 2.0 + 1.0\n").unwrap();

        // Input carries sfreq/index; the worker passes meta through opaquely.
        let mut meta = Meta::empty();
        meta.sfreq = Some(128.0);
        meta.index = Some(7);
        let buf: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![3], buf, meta).unwrap();

        // Two ticks on the same long-lived subprocess (proves it stays alive).
        let out1 = run(&mut node, d.clone());
        assert_eq!(floats(&out1), vec![3.0, 5.0, 7.0]);
        let out2 = run(&mut node, d);
        assert_eq!(floats(&out2), vec![3.0, 5.0, 7.0]);

        // Meta (sfreq/index) survived the opaque round-trip; dtype stayed f32.
        assert_eq!(out1.meta().sfreq, Some(128.0));
        assert_eq!(out1.meta().index, Some(7));
        match out1.value() {
            Value::Array(s) => {
                assert_eq!(s.dtype(), DType::F32);
                assert_eq!(s.shape(), &[3]);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn camel_matches_the_inprocess_naming() {
        assert_eq!(camel("triple"), "Triple");
        assert_eq!(camel("my_band_filter"), "MyBandFilter");
    }

    #[test]
    fn discover_yields_subprocess_types_that_run() {
        let Some(py) = usable_python() else {
            eprintln!("SKIP: no python3 with numpy available");
            return;
        };
        let dir = std::env::temp_dir().join(format!("goofi_subdisc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("negate.py"), "def process(x):\n    return -x\n").unwrap();
        std::fs::write(dir.join("_hidden.py"), "def process(x):\n    return x\n").unwrap();
        std::fs::write(dir.join("nope.py"), "x = 1\n").unwrap();

        let types = discover(&dir, &py).unwrap();
        let names: Vec<&str> = types.iter().map(|t| t.manifest.type_name).collect();
        assert_eq!(names, vec!["Negate"]);
        assert_eq!(types[0].manifest.category, "subprocess");
        assert_eq!(types[0].manifest.isolation, Isolation::Subprocess);

        // The factory builds a working node.
        let mut node = (types[0].factory)(&ParamGroups::new());
        let buf: Vec<u8> = [1.0f32, -2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F32, vec![2], buf, Meta::empty()).unwrap();
        let mut inmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        inmap.insert("data", Some(d));
        let inp = Inputs::new(&inmap);
        let mut outmap: IndexMap<&'static str, Option<Data>> = IndexMap::new();
        outmap.insert("out", None);
        let mut ctx = NodeCtx::new();
        {
            let mut out = Outputs::new(&mut outmap);
            node.process(&inp, &mut out, &mut ctx).unwrap();
        }
        let got = floats(outmap.get("out").unwrap().as_ref().unwrap());
        assert_eq!(got, vec![-1.0, 2.0]);
        node.terminate();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
