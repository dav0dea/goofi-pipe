//! goofi-core — the shared data vocabulary for the Rust engine.
//!
//! `Data` is the unit that flows between nodes: an immutable, `Arc`-backed value
//! (n-d array, string, or recursive table) plus a `Meta` sidecar. In-process, a
//! `Data` crosses a node boundary as a cheap `Clone` (one `Arc` bump) — never a
//! copy or a serialization. Serialization exists only at the browser (GOOF) and
//! subprocess boundaries, in other crates.
//!
//! Construction enforces the invariants the legacy `data.py` enforced:
//! f64 arrays narrow to f32, 0-d arrays promote to 1-d, and per-dim channel
//! coordinate lists must match the array shape. `shape`/`dtype` are DERIVED from
//! the array store and never stored in `Meta` (so array↔meta drift is
//! unrepresentable); the GOOF encoder projects them back into the wire meta dict.

use std::collections::BTreeMap;
use std::sync::Arc;

use indexmap::IndexMap;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoofiError {
    /// A `Data` construction invariant was violated.
    Invalid(String),
}

impl std::fmt::Display for GoofiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GoofiError::Invalid(m) => write!(f, "invalid Data: {m}"),
        }
    }
}

impl std::error::Error for GoofiError {}

pub type Result<T> = std::result::Result<T, GoofiError>;

// ---------------------------------------------------------------------------
// DType — the closed numeric element type set (matches numpy typestrings)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// numpy `dtype.str` — the GOOF ARRAY-body dtype string (and `decode.ts`
    /// contract). Single-byte types use `|`, multi-byte use little-endian `<`.
    pub fn numpy_typestr(self) -> &'static str {
        match self {
            DType::F16 => "<f2",
            DType::F32 => "<f4",
            DType::F64 => "<f8",
            DType::I8 => "|i1",
            DType::I16 => "<i2",
            DType::I32 => "<i4",
            DType::I64 => "<i8",
            DType::U8 => "|u1",
            DType::U16 => "<u2",
            DType::U32 => "<u4",
            DType::U64 => "<u8",
            DType::Bool => "|b1",
        }
    }

    /// numpy `str(dtype)` — the human name projected into `meta["dtype"]`.
    pub fn numpy_name(self) -> &'static str {
        match self {
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::Bool => "bool",
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::F16 | DType::I16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
        }
    }

    /// Parse a numpy typestring (`<f4`, `|u1`, `=i8`, …) — used by the decoder
    /// at the subprocess boundary. Big-endian (`>`) is rejected.
    pub fn from_numpy_typestr(s: &str) -> Option<DType> {
        let bytes = s.as_bytes();
        let core = match bytes.first() {
            Some(b'<') | Some(b'=') | Some(b'|') => &s[1..],
            Some(b'>') => return None,
            _ => s,
        };
        Some(match core {
            "f2" => DType::F16,
            "f4" => DType::F32,
            "f8" => DType::F64,
            "i1" => DType::I8,
            "i2" => DType::I16,
            "i4" => DType::I32,
            "i8" => DType::I64,
            "u1" => DType::U8,
            "u2" => DType::U16,
            "u4" => DType::U32,
            "u8" => DType::U64,
            "b1" => DType::Bool,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Array storage
// ---------------------------------------------------------------------------

/// A Rust-owned, row-major, little-endian, contiguous numeric buffer with a
/// derived shape. The buffer is `Arc`-shared so a numpy view (pyo3) or an audio
/// view can alias it zero-copy without a copy.
#[derive(Clone, Debug)]
pub struct RawArray {
    dtype: DType,
    shape: Vec<usize>,
    buf: Arc<[u8]>,
}

impl RawArray {
    /// Build without normalization — assumes `buf.len() == nelem * itemsize`.
    pub fn new(dtype: DType, shape: Vec<usize>, buf: Arc<[u8]>) -> RawArray {
        RawArray { dtype, shape, buf }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
    pub fn nelem(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Where an array's bytes live. In-process fan-out is an `Arc` clone. A `Py`
/// variant (adopting a numpy-owned buffer) is added with the pyo3 host (M5).
#[derive(Clone, Debug)]
pub enum ArrayStore {
    Rust(Arc<RawArray>),
}

impl ArrayStore {
    pub fn dtype(&self) -> DType {
        match self {
            ArrayStore::Rust(a) => a.dtype(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayStore::Rust(a) => a.shape(),
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ArrayStore::Rust(a) => a.as_bytes(),
        }
    }
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

/// A per-axis coordinate label (electrode name, frequency, time, …).
#[derive(Clone, Debug, PartialEq)]
pub enum Coord {
    Num(f64),
    Str(Arc<str>),
}

/// `channels[dim] -> coord list`, coords `Arc`-shared so large (kHz/HD) axes
/// don't copy on fan-out.
#[derive(Clone, Debug, Default)]
pub struct Channels(pub BTreeMap<usize, Arc<Vec<Coord>>>);

/// An arbitrary meta value (the open map the inspector renders).
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<MetaValue>),
    Map(BTreeMap<String, MetaValue>),
}

/// Typed hot fields (never a hashmap lookup on the tick path) plus an open map
/// for arbitrary keys. `shape`/`dtype` are intentionally absent — they are
/// derived from the array store and projected into the wire meta at encode time.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub sfreq: Option<f64>,
    pub index: Option<u64>,
    pub channels: Channels,
    pub reduced: Option<MetaValue>,
    /// Arbitrary keys, including the `/^__.*__$/` hidden-internal namespace.
    /// Reserved keys (shape/dtype/channels/sfreq/index/reduced) never live here.
    pub extra: BTreeMap<String, MetaValue>,
}

impl Meta {
    pub fn empty() -> Meta {
        Meta::default()
    }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Value {
    Array(ArrayStore),
    Str(Arc<str>),
    Table(Arc<IndexMap<String, Data>>),
}

impl Value {
    /// The GOOF dtype tag byte (0=ARRAY, 1=STRING, 2=TABLE).
    pub fn dtype_tag(&self) -> u8 {
        match self {
            Value::Array(_) => 0,
            Value::Str(_) => 1,
            Value::Table(_) => 2,
        }
    }
}

#[derive(Debug)]
pub struct DataInner {
    pub value: Value,
    pub meta: Meta,
}

/// The immutable, cheaply-cloneable unit of dataflow.
#[derive(Clone, Debug)]
pub struct Data(Arc<DataInner>);

impl Data {
    pub fn value(&self) -> &Value {
        &self.0.value
    }
    pub fn meta(&self) -> &Meta {
        &self.0.meta
    }
    pub fn dtype_tag(&self) -> u8 {
        self.0.value.dtype_tag()
    }

    /// A pre-normalized array `Data` (caller guarantees the invariants).
    pub fn array(store: ArrayStore, meta: Meta) -> Data {
        Data(Arc::new(DataInner {
            value: Value::Array(store),
            meta,
        }))
    }

    pub fn string(s: impl Into<Arc<str>>, meta: Meta) -> Data {
        Data(Arc::new(DataInner {
            value: Value::Str(s.into()),
            meta,
        }))
    }

    pub fn table(map: IndexMap<String, Data>, meta: Meta) -> Data {
        Data(Arc::new(DataInner {
            value: Value::Table(Arc::new(map)),
            meta,
        }))
    }

    /// Build an array `Data` from raw little-endian bytes, applying the legacy
    /// construction invariants: f64 → f32 narrowing, 0-d → 1-d promotion, and
    /// channel coord-length validation against the (post-promotion) shape.
    pub fn from_array_bytes(
        dtype: DType,
        shape: Vec<usize>,
        buf: Vec<u8>,
        meta: Meta,
    ) -> Result<Data> {
        let nelem: usize = shape.iter().product();
        let expect = nelem * dtype.itemsize();
        if buf.len() != expect {
            return Err(GoofiError::Invalid(format!(
                "buffer length {} != nelem {} * itemsize {} = {}",
                buf.len(),
                nelem,
                dtype.itemsize(),
                expect
            )));
        }

        // f64 -> f32 narrowing (never carry a >4-byte float on the wire).
        let (dtype, buf) = if dtype == DType::F64 {
            let mut nb = Vec::with_capacity(buf.len() / 2);
            for chunk in buf.chunks_exact(8) {
                let v = f64::from_le_bytes(chunk.try_into().unwrap()) as f32;
                nb.extend_from_slice(&v.to_le_bytes());
            }
            (DType::F32, nb)
        } else {
            (dtype, buf)
        };

        // 0-d -> 1-d promotion (a scalar becomes a length-1 vector).
        let shape = if shape.is_empty() { vec![1] } else { shape };

        // Validate channel coordinate lengths against the array shape.
        for (&dim, coords) in meta.channels.0.iter() {
            if dim >= shape.len() {
                return Err(GoofiError::Invalid(format!(
                    "channels dim{dim} exceeds ndim {}",
                    shape.len()
                )));
            }
            if coords.len() != shape[dim] {
                return Err(GoofiError::Invalid(format!(
                    "channels dim{dim} has {} coords, expected {}",
                    coords.len(),
                    shape[dim]
                )));
            }
        }

        let raw = RawArray::new(dtype, shape, Arc::from(buf.into_boxed_slice()));
        Ok(Data::array(ArrayStore::Rust(Arc::new(raw)), meta))
    }
}

// ---------------------------------------------------------------------------
// SlotType — the Data kind a slot carries (frontend `input_slots` dtype name)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SlotType {
    Array,
    String,
    Table,
}

impl SlotType {
    /// Frontend-facing name (`input_slots`/`output_slots` value).
    pub fn name(self) -> &'static str {
        match self {
            SlotType::Array => "ARRAY",
            SlotType::String => "STRING",
            SlotType::Table => "TABLE",
        }
    }
    pub fn tag(self) -> u8 {
        match self {
            SlotType::Array => 0,
            SlotType::String => 1,
            SlotType::Table => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Param — the typed parameter descriptors (the `common` scheduling group is
// lifted into RunPolicy elsewhere and is NOT a Param).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Float {
        value: f64,
        vmin: f64,
        vmax: f64,
    },
    Int {
        value: i64,
        vmin: i64,
        vmax: i64,
    },
    Bool {
        value: bool,
    },
    Str {
        value: String,
        options: Option<Vec<String>>,
        /// Name of a node method that re-enumerates `options` (device pickers).
        refresh: Option<&'static str>,
    },
    /// Momentary trigger: `take_trigger` returns the state then resets to false.
    Trigger {
        fired: bool,
    },
}

impl Param {
    pub fn float(value: f64, vmin: f64, vmax: f64) -> Param {
        Param::Float { value, vmin, vmax }
    }
    pub fn int(value: i64, vmin: i64, vmax: i64) -> Param {
        Param::Int { value, vmin, vmax }
    }
    pub fn boolean(value: bool) -> Param {
        Param::Bool { value }
    }
    pub fn str_free(value: impl Into<String>) -> Param {
        Param::Str {
            value: value.into(),
            options: None,
            refresh: None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Param::Float { value, .. } => Some(*value),
            Param::Int { value, .. } => Some(*value as f64),
            Param::Bool { value } => Some(if *value { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Param::Int { value, .. } => Some(*value),
            Param::Float { value, .. } => Some(*value as i64),
            Param::Bool { value } => Some(*value as i64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Param::Bool { value } => Some(*value),
            Param::Trigger { fired } => Some(*fired),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Param::Str { value, .. } => Some(value),
            _ => None,
        }
    }
    /// Consume a momentary trigger: returns whether it fired, resetting to false.
    pub fn take_trigger(&mut self) -> bool {
        if let Param::Trigger { fired } = self {
            let f = *fired;
            *fired = false;
            f
        } else {
            false
        }
    }
}
