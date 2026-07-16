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

    /// A copy of this `Data` with `meta.index` set to `index`, sharing the value
    /// buffer (an `Arc` bump — never a payload copy). Only the small `Meta` sidecar
    /// is cloned. The engine stamps the continuity index this way after a node emits.
    pub fn with_index(&self, index: u64) -> Data {
        let mut meta = self.0.meta.clone();
        meta.index = Some(index);
        Data(Arc::new(DataInner {
            value: self.0.value.clone(),
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
        // Checked arithmetic: a hostile/garbled shape (e.g. from a decoded frame)
        // could otherwise overflow `usize` and wrap `expect` to a small value that
        // spuriously matches a short buffer. Overflow is a clean error, never a wrap.
        let nelem: usize = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| GoofiError::Invalid("array element count overflows usize".into()))?;
        let expect = nelem
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| GoofiError::Invalid("array byte length overflows usize".into()))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_projections_and_parsing() {
        assert_eq!(DType::F32.numpy_typestr(), "<f4");
        assert_eq!(DType::U8.numpy_typestr(), "|u1");
        assert_eq!(DType::Bool.numpy_typestr(), "|b1");
        assert_eq!(DType::F16.numpy_typestr(), "<f2");
        assert_eq!(DType::I64.numpy_name(), "int64");
        assert_eq!(DType::F32.itemsize(), 4);
        assert_eq!(DType::from_numpy_typestr("<f4"), Some(DType::F32));
        assert_eq!(DType::from_numpy_typestr("|u1"), Some(DType::U8));
        assert_eq!(DType::from_numpy_typestr(">f4"), None); // big-endian rejected
        assert_eq!(DType::from_numpy_typestr("<x9"), None);
    }

    #[test]
    fn f64_array_narrows_to_f32() {
        let buf: Vec<u8> = [1.5f64, 2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let d = Data::from_array_bytes(DType::F64, vec![2], buf, Meta::empty()).unwrap();
        let Value::Array(s) = d.value() else { panic!() };
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.as_bytes().len(), 8); // 2 x f32
        assert_eq!(f32::from_le_bytes(s.as_bytes()[0..4].try_into().unwrap()), 1.5);
    }

    #[test]
    fn zero_d_promotes_to_1d() {
        let d = Data::from_array_bytes(DType::F32, vec![], 3.0f32.to_le_bytes().to_vec(), Meta::empty())
            .unwrap();
        let Value::Array(s) = d.value() else { panic!() };
        assert_eq!(s.shape(), &[1]);
    }

    #[test]
    fn wrong_buffer_length_errors() {
        assert!(Data::from_array_bytes(DType::F32, vec![3], vec![0u8; 8], Meta::empty()).is_err());
    }

    #[test]
    fn channel_length_must_match_shape() {
        let mut ch = BTreeMap::new();
        ch.insert(0usize, Arc::new(vec![Coord::Str("a".into())])); // len 1
        let meta = Meta {
            channels: Channels(ch),
            ..Default::default()
        };
        let buf: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(); // shape[0]=2
        assert!(Data::from_array_bytes(DType::F32, vec![2], buf, meta).is_err());
    }

    #[test]
    fn param_accessors_and_trigger_consume() {
        assert_eq!(Param::float(2.5, 0.0, 10.0).as_f64(), Some(2.5));
        assert_eq!(Param::int(4, 0, 10).as_i64(), Some(4));
        assert_eq!(Param::boolean(true).as_bool(), Some(true));
        assert_eq!(Param::str_free("hi").as_str(), Some("hi"));
        let mut t = Param::Trigger { fired: true };
        assert!(t.take_trigger());
        assert!(!t.take_trigger()); // consumed on read
    }

    #[test]
    fn slot_type_names_and_tags() {
        assert_eq!(SlotType::Array.name(), "ARRAY");
        assert_eq!(SlotType::String.name(), "STRING");
        assert_eq!(SlotType::Table.tag(), 2);
    }

    #[test]
    fn with_index_sets_meta_and_shares_buffer() {
        let d = Data::from_array_bytes(DType::F32, vec![2], vec![0u8; 8], Meta::empty()).unwrap();
        assert_eq!(d.meta().index, None);
        let stamped = d.with_index(7);
        assert_eq!(stamped.meta().index, Some(7));
        assert_eq!(d.meta().index, None, "original is untouched (immutable)");
        // The value buffer is shared, not copied (Arc bump).
        if let (Value::Array(a), Value::Array(b)) = (d.value(), stamped.value()) {
            assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
        } else {
            panic!()
        }
    }

    #[test]
    fn data_fan_out_is_arc_clone() {
        let d = Data::from_array_bytes(DType::F32, vec![2], vec![0u8; 8], Meta::empty()).unwrap();
        let d2 = d.clone();
        // Both refer to the same underlying buffer (zero-copy fan-out).
        if let (Value::Array(a), Value::Array(b)) = (d.value(), d2.value()) {
            assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
        } else {
            panic!()
        }
    }
}
