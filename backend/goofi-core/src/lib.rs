//! The shared data vocabulary: `Data` — an immutable `Arc`-backed value (array, string or table)
//! plus a `Meta` sidecar. Every array is f32; `shape`/`dtype` are derived, never stored.

use std::collections::BTreeMap;
use std::sync::Arc;

use indexmap::IndexMap;

pub mod globals;
pub mod home;
pub mod path;
pub mod probe;
pub mod reduce;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoofiError {
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

/// An ingest-only source dtype, used to cast a foreign array to f32; never stored on a `Data`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SrcDtype {
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

impl SrcDtype {
    pub fn itemsize(self) -> usize {
        match self {
            SrcDtype::Bool | SrcDtype::I8 | SrcDtype::U8 => 1,
            SrcDtype::F16 | SrcDtype::I16 | SrcDtype::U16 => 2,
            SrcDtype::F32 | SrcDtype::I32 | SrcDtype::U32 => 4,
            SrcDtype::F64 | SrcDtype::I64 | SrcDtype::U64 => 8,
        }
    }

    pub fn numpy_name(self) -> &'static str {
        match self {
            SrcDtype::F16 => "float16",
            SrcDtype::F32 => "float32",
            SrcDtype::F64 => "float64",
            SrcDtype::I8 => "int8",
            SrcDtype::I16 => "int16",
            SrcDtype::I32 => "int32",
            SrcDtype::I64 => "int64",
            SrcDtype::U8 => "uint8",
            SrcDtype::U16 => "uint16",
            SrcDtype::U32 => "uint32",
            SrcDtype::U64 => "uint64",
            SrcDtype::Bool => "bool",
        }
    }

    /// Parse a numpy typestring (`<f4`, `|u1`, `=i8`, …). Big-endian (`>`) rejected.
    pub fn from_numpy_typestr(s: &str) -> Option<SrcDtype> {
        let core = match s.as_bytes().first() {
            Some(b'<') | Some(b'=') | Some(b'|') => &s[1..],
            Some(b'>') => return None,
            _ => s,
        };
        Some(match core {
            "f2" => SrcDtype::F16,
            "f4" => SrcDtype::F32,
            "f8" => SrcDtype::F64,
            "i1" => SrcDtype::I8,
            "i2" => SrcDtype::I16,
            "i4" => SrcDtype::I32,
            "i8" => SrcDtype::I64,
            "u1" => SrcDtype::U8,
            "u2" => SrcDtype::U16,
            "u4" => SrcDtype::U32,
            "u8" => SrcDtype::U64,
            "b1" => SrcDtype::Bool,
            _ => return None,
        })
    }

    fn read_f32(self, b: &[u8], i: usize) -> f32 {
        let sz = self.itemsize();
        let s = &b[i * sz..i * sz + sz];
        match self {
            SrcDtype::F32 => f32::from_le_bytes(s.try_into().unwrap()),
            SrcDtype::F64 => f64::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::F16 => f16_to_f32(u16::from_le_bytes(s.try_into().unwrap())),
            SrcDtype::I8 => s[0] as i8 as f32,
            SrcDtype::I16 => i16::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::I32 => i32::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::I64 => i64::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::U8 | SrcDtype::Bool => s[0] as f32,
            SrcDtype::U16 => u16::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::U32 => u32::from_le_bytes(s.try_into().unwrap()) as f32,
            SrcDtype::U64 => u64::from_le_bytes(s.try_into().unwrap()) as f32,
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f32;
    let val = if exp == 0 {
        frac * (2f32).powi(-24) // subnormal / zero
    } else if exp == 0x1f {
        if frac == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac / 1024.0) * (2f32).powi(exp - 15)
    };
    if sign == 1 { -val } else { val }
}

/// Reinterpret a foreign little-endian buffer as f32 LE bytes; the flag is false when no cast ran.
pub fn cast_to_f32(src: SrcDtype, bytes: &[u8]) -> Result<(Vec<u8>, bool)> {
    let sz = src.itemsize();
    if !bytes.len().is_multiple_of(sz) {
        return Err(GoofiError::Invalid(format!(
            "buffer length {} is not a multiple of {sz}-byte {src:?}",
            bytes.len()
        )));
    }
    if src == SrcDtype::F32 {
        return Ok((bytes.to_vec(), false));
    }
    let n = bytes.len() / sz;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        out.extend_from_slice(&src.read_f32(bytes, i).to_le_bytes());
    }
    Ok((out, true))
}

/// Warn once per source dtype that `slot`'s output was cast to f32; true on the tick it warns.
pub fn warn_cast_once(warned: &mut std::collections::HashSet<SrcDtype>, slot: &str, src: SrcDtype) -> bool {
    if src == SrcDtype::F32 || !warned.insert(src) {
        return false;
    }
    eprintln!(
        "warning: node output slot `{slot}` produced {} — cast to float32 (further {} casts suppressed)",
        src.numpy_name(),
        src.numpy_name(),
    );
    true
}

/// A row-major, little-endian, contiguous f32 array; the buffer is `Arc`-shared, so fan-out
/// is a refcount bump and a numpy or audio view can alias it zero-copy.
#[derive(Clone, Debug)]
pub struct ArrayStore {
    shape: Vec<usize>,
    buf: Arc<[u8]>, // f32 LE, `buf.len() == nelem * 4`
}

impl ArrayStore {
    /// Build without normalization — the caller guarantees `buf.len() == nelem * 4`.
    pub fn new(shape: Vec<usize>, buf: Arc<[u8]>) -> ArrayStore {
        ArrayStore { shape, buf }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// A per-axis coordinate label (electrode name, frequency, time, …).
#[derive(Clone, Debug, PartialEq)]
pub enum Coord {
    Num(f64),
    Str(Arc<str>),
}

/// Labels for one array dimension; `coords`, when present, has one entry per index along it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Axis {
    pub coords: Option<Arc<[Coord]>>,
}

impl Axis {
    pub fn coords(c: impl Into<Arc<[Coord]>>) -> Axis {
        Axis { coords: Some(c.into()) }
    }
    pub fn is_empty(&self) -> bool {
        self.coords.is_none()
    }
}

/// Positional per-dimension labels; trailing unlabeled dimensions may be omitted (`len <= ndim`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Axes(pub Vec<Axis>);

impl Axes {
    pub fn new() -> Axes {
        Axes(Vec::new())
    }
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Axis::is_empty)
    }
    /// The ONE wire spelling of positional axes: a `dimN` key per labeled dimension. Every
    /// projection (msgpack, Python dict, JSON) maps these pairs into its own value type.
    pub fn dims(&self) -> impl Iterator<Item = (String, &[Coord])> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, a)| Some((format!("dim{i}"), &**a.coords.as_ref()?)))
    }
    pub fn get(&self, dim: usize) -> Option<&Axis> {
        self.0.get(dim)
    }
    /// Set dimension `dim` to `axis`, padding intermediate dimensions with empty axes.
    pub fn with(mut self, dim: usize, axis: Axis) -> Axes {
        if self.0.len() <= dim {
            self.0.resize(dim + 1, Axis::default());
        }
        self.0[dim] = axis;
        self
    }

    /// Subset dimension `dim`'s coords to `indices`; a missing index is skipped.
    pub fn sliced(&self, dim: usize, indices: &[usize]) -> Axes {
        let mut v = self.0.clone();
        if let Some(a) = v.get_mut(dim) {
            if let Some(c) = &a.coords {
                let picked: Vec<Coord> = indices.iter().filter_map(|&i| c.get(i).cloned()).collect();
                a.coords = Some(picked.into());
            }
        }
        Axes(v)
    }
}

/// A meta value. serde is `untagged`, so a value serializes as itself; `Bytes`/`Axes` are skipped
/// because pymod maps `channels` with its own `{dimN:[…]}` form and the untagged list/bytes clash.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    List(Vec<MetaValue>),
    Map(BTreeMap<String, MetaValue>),
    #[serde(skip)]
    Bytes(Vec<u8>),
    #[serde(skip)]
    Axes(Axes),
}

/// Reserved builtin meta keys — always present in a [`Meta`], `Null`/empty when unset.
pub const META_SFREQ: &str = "sfreq";
pub const META_UFREQ: &str = "ufreq";
pub const META_INDEX: &str = "index";
pub const META_CHANNELS: &str = "channels";
pub const META_REDUCED: &str = "reduced";
const BUILTIN_KEYS: [&str; 5] = [META_SFREQ, META_UFREQ, META_INDEX, META_CHANNELS, META_REDUCED];

static EMPTY_AXES: Axes = Axes(Vec::new());

/// A `Data`'s metadata sidecar: an insertion-ordered map with the builtin keys always present.
#[derive(Clone, Debug)]
pub struct Meta(IndexMap<String, MetaValue>);

impl Default for Meta {
    fn default() -> Meta {
        Meta::new()
    }
}

impl Meta {
    /// A meta with the builtin keys present but unset (`Null`, empty channels).
    pub fn new() -> Meta {
        let mut m = IndexMap::with_capacity(BUILTIN_KEYS.len());
        m.insert(META_SFREQ.to_string(), MetaValue::Null);
        m.insert(META_UFREQ.to_string(), MetaValue::Null);
        m.insert(META_INDEX.to_string(), MetaValue::Null);
        m.insert(META_CHANNELS.to_string(), MetaValue::Axes(Axes::new()));
        m.insert(META_REDUCED.to_string(), MetaValue::Null);
        Meta(m)
    }
    pub fn empty() -> Meta {
        Meta::new()
    }

    /// The value at `key`, or `None` if absent OR present-but-`Null` (an unset builtin).
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key).filter(|v| !matches!(v, MetaValue::Null))
    }
    pub fn set(&mut self, key: impl Into<String>, v: MetaValue) {
        self.0.insert(key.into(), v);
    }
    /// Iterate ALL entries, `Null` builtins included.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetaValue)> {
        self.0.iter()
    }

    pub fn sfreq(&self) -> Option<f64> {
        as_f64(self.0.get(META_SFREQ))
    }
    pub fn set_sfreq(&mut self, v: Option<f64>) {
        self.set(META_SFREQ, v.map_or(MetaValue::Null, MetaValue::Float));
    }
    pub fn set_ufreq(&mut self, v: Option<f64>) {
        self.set(META_UFREQ, v.map_or(MetaValue::Null, MetaValue::Float));
    }
    pub fn index(&self) -> Option<u64> {
        match self.0.get(META_INDEX) {
            Some(MetaValue::Uint(u)) => Some(*u),
            Some(MetaValue::Int(i)) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }
    pub fn set_index(&mut self, v: Option<u64>) {
        self.set(META_INDEX, v.map_or(MetaValue::Null, MetaValue::Uint));
    }
    pub fn channels(&self) -> &Axes {
        match self.0.get(META_CHANNELS) {
            Some(MetaValue::Axes(a)) => a,
            _ => &EMPTY_AXES,
        }
    }
    pub fn set_channels(&mut self, ch: Axes) {
        self.set(META_CHANNELS, MetaValue::Axes(ch));
    }
    pub fn reduced(&self) -> Option<&MetaValue> {
        self.get(META_REDUCED)
    }
    pub fn set_reduced(&mut self, v: Option<MetaValue>) {
        self.set(META_REDUCED, v.unwrap_or(MetaValue::Null));
    }

    pub fn with_sfreq(mut self, v: Option<f64>) -> Meta {
        self.set_sfreq(v);
        self
    }
    pub fn with_index(mut self, v: Option<u64>) -> Meta {
        self.set_index(v);
        self
    }
    pub fn with_channels(mut self, ch: Axes) -> Meta {
        self.set_channels(ch);
        self
    }
    pub fn with(mut self, key: impl Into<String>, v: MetaValue) -> Meta {
        self.set(key, v);
        self
    }
}

/// Coerce a meta value to f64 — a wire round-trip may deliver an integer for a rate.
fn as_f64(v: Option<&MetaValue>) -> Option<f64> {
    match v {
        Some(MetaValue::Float(f)) => Some(*f),
        Some(MetaValue::Int(i)) => Some(*i as f64),
        Some(MetaValue::Uint(u)) => Some(*u as f64),
        _ => None,
    }
}

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

/// Non-array frames report 0 dims.
impl goofi_view::Reducible for Data {
    fn dtype_tag(&self) -> u8 {
        self.0.value.dtype_tag()
    }
    fn ndim(&self) -> usize {
        match &self.0.value {
            Value::Array(s) => s.shape().len(),
            _ => 0,
        }
    }
    fn shape(&self) -> &[usize] {
        match &self.0.value {
            Value::Array(s) => s.shape(),
            _ => &[],
        }
    }
}

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

    /// A copy with the engine-owned `index` and `ufreq` stamped on — overwritten, never inherited.
    pub fn with_stamps(&self, index: u64, ufreq: Option<f64>) -> Data {
        let mut meta = self.0.meta.clone();
        meta.set_index(Some(index));
        meta.set_ufreq(ufreq);
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

    /// Build an f32 array `Data`, promoting 0-d to 1-d and validating channel coords.
    pub fn array_f32(shape: Vec<usize>, buf: Vec<u8>, meta: Meta) -> Result<Data> {
        // Checked: a garbled shape could otherwise wrap `expect` down onto a short buffer.
        let nelem: usize = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| GoofiError::Invalid("array element count overflows usize".into()))?;
        let expect = nelem
            .checked_mul(4) // f32 itemsize
            .ok_or_else(|| GoofiError::Invalid("array byte length overflows usize".into()))?;
        if buf.len() != expect {
            return Err(GoofiError::Invalid(format!(
                "buffer length {} != nelem {nelem} * 4 = {expect}",
                buf.len(),
            )));
        }

        let shape = if shape.is_empty() { vec![1] } else { shape };

        if meta.channels().0.len() > shape.len() {
            return Err(GoofiError::Invalid(format!(
                "channels has {} axes, exceeds ndim {}",
                meta.channels().0.len(),
                shape.len()
            )));
        }
        for (dim, axis) in meta.channels().0.iter().enumerate() {
            if let Some(coords) = &axis.coords {
                if coords.len() != shape[dim] {
                    return Err(GoofiError::Invalid(format!(
                        "channels dim{dim} has {} coords, expected {}",
                        coords.len(),
                        shape[dim]
                    )));
                }
            }
        }

        Ok(Data::array(ArrayStore::new(shape, Arc::from(buf.into_boxed_slice())), meta))
    }

    pub fn as_array(&self) -> std::result::Result<&ArrayStore, String> {
        match &self.0.value {
            Value::Array(a) => Ok(a),
            Value::Str(_) => Err("expected an array, got a string".into()),
            Value::Table(_) => Err("expected an array, got a table".into()),
        }
    }

    /// State the rank this node needs and get the array back when it holds.
    pub fn assert_ndims(&self) -> Ndims<'_> {
        Ndims(self)
    }
}

/// A pending claim about a frame's rank; it carries the frame so the error can name the shape.
pub struct Ndims<'a>(&'a Data);

impl<'a> Ndims<'a> {
    fn check(
        self,
        holds: impl Fn(usize) -> bool,
        want: &str,
    ) -> std::result::Result<&'a ArrayStore, String> {
        let a = self.0.as_array()?;
        if holds(a.ndim()) {
            Ok(a)
        } else {
            Err(format!("needs {want}, got {:?}", a.shape()))
        }
    }

    pub fn at_least(self, n: usize) -> std::result::Result<&'a ArrayStore, String> {
        self.check(|d| d >= n, &format!("at least {n} dimension(s)"))
    }
}

/// Resolve a signed axis against a rank the way numpy does; out of range is an error, never a clamp.
/// Throughout goofi time is the last dimension and channels the second-to-last.
pub fn resolve_axis(axis: i64, ndim: usize) -> std::result::Result<usize, String> {
    let n = ndim as i64;
    let resolved = if axis < 0 { n + axis } else { axis };
    if resolved < 0 || resolved >= n {
        return Err(format!("axis {axis} is out of range for {ndim} dimension(s)"));
    }
    Ok(resolved as usize)
}

/// The `Data` kind a slot carries.
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
    pub fn from_name(name: &str) -> Option<SlotType> {
        match name {
            "ARRAY" => Some(SlotType::Array),
            "STRING" => Some(SlotType::String),
            "TABLE" => Some(SlotType::Table),
            _ => None,
        }
    }
}

/// A typed parameter descriptor; the `common` scheduling group is a RunPolicy, not a `Param`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
        /// Whether the UI's ⟳ button makes the node re-enumerate `options` via `on_param_refreshed`.
        refresh: bool,
    },
    /// Momentary trigger; the graph clears `fired` by writing the param, not by a read-and-clear.
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
            refresh: false,
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
}
