//! `Data`, `Meta`, `Param`, the reduction kernels, the globals table and path spelling — the
//! vocabulary every other crate is written in.

use goofi_core::*;


#[test]
fn viewspec_admits_a_real_data_frame() {
    use goofi_view::{DimCmp, DimConstraint, ViewDtype, ViewSpec};
    // A 2-D (3, 1000) f32 array frame.
    let d = Data::array_f32(vec![3, 1000], vec![0u8; 3 * 1000 * 4], Meta::empty()).unwrap();
    assert_eq!(goofi_view::Reducible::dtype_tag(&d), 0, "array tag");
    assert_eq!(goofi_view::Reducible::shape(&d), &[3, 1000]);
    // A line viewer (array, <=2d) admits it.
    let line = ViewSpec { dtype: ViewDtype::Array, ndim: vec![(DimCmp::Le, 2)], dims: vec![], reduce: vec![] };
    assert!(line.admits(&d));
    // A topomap viewer (array, ndim == 1) rejects a 2-D frame.
    let topo = ViewSpec { dtype: ViewDtype::Array, ndim: vec![(DimCmp::Eq, 1)], dims: vec![], reduce: vec![] };
    assert!(!topo.admits(&d));
    // An image viewer needing the last dim to be a channel count (<=4) rejects 1000.
    let image = ViewSpec {
        dtype: ViewDtype::Array,
        ndim: vec![],
        dims: vec![DimConstraint { dim: -1, cmp: DimCmp::Le, n: 4 }],
        reduce: vec![],
    };
    assert!(!image.admits(&d));
}

#[test]
fn meta_map_builtins_accessors_and_builders() {
    // Builtins are guaranteed present at construction (as unset).
    let m = Meta::new();
    assert_eq!(m.sfreq(), None);
    assert_eq!(m.index(), None);
    assert!(m.channels().is_empty());
    assert!(m.get("sfreq").is_none(), "an unset builtin reads as absent");
    let keys: Vec<String> = m.iter().map(|(k, _)| k.clone()).collect();
    for b in [META_SFREQ, META_UFREQ, META_INDEX, META_CHANNELS, META_REDUCED] {
        assert!(keys.iter().any(|k| k == b), "builtin `{b}` always present in the map");
    }
    // Typed builders + generic set round-trip.
    let m = Meta::new()
        .with_sfreq(Some(250.0))
        .with_index(Some(7))
        .with("label", MetaValue::Str("eeg".into()));
    assert_eq!(m.sfreq(), Some(250.0));
    assert_eq!(m.index(), Some(7));
    assert_eq!(m.get("label"), Some(&MetaValue::Str("eeg".into())));
    // channels lives in the map as a MetaValue::Axes (keeps its typed API).
    let m = Meta::new().with_channels(Axes::new().with(0, Axis::coords(vec![Coord::Num(1.0)])));
    assert!(matches!(m.get("channels"), Some(MetaValue::Axes(_))));
    assert_eq!(m.channels().0.len(), 1);
    // Lenient coercion: an index delivered as msgpack Int still reads as u64.
    let mut m = Meta::new();
    m.set(META_INDEX, MetaValue::Int(42));
    assert_eq!(m.index(), Some(42));
}

#[test]
fn warn_cast_once_dedups_per_dtype() {
    use std::collections::HashSet;
    let mut w: HashSet<SrcDtype> = HashSet::new();
    assert!(warn_cast_once(&mut w, "out", SrcDtype::F64), "first f64 warns");
    assert!(!warn_cast_once(&mut w, "out", SrcDtype::F64), "second f64 suppressed");
    assert!(warn_cast_once(&mut w, "out", SrcDtype::I16), "a different dtype still warns");
    assert!(!warn_cast_once(&mut w, "out", SrcDtype::F32), "f32 never warns");
}

#[test]
fn cast_to_f32_converts_each_source_dtype() {
    fn as_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    // f32 passes through unchanged; did_cast is false.
    let f: Vec<u8> = [1.5f32, -2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (out, did) = cast_to_f32(SrcDtype::F32, &f).unwrap();
    assert_eq!(out, f);
    assert!(!did, "f32 in -> no cast");
    // f64 -> f32.
    let d: Vec<u8> = [1.5f64, -2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (out, did) = cast_to_f32(SrcDtype::F64, &d).unwrap();
    assert_eq!(as_f32(&out), vec![1.5, -2.0]);
    assert!(did);
    // signed int -> f32.
    let i: Vec<u8> = [3i16, -4].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(as_f32(&cast_to_f32(SrcDtype::I16, &i).unwrap().0), vec![3.0, -4.0]);
    // unsigned byte + bool -> f32.
    assert_eq!(as_f32(&cast_to_f32(SrcDtype::U8, &[10u8, 20]).unwrap().0), vec![10.0, 20.0]);
    assert_eq!(as_f32(&cast_to_f32(SrcDtype::Bool, &[1u8, 0]).unwrap().0), vec![1.0, 0.0]);
    // f16 -> f32 (0x3C00 = 1.0, 0x4000 = 2.0, little-endian).
    assert_eq!(as_f32(&cast_to_f32(SrcDtype::F16, &[0x00, 0x3C, 0x00, 0x40]).unwrap().0), vec![1.0, 2.0]);
    // a buffer length not a multiple of itemsize is an error, never a silent misread.
    assert!(cast_to_f32(SrcDtype::F32, &[0u8; 3]).is_err());
    assert!(cast_to_f32(SrcDtype::F64, &[0u8; 4]).is_err());
}

#[test]
fn src_dtype_parses_numpy_typestrings() {
    assert_eq!(SrcDtype::from_numpy_typestr("<f4"), Some(SrcDtype::F32));
    assert_eq!(SrcDtype::from_numpy_typestr("|u1"), Some(SrcDtype::U8));
    assert_eq!(SrcDtype::from_numpy_typestr("<f8"), Some(SrcDtype::F64));
    assert_eq!(SrcDtype::from_numpy_typestr(">f4"), None, "big-endian rejected");
    assert_eq!(SrcDtype::from_numpy_typestr("<x9"), None);
}

#[test]
fn zero_d_promotes_to_1d() {
    let d = Data::array_f32(vec![], 3.0f32.to_le_bytes().to_vec(), Meta::empty())
        .unwrap();
    let Value::Array(s) = d.value() else { panic!() };
    assert_eq!(s.shape(), &[1]);
}

#[test]
fn wrong_buffer_length_errors() {
    assert!(Data::array_f32(vec![3], vec![0u8; 8], Meta::empty()).is_err());
}

#[test]
fn channel_length_must_match_shape() {
    // dim0 labeled with 1 coord but shape[0] == 2 -> reject.
    let meta = Meta::new().with_channels(Axes::new().with(0, Axis::coords(vec![Coord::Str("a".into())])));
    let buf: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(); // shape[0]=2
    assert!(Data::array_f32(vec![2], buf, meta).is_err());
}

#[test]
fn too_many_axes_for_ndim_is_rejected() {
    // Two labeled axes on a 1-D array -> reject (axes.len() > ndim).
    let meta =
        Meta::new().with_channels(Axes(vec![Axis::default(), Axis::coords(vec![Coord::Num(1.0)])]));
    let buf: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    assert!(Data::array_f32(vec![1], buf, meta).is_err());
}

#[test]
fn axes_with_pads_empty_leading_dims() {
    // Labeling only dim1 yields [empty, {coords}] — the "null entry" for dim0.
    let axes = Axes::new().with(1, Axis::coords(vec![Coord::Num(10.0), Coord::Num(20.0)]));
    assert_eq!(axes.0.len(), 2);
    assert!(axes.get(0).unwrap().is_empty());
    assert!(axes.get(1).unwrap().coords.is_some());
}

#[test]
fn axes_sliced_subsets_coords() {
    let axes = Axes(vec![Axis::coords(vec![Coord::Num(0.0), Coord::Num(1.0), Coord::Num(2.0)])]);
    let s = axes.sliced(0, &[2, 0]);
    let c = s.get(0).unwrap().coords.as_ref().unwrap();
    assert_eq!(c.as_ref(), &[Coord::Num(2.0), Coord::Num(0.0)]);
}

#[test]
fn param_accessors_read_each_variant() {
    assert_eq!(Param::float(2.5, 0.0, 10.0).as_f64(), Some(2.5));
    assert_eq!(Param::int(4, 0, 10).as_i64(), Some(4));
    assert_eq!(Param::boolean(true).as_bool(), Some(true));
    assert_eq!(Param::str_free("hi").as_str(), Some("hi"));
    assert_eq!(Param::Trigger { fired: true }.as_bool(), Some(true));
}

#[test]
fn slot_type_names() {
    assert_eq!(SlotType::Array.name(), "ARRAY");
    assert_eq!(SlotType::String.name(), "STRING");
    assert_eq!(SlotType::Table.name(), "TABLE");
}

#[test]
fn slot_type_from_name_roundtrips() {
    assert_eq!(SlotType::from_name("ARRAY"), Some(SlotType::Array));
    assert_eq!(SlotType::from_name("STRING"), Some(SlotType::String));
    assert_eq!(SlotType::from_name("TABLE"), Some(SlotType::Table));
    assert_eq!(SlotType::from_name("nope"), None);
}

#[test]
fn with_stamps_sets_engine_meta_and_shares_buffer() {
    let d = Data::array_f32(vec![2], vec![0u8; 8], Meta::empty()).unwrap();
    assert_eq!(d.meta().index(), None);
    assert_eq!(d.meta().ufreq(), None);
    let stamped = d.with_stamps(7, Some(50.0));
    assert_eq!(stamped.meta().index(), Some(7));
    assert_eq!(stamped.meta().ufreq(), Some(50.0));
    assert_eq!(d.meta().index(), None, "original is untouched (immutable)");
    assert_eq!(d.meta().ufreq(), None, "original is untouched (immutable)");
    // The value buffer is shared, not copied (Arc bump).
    if let (Value::Array(a), Value::Array(b)) = (d.value(), stamped.value()) {
        assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
    } else {
        panic!()
    }
}

#[test]
fn with_stamps_none_ufreq_leaves_field_none() {
    let d = Data::array_f32(vec![2], vec![0u8; 8], Meta::empty()).unwrap();
    let stamped = d.with_stamps(3, None);
    assert_eq!(stamped.meta().index(), Some(3));
    assert_eq!(stamped.meta().ufreq(), None, "no measurement yet ⇒ no ufreq stamped");
}

#[test]
fn data_fan_out_is_arc_clone() {
    let d = Data::array_f32(vec![2], vec![0u8; 8], Meta::empty()).unwrap();
    let d2 = d.clone();
    // Both refer to the same underlying buffer (zero-copy fan-out).
    if let (Value::Array(a), Value::Array(b)) = (d.value(), d2.value()) {
        assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
    } else {
        panic!()
    }
}

// ---------------------------------------------------------------------------
// Reduction kernels
// ---------------------------------------------------------------------------

use goofi_core::reduce::*;
// The kernels take the method the ViewSpec algebra names.
use goofi_view::ReduceMethod;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn subsample_idx_is_evenly_spaced_with_endpoints() {
    assert_eq!(subsample_idx(10, 3), vec![0, 5, 9]);
    assert_eq!(subsample_idx(5, 5), vec![0, 1, 2, 3, 4]);
    assert_eq!(subsample_idx(5, 9), vec![0, 1, 2, 3, 4], "m>=n returns all");
    assert_eq!(subsample_idx(100, 1), vec![0]);
}

#[test]
fn subsample_axis_gathers_rows() {
    // (3 channels, 2 samples) f32 — subsample the channel axis to 2 → rows 0 and 2.
    let bytes = f32_bytes(&[10.0, 11.0, /*c0*/ 20.0, 21.0, /*c1*/ 30.0, 31.0 /*c2*/]);
    let r = reduce_axis(&bytes, &[3, 2], 0, 2, ReduceMethod::Subsample).unwrap();
    assert_eq!(r.new_len, 2);
    assert_eq!(r.centers, vec![0, 2]);
    assert_eq!(as_f32(&r.bytes), vec![10.0, 11.0, 30.0, 31.0], "kept channels 0 and 2, both samples each");
}

#[test]
fn envelope_axis_emits_min_max_per_bin() {
    // 1-D f32 of 8 samples → W=2 bins → 4 outputs: [min,max] of [1,4,2,3] and [8,5,7,6].
    let d = f32_bytes(&[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0]);
    let r = reduce_axis(&d, &[8], 0, 2, ReduceMethod::Envelope).unwrap();
    assert_eq!(r.new_len, 4);
    assert_eq!(as_f32(&r.bytes), vec![1.0, 4.0, 5.0, 8.0], "min,max per bin");
    assert_eq!(r.centers.len(), 4, "one center per output entry");
}

#[test]
fn envelope_skips_when_it_would_not_shrink_twofold() {
    // 6 samples, max 4 → W=4, 2W=8 > 6 → no reduction.
    let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(reduce_axis(&d, &[6], 0, 4, ReduceMethod::Envelope).is_none());
}

#[test]
fn envelope_per_channel_on_2d() {
    // (2 channels, 4 samples), W=2 → each channel → [min,max]×2 = (2,4).
    let d = f32_bytes(&[0.0, 2.0, 1.0, 3.0, /*c0*/ 9.0, 5.0, 8.0, 6.0 /*c1*/]);
    let r = reduce_axis(&d, &[2, 4], 1, 2, ReduceMethod::Envelope).unwrap();
    assert_eq!(r.new_len, 4);
    // c0: [min(0,2),max(0,2), min(1,3),max(1,3)] = [0,2,1,3]; c1: [5,9,6,8]
    assert_eq!(as_f32(&r.bytes), vec![0.0, 2.0, 1.0, 3.0, 5.0, 9.0, 6.0, 8.0]);
}

#[test]
fn envelope_all_nan_bin_stays_nan() {
    // An all-NaN bin must not fabricate the ±INF seeds: a viewer draws NaN as a gap,
    // but a single +INF destroys autoscale for every channel in the frame.
    let d = f32_bytes(&[f32::NAN; 8]);
    let r = reduce_axis(&d, &[8], 0, 2, ReduceMethod::Envelope).unwrap();
    assert_eq!(r.new_len, 4);
    assert!(as_f32(&r.bytes).iter().all(|v| v.is_nan()), "all-NaN bin reduces to NaN, got {:?}", as_f32(&r.bytes));
}

#[test]
fn envelope_skips_nan_when_the_bin_has_any_finite_value() {
    // [NaN,5,NaN,3] and [NaN,NaN,2,NaN] → the finite values still win their bin.
    let d = f32_bytes(&[f32::NAN, 5.0, f32::NAN, 3.0, f32::NAN, f32::NAN, 2.0, f32::NAN]);
    let r = reduce_axis(&d, &[8], 0, 2, ReduceMethod::Envelope).unwrap();
    assert_eq!(as_f32(&r.bytes), vec![3.0, 5.0, 2.0, 2.0], "NaN skipped where a finite value exists");
}

#[test]
fn area_axis_is_block_mean() {
    // 1-D f32 of 6 → 3 bins of 2 → means [1.5, 3.5, 5.5].
    let d = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = reduce_axis(&d, &[6], 0, 3, ReduceMethod::Area).unwrap();
    assert_eq!(r.new_len, 3);
    assert_eq!(as_f32(&r.bytes), vec![1.5, 3.5, 5.5]);
}

#[test]
fn area_2d_block_mean_over_the_inner_axis() {
    // (2, 4) mean the last axis to 2 → per channel means of pairs.
    let d = f32_bytes(&[1.0, 3.0, 5.0, 7.0, /*c0*/ 0.0, 0.0, 10.0, 10.0 /*c1*/]);
    let r = reduce_axis(&d, &[2, 4], 1, 2, ReduceMethod::Area).unwrap();
    assert_eq!(as_f32(&r.bytes), vec![2.0, 6.0, 0.0, 10.0]);
}

#[test]
fn no_reduction_when_already_small() {
    let d = f32_bytes(&[1.0, 2.0, 3.0]);
    assert!(reduce_axis(&d, &[3], 0, 10, ReduceMethod::Subsample).is_none());
    assert!(reduce_axis(&d, &[3], 0, 10, ReduceMethod::Area).is_none());
}

// --- reduce_for_view (Data-level composition) ---
use crate::{Axes, Axis, Coord, Meta};
use goofi_view::{MergedViewSpec, PlannedAxis};

fn f32_frame(shape: Vec<usize>, vals: &[f32], meta: Meta) -> Data {
    Data::array_f32(shape, f32_bytes(vals), meta).unwrap()
}

#[test]
fn reduce_for_view_applies_plan_and_records_meta() {
    // 8-sample waveform, envelope to W=2 → 4 body samples; meta.reduced["0"] recorded;
    // source timeline (index/ufreq/sfreq) carried verbatim.
    let meta = Meta::new().with_sfreq(Some(250.0)).with_ufreq(Some(30.0)).with_index(Some(7));
    let f = f32_frame(vec![8], &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0], meta);
    let plan = MergedViewSpec { axes: vec![PlannedAxis { dim: 0, max: 2, method: ReduceMethod::Envelope }] };
    let r = reduce_for_view(&f, &plan);
    let Value::Array(s) = r.value() else { panic!() };
    assert_eq!(s.shape(), &[4], "envelope 2W=4");
    assert_eq!(as_f32(s.as_bytes()), vec![1.0, 4.0, 5.0, 8.0]);
    // Source timeline verbatim.
    assert_eq!(r.meta().sfreq(), Some(250.0));
    assert_eq!(r.meta().ufreq(), Some(30.0));
    assert_eq!(r.meta().index(), Some(7), "index rides through untouched");
    // reduced meta records the original length + method.
    let Some(MetaValue::Map(m)) = &r.meta().reduced() else { panic!("reduced meta set") };
    let Some(MetaValue::Map(e)) = m.get("0") else { panic!("dim 0 recorded") };
    assert_eq!(e.get("orig_len"), Some(&MetaValue::Uint(8)));
    assert_eq!(e.get("method"), Some(&MetaValue::Str("envelope".into())));
}

#[test]
fn reduce_for_view_coreduces_channel_coords() {
    // (3 channels, 2 samples), subsample the channel axis to 2 → coords ["a","c"].
    let ch = Axes::new().with(
        0,
        Axis::coords(vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]),
    );
    let meta = Meta::new().with_channels(ch);
    let f = f32_frame(vec![3, 2], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], meta);
    let plan = MergedViewSpec { axes: vec![PlannedAxis { dim: 0, max: 2, method: ReduceMethod::Subsample }] };
    let r = reduce_for_view(&f, &plan);
    let Value::Array(s) = r.value() else { panic!() };
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(as_f32(s.as_bytes()), vec![10.0, 11.0, 30.0, 31.0], "kept channels 0 and 2");
    let coords = r.meta().channels().get(0).and_then(|a| a.coords.clone()).expect("coords co-reduced");
    assert_eq!(coords.as_ref(), &[Coord::Str("a".into()), Coord::Str("c".into())]);
}

#[test]
fn reduce_for_view_carries_verbatim_coords_for_small_subsample_axes() {
    // G5: a subsampled axis with ≤4096 original entries carries its ORIGINAL coord labels
    // verbatim in meta.reduced, so the inspector reconstructs exact labels (not
    // approximations). Only for subsample (channels/trajectory), only when small.
    let ch = Axes::new().with(
        0,
        Axis::coords(vec![Coord::Str("a".into()), Coord::Str("b".into()), Coord::Str("c".into())]),
    );
    let meta = Meta::new().with_channels(ch);
    let f = f32_frame(vec![3, 2], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], meta);
    let plan = MergedViewSpec { axes: vec![PlannedAxis { dim: 0, max: 2, method: ReduceMethod::Subsample }] };
    let r = reduce_for_view(&f, &plan);
    let Some(MetaValue::Map(reduced)) = r.meta().reduced().as_ref() else { panic!("reduced meta") };
    let MetaValue::Map(entry) = reduced.get("0").expect("dim 0 reduced") else { panic!("map") };
    assert_eq!(
        entry.get("orig_coord"),
        Some(&MetaValue::List(vec![
            MetaValue::Str("a".into()),
            MetaValue::Str("b".into()),
            MetaValue::Str("c".into()),
        ])),
        "the three original channel labels are carried verbatim"
    );
}

#[test]
fn reduce_for_view_omits_verbatim_coords_for_non_subsample_or_large_axes() {
    // Envelope (not subsample) → no verbatim coords, even when small.
    let ch = Axes::new().with(0, Axis::coords((0..8).map(|i| Coord::Num(i as f64)).collect::<Vec<_>>()));
    let meta = Meta::new().with_channels(ch);
    let f = f32_frame(vec![8], &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0], meta);
    let plan = MergedViewSpec { axes: vec![PlannedAxis { dim: 0, max: 2, method: ReduceMethod::Envelope }] };
    let r = reduce_for_view(&f, &plan);
    let Some(MetaValue::Map(reduced)) = r.meta().reduced().as_ref() else { panic!("reduced meta") };
    let MetaValue::Map(entry) = reduced.get("0").unwrap() else { panic!("map") };
    assert!(entry.get("orig_coord").is_none(), "envelope axis carries no verbatim coords");
}

#[test]
fn reduce_for_view_passthrough_string_and_empty_plan() {
    let s = Data::string("hello", Meta::empty());
    let plan = MergedViewSpec { axes: vec![PlannedAxis { dim: 0, max: 1, method: ReduceMethod::Subsample }] };
    assert!(matches!(reduce_for_view(&s, &plan).value(), Value::Str(v) if v.as_ref() == "hello"));
    // Empty plan → unchanged array.
    let f = f32_frame(vec![4], &[1.0, 2.0, 3.0, 4.0], Meta::empty());
    let r = reduce_for_view(&f, &MergedViewSpec::default());
    let Value::Array(s2) = r.value() else { panic!() };
    assert_eq!(s2.shape(), &[4]);
    assert!(r.meta().reduced().is_none(), "no reduction, no reduced meta");
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

use goofi_core::globals::*;

#[test]
fn value_coercions() {
    assert_eq!(GlobalValue::Int(3).as_f64(), Some(3.0));
    assert_eq!(GlobalValue::Float(2.7).as_i64(), Some(3)); // rounds
    assert_eq!(GlobalValue::Bool(true).as_f64(), Some(1.0));
    assert_eq!(GlobalValue::Str("x".into()).as_f64(), None);
    assert_eq!(GlobalValue::Float(1.0).type_tag(), "float");
    assert_eq!(GlobalValue::Str("x".into()).type_tag(), "string");
}

#[test]
fn name_validation() {
    assert!(is_valid_global_name("default_ufreq"));
    assert!(is_valid_global_name("_x1"));
    assert!(!is_valid_global_name("")); // empty
    assert!(!is_valid_global_name("1x")); // leading digit
    assert!(!is_valid_global_name("a b")); // space
    assert!(!is_valid_global_name("a.b")); // dot
    assert!(!is_valid_global_name("globals")); // reserved namespace
}

#[test]
fn seeds_system_globals_on_new() {
    let s = GlobalStore::new();
    assert_eq!(s.get("default_ufreq"), Some(&GlobalValue::Float(30.0)));
    assert!(s.is_system("default_ufreq"));
    assert_eq!(s.entries().count(), SYSTEM_GLOBALS.len());
}

#[test]
fn set_keeps_the_declared_type() {
    let mut s = GlobalStore::new();
    // Editing default_ufreq (Float) with an Int coerces back to Float — the type stays stable.
    s.set("default_ufreq", GlobalValue::Int(60)).unwrap();
    assert_eq!(s.get("default_ufreq"), Some(&GlobalValue::Float(60.0)));
    // Setting a non-existent global errors (use add).
    assert!(s.set("nope", GlobalValue::Float(1.0)).is_err());
}

#[test]
fn add_edit_remove_user_globals() {
    let mut s = GlobalStore::new();
    s.add("subject_id", GlobalValue::Str("P07".into())).unwrap();
    assert!(!s.is_system("subject_id"));
    assert_eq!(s.get("subject_id"), Some(&GlobalValue::Str("P07".into())));
    // Duplicate + invalid-name adds are rejected.
    assert!(s.add("subject_id", GlobalValue::Str("x".into())).is_err());
    assert!(s.add("1bad", GlobalValue::Float(0.0)).is_err());
    // Edit a user global.
    s.set("subject_id", GlobalValue::Str("P08".into())).unwrap();
    assert_eq!(s.get("subject_id"), Some(&GlobalValue::Str("P08".into())));
    // Remove a user global.
    s.remove("subject_id").unwrap();
    assert!(!s.contains("subject_id"));
}

#[test]
fn a_system_global_cannot_be_deleted_and_is_reasserted_after_a_load() {
    let mut s = GlobalStore::new();
    assert!(s.remove("default_ufreq").is_err(), "a system global cannot be deleted");
    // The back-fill's real caller is a LOAD: a patch saved before a system global existed brings a
    // table without it, and `reassert_system` is what puts it back at its default. Asserted through
    // the public pair rather than by reaching in and removing the entry.
    s.set("default_ufreq", GlobalValue::Float(7.0)).unwrap();
    s.reassert_system();
    assert_eq!(s.get("default_ufreq"), Some(&GlobalValue::Float(7.0)),
               "a value the patch carries is not clobbered by the back-fill");
    assert!(s.is_system("default_ufreq"), "…and it is still flagged as the system's");
}

#[test]
fn apply_change_routes_add_edit_delete() {
    let mut s = GlobalStore::new();
    s.apply_change("k", Some(GlobalValue::Int(5))).unwrap(); // add
    assert_eq!(s.get("k"), Some(&GlobalValue::Int(5)));
    s.apply_change("k", Some(GlobalValue::Int(7))).unwrap(); // edit
    assert_eq!(s.get("k"), Some(&GlobalValue::Int(7)));
    s.apply_change("k", None).unwrap(); // delete
    assert!(!s.contains("k"));
    assert!(s.apply_change("default_ufreq", None).is_err(), "system delete rejected");
}

#[test]
fn snapshot_reads() {
    let mut s = GlobalStore::new();
    s.add("g", GlobalValue::Float(1.5)).unwrap();
    let snap = s.snapshot();
    assert_eq!(snap.f64("default_ufreq"), Some(30.0));
    assert_eq!(snap.f64("g"), Some(1.5));
    assert_eq!(snap.f64("absent"), None);
}

// ---------------------------------------------------------------------------
// Path spelling
// ---------------------------------------------------------------------------

use std::path::Path;

use goofi_core::path::*;

/// A path built the platform's own way comes back in goofi's spelling. On Windows that means
/// the separators are gone; on unix it was already spelled this way and must survive untouched.
#[test]
fn a_path_is_spelled_with_slashes_whatever_the_platform_builds() {
    let p = Path::new("one").join("two").join("three.txt");
    assert_eq!(to_slash(&p), "one/two/three.txt");
}

/// …and a name that merely CONTAINS the other platform's separator is not mangled by it. On
/// unix `a\b` is one legal filename and must stay one; on Windows it cannot occur at all, so
/// the same assertion reads as "two components", which is equally what the platform means.
#[test]
fn a_backslash_is_a_separator_only_where_the_platform_says_so() {
    let one = to_slash(Path::new("plain.txt"));
    assert_eq!(one, "plain.txt");
    let joined = Path::new("dir").join("a\\b");
    // Whatever this platform thinks `a\b` is, `to_slash` must agree with it rather than impose
    // an answer: the string round-trips back to the same `Path` either way.
    assert_eq!(Path::new(&to_slash(&joined)).components().count(), joined.components().count());
}

/// Every goofi path is born through [`canonical`], so the extended-length prefix has to die
/// here — downstream is too late, because a `\\?\` path cannot hold the forward slashes the
/// rest of goofi speaks in.
#[test]
fn a_canonical_path_carries_no_verbatim_prefix() {
    let real = canonical(&std::env::temp_dir()).expect("the temp dir canonicalizes");
    let spelled = to_slash(&real);
    assert!(!spelled.starts_with("//?/"), "verbatim prefix survived: {spelled}");
    assert!(!spelled.contains('\\'), "a separator survived: {spelled}");
}
