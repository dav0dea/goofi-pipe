//! The bundles goofi's own repo ships under `node-bundles/`, run as a user gets them: the real
//! `.py` files, installed through the same probe the CLI's scan uses, wired to real producers and
//! read back.
#![cfg(not(feature = "embed"))]

use std::path::{Path, PathBuf};

use goofi_core::Coord;
use goofi_tests::{f32s, hex, install, j, require_python, shape, Goofi};

/// One of the `.py` files a bundle ships, installed through the same seam a user's own file takes.
fn install_bundled(g: &Goofi, py: &str, bundle: &str, file: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../node-bundles").join(bundle).join(file);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the bundled node {}: {e}", path.display()));
    install(g, py, file, &source)
}

/// Wait for a node to come up, reading its error channel WHILE waiting so a node that says why
/// fails with its own words, then for a frame `keep` accepts.
fn first_frame(
    g: &Goofi,
    ty: &str,
    node: goofi_tests::Uid,
    probe: &goofi_tests::OutputProbe,
    mut keep: impl FnMut(&goofi_core::Data) -> bool,
) -> goofi_core::Data {
    g.until(&format!("{ty} to start"), |g| {
        if let Some(e) = g.error(node) {
            panic!("{ty} failed to start: {e}");
        }
        (g.stage(node) == "ready").then_some(())
    });
    g.until(&format!("{ty} to answer once it is ready"), |g| {
        if let Some(e) = g.error(node) {
            panic!("{ty} failed instead of answering: {e}");
        }
        probe.latest().filter(&mut keep)
    })
}

fn labels(d: &goofi_core::Data, dim: &str) -> Vec<String> {
    d.meta()
        .channels()
        .dims()
        .find(|(k, _)| k == dim)
        .map(|(_, coords)| {
            coords
                .iter()
                .map(|c| match c {
                    Coord::Str(s) => s.to_string(),
                    Coord::Num(n) => n.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn the_complexity_bundle_reduces_the_time_axis_and_leaves_the_channels_alone() {
    // A frame that is NOT a vector: against a single channel a flattening node reads as correct.
    let py = require_python();
    let g = Goofi::new();
    let src = g.add("_TestGrid");
    let buf = g.add("Buffer");
    g.set_param(buf, "buffer", "size", 256);
    g.link(src, "out", buf, "data");
    // The window fills BEFORE a node is wired: a growing one answers, and answers differently.
    let window = g.probe(buf, "out");
    g.until("a full window", |_| window.latest().filter(|d| shape(d) == vec![3, 256]));

    // All PROBED AND WIRED AT ONCE, one interpreter per file, as a real scan does — so the
    // ceiling below covers the node boots rather than the probes.
    let (g, py) = (&g, &py.py);
    let nodes: Vec<_> = std::thread::scope(|s| {
        [
            ("lempel_ziv.py", "complexity"),
            ("permutation_entropy.py", "entropy"),
            ("spectral_entropy.py", "entropy"),
            ("detrended_fluctuation.py", "exponent"),
            ("sample_entropy.py", "entropy"),
            ("hjorth.py", "complexity"),
            ("fractal_dimension.py", "dimension"),
            ("zero_crossings.py", "count"),
        ]
        .map(|(file, slot)| {
            s.spawn(move || {
                let ty = install_bundled(g, py, "complexity", file);
                let node = g.add(&ty);
                let probe = g.probe(node, slot);
                g.link(buf, "out", node, "data");
                (ty, node, probe)
            })
        })
        .into_iter()
        .map(|h| h.join().expect("a probe thread panicked"))
        .collect()
    });

    for (ty, node, probe) in nodes {
        let d = first_frame(g, &ty, node, &probe, |d| shape(d) == vec![3]);
        let v = f32s(&d);
        assert!(v.iter().all(|x| x.is_finite()), "{ty} answered {v:?}");
        // The three rows are one signal at three offsets, so answers that DISAGREE mean a mix.
        assert!(
            v.iter().all(|x| (x - v[0]).abs() <= v[0].abs() * 1e-3 + 1e-4),
            "{ty} read the three channels as three different signals: {v:?}",
        );
        assert!(g.error(node).is_none(), "{ty} carries no error: {:?}", g.error(node));
    }
}

#[test]
fn a_complexity_node_reads_a_real_signal_rather_than_answering_a_constant() {
    // An 8 Hz sine over a one-second window has known answers: 15 or 16 zero crossings, a Hjorth
    // complexity of exactly 1, and every entropy solidly inside its range rather than at an edge.
    let py = require_python();
    let g = Goofi::new();
    let osc = g.add("Oscillator");
    let buf = g.add("Buffer");
    g.set_param(osc, "oscillator", "sfreq", 256.0);
    g.set_param(osc, "oscillator", "frequency", 8.0);
    g.set_param(buf, "buffer", "size", 256);
    g.link(osc, "out", buf, "data");
    // Every oracle below is for a FULL window: a growing one holds fewer cycles, or one cut short.
    let window = g.probe(buf, "out");
    g.until("a full window", |_| window.latest().filter(|d| shape(d) == vec![256]));

    let (g, py) = (&g, &py.py);
    let nodes: Vec<_> = std::thread::scope(|s| {
        [
            ("permutation_entropy.py", "entropy", 0.3..0.9),
            ("sample_entropy.py", "entropy", 0.05..0.8),
            ("svd_entropy.py", "entropy", 0.1..0.7),
            ("spectral_entropy.py", "entropy", 0.0..0.5),
            ("lempel_ziv.py", "complexity", 0.0..0.5),
            ("hjorth.py", "complexity", 0.9..1.1),
            ("fractal_dimension.py", "dimension", 1.0..1.1),
            ("zero_crossings.py", "count", 13.0..17.5),
        ]
        .map(|(file, slot, range)| {
            s.spawn(move || {
                let ty = install_bundled(g, py, "complexity", file);
                let node = g.add(&ty);
                let probe = g.probe(node, slot);
                g.link(buf, "out", node, "data");
                (ty, node, probe, range)
            })
        })
        .into_iter()
        .map(|h| h.join().expect("a probe thread panicked"))
        .collect()
    });

    for (ty, node, probe, range) in nodes {
        let d = first_frame(g, &ty, node, &probe, |d| shape(d) == vec![1]);
        let v = f32s(&d)[0];
        assert!(range.contains(&v), "{ty} of an 8 Hz sine is {v}, outside {range:?}");
    }
}

#[test]
fn every_bundle_names_its_packages_and_both_interpreters_hold_them() {
    // Provisioning installs each bundle's `requirements.txt` and startup checks the same files, so
    // a bundle without one, or an interpreter short of one, is what either would silently pass.
    let root = goofi_init::repo_root();
    let bundles = goofi_init::bundle_dirs(&root);
    assert!(!bundles.is_empty(), "the repo ships bundles under node-bundles/");
    for b in &bundles {
        assert!(b.join("requirements.txt").is_file(), "{} names its packages", b.display());
    }
    let reqs = goofi_init::requirements_in(&bundles);
    let gap = std::env::temp_dir().join(format!("goofi-gap-{}.txt", std::process::id()));
    std::fs::write(&gap, "cowsay\n").unwrap();
    for venv in [goofi_init::FT_VENV, goofi_init::GIL_VENV] {
        let py = goofi_init::venv_python(&root.join(venv))
            .unwrap_or_else(|| panic!("no {venv}: {}", goofi_init::RUN_ME));
        let missing = goofi_init::missing_packages(&py, &reqs).expect("uv audits the interpreter");
        assert!(missing.is_empty(), "{venv} lacks {missing:?}: {}", goofi_init::RUN_ME);
        // The check can SEE a gap: naming what is absent takes the index, as the install would.
        let missing = goofi_init::missing_packages(&py, std::slice::from_ref(&gap)).expect("uv resolves the gap");
        assert!(missing.iter().any(|m| m.starts_with("cowsay==")), "{venv}: the gap is named: {missing:?}");
    }
}

/// A python of the tier's own, spawned as the node tier spawns one: the embedding's
/// `PYTHONHOME`/`PYTHONPATH` in this process would point a GIL interpreter at the free-threaded tree.
fn python(py: &str, script: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(py);
    cmd.args(["-c", script]).env_remove("PYTHONPATH").env_remove("PYTHONHOME");
    cmd
}

/// A child killed when the test is done with it, however the test ends.
struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Four channels of white noise under a strong 10 Hz sine, in volts, saved as a FIF by mne itself.
fn write_recording(py: &str, dir: &Path) -> PathBuf {
    let path = dir.join("rec_raw.fif");
    let script = format!(
        r#"
import mne, numpy as np
sf = 128.0; t = np.arange(int(sf * 8)) / sf; rng = np.random.default_rng(0)
x = np.stack([rng.standard_normal(t.size) + 10 * np.sin(2 * np.pi * 10 * t) for _ in range(4)]) * 1e-6
mne.io.RawArray(x, mne.create_info(["Fz", "Cz", "Pz", "Oz"], sf, "eeg"), verbose=False).save({path:?}, overwrite=True, verbose=False)
"#,
        path = path.to_string_lossy()
    );
    let out = python(py, &script).output().expect("spawn python");
    assert!(out.status.success(), "mne could not write the recording: {}", String::from_utf8_lossy(&out.stderr));
    path
}

const OUTLET: &str = r#"
import pylsl, time
info = pylsl.StreamInfo("goofi-test", "EEG", 3, 100, "float32", "goofi-test-src")
chs = info.desc().append_child("channels")
for name in ["A", "B", "C"]:
    chs.append_child("channel").append_child_value("label", name)
out = pylsl.StreamOutlet(info)
k = 0
while True:
    out.push_chunk([[k + j, 2.0 * (k + j), 3.0 * (k + j)] for j in range(10)])
    k += 10
    time.sleep(0.1)
"#;

#[test]
fn the_eeg_bundle_plays_a_recording_reads_its_spectrum_and_receives_a_live_stream() {
    let py = require_python();
    let g = Goofi::new();
    let dir = std::env::temp_dir().join(format!("goofi-eeg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let recording = write_recording(&py.py, &dir);

    // The playback: nothing until a file is named, then the recording's own channels and rate.
    let play_ty = install_bundled(&g, &py.py, "eeg", "eeg_playback.py");
    let play = g.add(&play_ty);
    let played = g.probe(play, "out");
    g.ready(play);
    assert!(g.stays(|_| played.count() == 0), "no file, no frames");
    g.set_param(play, "playback", "file", recording.to_string_lossy().to_string());
    let d = first_frame(&g, &play_ty, play, &played, |_| true);
    assert_eq!(shape(&d)[0], 4, "one row per channel: {:?}", shape(&d));
    assert_eq!(d.meta().sfreq(), Some(128.0), "the recording's rate rides the frame");
    assert_eq!(labels(&d, "dim0"), ["Fz", "Cz", "Pz", "Oz"], "and so do its channel names");

    // Its spectrum, through the shipped Buffer and Psd, into the two spectral nodes.
    let buf = g.add("Buffer");
    g.set_param(buf, "buffer", "size", 256);
    let window = g.probe(buf, "out");
    let psd = g.add("Psd");
    let bands_ty = install_bundled(&g, &py.py, "eeg", "eeg_power_bands.py");
    let bands = g.add(&bands_ty);
    let power = g.probe(bands, "power");
    let fooof_ty = install_bundled(&g, &py.py, "eeg", "fooof.py");
    let fooof = g.add(&fooof_ty);
    let peaks = g.probe(fooof, "peaks");
    let aperiodic = g.probe(fooof, "aperiodic");
    g.link(play, "out", buf, "data");
    g.link(buf, "out", psd, "data");
    g.link(psd, "psd", bands, "psd");
    g.link(psd, "psd", fooof, "psd");

    let d = g.until("a full window", |_| window.latest().filter(|d| shape(d) == vec![4, 256]));
    let peak = f32s(&d).iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!((8.0..40.0).contains(&peak), "volts scaled to microvolts: the sine peaks near 10, not 1e-5 ({peak})");

    // A full window is two seconds of playback; until then the 10 Hz line is smeared.
    let d = first_frame(&g, &bands_ty, bands, &power, |d| shape(d) == vec![4, 5] && f32s(d)[2] > 0.0);
    assert_eq!(labels(&d, "dim1"), ["delta", "theta", "alpha", "beta", "gamma"]);
    assert_eq!(labels(&d, "dim0"), ["Fz", "Cz", "Pz", "Oz"], "the channel axis survives the reduction");
    let alpha_wins = |d: &goofi_core::Data| {
        f32s(d).chunks(5).all(|row| row.iter().all(|b| *b <= row[2]))
    };
    g.until("the alpha band to carry the 10 Hz sine", |_| power.latest().filter(alpha_wins));
    g.set_param(bands, "bands", "relative", true);
    g.until("relative power to answer a share", |_| {
        power.latest().filter(|d| f32s(d).chunks(5).all(|row| row[2] > 0.5 && row[2] <= 1.0))
    });
    g.set_param(bands, "bands", "gamma", "");
    g.until("a dropped band to leave the axis", |_| {
        power.latest().filter(|d| shape(d) == vec![4, 4] && labels(d, "dim1") == ["delta", "theta", "alpha", "beta"])
    });

    let d = first_frame(&g, &fooof_ty, fooof, &peaks, |d| shape(d) == vec![4, 6, 3]);
    assert_eq!(labels(&d, "dim2"), ["cf", "pw", "bw"]);
    g.until("every channel's strongest peak to sit at 10 Hz", |_| {
        peaks.latest().filter(|d| f32s(d).chunks(18).all(|row| (8.5..11.5).contains(&row[0])))
    });
    let d = g.until("the aperiodic fit", |_| aperiodic.latest().filter(|d| shape(d) == vec![4, 2]));
    assert_eq!(labels(&d, "dim1"), ["offset", "exponent"]);
    // One 256-sample periodogram is a noisy thing to fit a slope to; the bound only has to
    // separate flat from the 2 a random walk would answer.
    assert!(f32s(&d).chunks(2).all(|row| row[1].abs() < 1.5), "white noise is flat: {:?}", f32s(&d));
    g.set_param(fooof, "fooof", "mode", "knee");
    let d = g.until("the knee mode to widen the fit", |_| aperiodic.latest().filter(|d| shape(d) == vec![4, 3]));
    assert_eq!(labels(&d, "dim1"), ["offset", "knee", "exponent"]);

    // The end of the recording, once looping is off, is silence — and looping back on resumes.
    g.set_param(play, "playback", "loop", false);
    g.until("the playback to stop at the end of the recording", |g| {
        let n = played.count();
        g.stays(|_| played.count() == n).then_some(())
    });
    g.set_param(play, "playback", "loop", true);
    let n = played.count();
    g.until("the playback to resume", |_| (played.count() > n).then_some(()));

    // A live stream: the node is present and silent until the stream exists, then it is wired.
    let lsl_ty = install_bundled(&g, &py.py, "eeg", "lsl_in.py");
    let lsl = g.add(&lsl_ty);
    let received = g.probe(lsl, "out");
    g.set_param(lsl, "lsl", "name", "goofi-test");
    g.ready(lsl);
    assert!(g.stays(|_| received.count() == 0), "no stream, no frames, no error");
    assert!(g.error(lsl).is_none(), "an absent stream is not an error: {:?}", g.error(lsl));
    let _outlet = Child(python(&py.py, OUTLET).stdout(std::process::Stdio::null()).spawn().expect("spawn the outlet"));
    let d = first_frame(&g, &lsl_ty, lsl, &received, |d| shape(d)[0] == 3 && shape(d)[1] > 0);
    assert_eq!(d.meta().sfreq(), Some(100.0), "the stream's nominal rate rides the frame");
    assert_eq!(labels(&d, "dim0"), ["A", "B", "C"], "and so do its channel labels");
    let v = f32s(&d);
    let cols = shape(&d)[1];
    assert!(
        (0..cols).all(|t| v[cols + t] == 2.0 * v[t] && v[2 * cols + t] == 3.0 * v[t]),
        "samples land channel-major, not transposed: {v:?}"
    );

    // The ⟳ on the stream picker lists what is on the network.
    let mut ev = g.events();
    g.call("node param refresh", j!({ "node": hex(lsl), "param": "lsl/name" }));
    let p = g.until("the picker's echo", |_| {
        let p = ev.next("state_update");
        (p["node"] == hex(lsl) && p["refreshed_params"] == j!([["lsl", "name"]])).then_some(p)
    });
    let options = p["params"]["lsl"]["name"]["options"].as_array().cloned().unwrap_or_default();
    assert!(options.contains(&j!("goofi-test")), "the live stream is offered: {options:?}");
    assert!(options.contains(&j!("")), "…and `any` stays an option: {options:?}");
}
