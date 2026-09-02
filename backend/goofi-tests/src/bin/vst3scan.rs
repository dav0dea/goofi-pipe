//! The harness's `goofi`: the one door the audio engine spawns, `vst3-scan`, as the real binary
//! answers it.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("vst3-scan") => goofi_audio::vst3::scan_main(&args[1..]),
        _ => 2,
    };
    std::process::exit(code);
}
