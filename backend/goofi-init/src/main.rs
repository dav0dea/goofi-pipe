//! `cargo run -p goofi-init` — the one setup step a fresh clone needs.
//!
//! See the crate docs for why this is a binary rather than a build script, and Rust rather than a
//! shell script. Idempotent: run it again after a `goofi-pymod` version bump, or any time you are
//! not sure.

fn main() {
    // The repo root as known at COMPILE time, so the command works from any working directory —
    // `cargo run -p goofi-init` is run from wherever the developer happens to be standing.
    let root = goofi_init::repo_root();
    println!("provisioning goofi's Python interpreters…");
    match goofi_init::init(&root) {
        Ok(()) => println!(
            "ready — `cargo build`, `cargo test` and `cargo run` now work.\n  {}\n  {}",
            goofi_init::FT_VENV,
            goofi_init::GIL_VENV,
        ),
        Err(e) => {
            eprintln!("goofi-init: {e}");
            std::process::exit(1);
        }
    }
}
