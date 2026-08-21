//! `cargo run -p goofi-init` — the one setup step a fresh clone needs. Idempotent.

fn main() {
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
