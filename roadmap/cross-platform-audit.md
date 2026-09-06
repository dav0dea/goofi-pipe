# Cross-platform audit, 2026-09-06

Three finders over Windows, macOS and what is written once but behaves differently. What was fixed
is in the commit; what is LEFT is here, because a finding nobody records is a finding nobody acts on.

The one fact behind most of it: **nothing in this tree has run on a Windows or a macOS machine.**
The suite's window host is screenless on every platform, so the whole `present` path — three
platform bodies and the swizzle — has never executed anywhere. CI compiles it and no test calls it.

## Open, and worth doing next

- **A macOS VST3 plugin never gets `InitModule()`.** `vst3/module.rs:40` passes NULL to
  `bundleEntry`. The SDK's `macmain.cpp` runs `InitModule()` only inside `if (ref)` and returns
  `true` either way, so goofi reads success and the plugin's own initialization never ran. The fix
  needs a `CFBundleRef` from `CFBundleCreate`; `objc2-core-foundation` is already in the lock.
- **The process is DPI-unaware on Windows.** Nothing embeds a manifest or calls
  `SetProcessDpiAwarenessContext`, and `framed()` uses the system-DPI `AdjustWindowRect`. On a
  scaled display DWM stretches the window, so `Screen::present`'s "one texel to one pixel" is false
  and every plugin editor is bitmap-scaled.
- **`STYLE` has no `WS_CLIPCHILDREN`**, so a parent paint flickers over an embedded VST3 editor.
- **`gpu.rs` asks one adapter and treats a refusal as final**, and never calls `.with_env()`, so
  `WGPU_BACKEND=dx12` does nothing. Windows is the platform with two backends, and a broken Vulkan
  ICD there takes DX12 down with it. `shared()` then caches the `Err` for the process life.
- **The graphics suite needs an adapter on all three runners and CI provisions one on Linux only.**
  Reading says macOS answers with Metal and Windows with the DX12 WARP software adapter, so the job
  is green by the runner images alone and nothing in the tree says so. WARP also renders every
  shipped `.wgsl` in software against the 240-minute ceiling.
- **`stop_all` does not wait, so Windows cannot delete a retired mount.** `taskkill` is
  asynchronous and a live process holds its working directory, so `remove_dir_all` fails and the
  error is discarded. Every `session new`, `session load` and test teardown leaks a temp tree.
  `reap_all` already has the bounded wait this needs.
- **Up to eight e2e backends now share one `GOOFI_BUILD_DIR`.** `goofi-build` discards a failed
  rename and Windows refuses to rename onto a mapped DLL, so a lost race leaves a `.tmp.<pid>`.
- **A `.gfi` can merge two workspace files into one** on a case-insensitive filesystem: `zip.extract`
  writes one file per entry and the second truncates the first, and the load reports success.
  `camel()` has the same collision on Linux, where both files can exist at once.
- **CI starts three 240-minute jobs per push and cancels none** — no `concurrency:` group.
- Smaller: `Loop::open()` cannot fail on Windows, so the "display is gone" state is unreachable
  there; the agent e2e scenario is POSIX-only by construction; `subproc.rs` puts a whole node source
  into an environment variable against Windows's 32767-character block; extension tests are
  case-sensitive, so `Patch.GFI` and `Plug.VST3` are not seen; `patchfile.rs` builds one op argument
  with `to_string_lossy` rather than `to_slash`, and puts an unquoted filename in
  `Content-Disposition`; the embedded asset table is built from an unsorted `read_dir`.

## Judged and NOT a defect

- **Byte order.** `goofi-codec` states little-endian and writes `to_le_bytes`; the numpy ingest
  rejects `>`; WAV and NPY specify LE. The GPU sites use the wire's spelling for native memory,
  which is only cosmetically wrong. No `bytemuck`, no `align_to`, no buffer transmute, so aarch64
  has no alignment fault waiting either.
- **iceoryx2 on macOS**: the 31-character `shm_open` limit does not bite — the macOS PAL generates a
  short name and keeps the mapping in a state file, so the long derived service names pass.
- **The `win.rs` present path type-checks** against windows-sys 0.61.2 term for term, including the
  negative `biHeight` for a top-down DIB and the DWORD-aligned scan lines.
- **Every selector and enum value in `mac.rs`** matches `objc2-app-kit`'s own bindings, and objc2's
  encoding assertions accept the integer widths used.
