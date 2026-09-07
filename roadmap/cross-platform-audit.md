# Cross-platform audit, 2026-09-06

Three finders over Windows, macOS and what is written once but behaves differently. Two rounds are
done; what is LEFT is here, because a finding nobody records is a finding nobody acts on.

The one fact behind most of it: **nothing in this tree has run on a Windows or a macOS machine.**
The suite's window host is screenless on every platform, so the whole `present` path — three
platform bodies and the swizzle — has never executed anywhere. CI compiles it and no test calls it.

## Open

- **The process is DPI-unaware on Windows**, and making it aware is TWO halves. Nothing embeds a
  manifest or calls `SetProcessDpiAwarenessContext`, so DWM stretches every window on a scaled
  display and `Screen::present`'s "one texel to one pixel" is false. The call alone is not the fix:
  a DPI-aware host must also tell each plugin its scale through
  `IPlugViewContentScaleSupport::setContentScaleFactor`, or a crisp editor comes back at a third of
  its size. Both halves need a scaled Windows display to judge, which nothing here has.
- **The graphics suite needs an adapter on all three runners and CI provisions one on Linux only.**
  Reading says macOS answers with Metal and Windows with the DX12 WARP software adapter, so the job
  is green by the runner images alone and nothing in the tree says so. WARP also renders every
  shipped `.wgsl` in software against the 240-minute ceiling.
- **A node file's extension is matched case-sensitively**, so a Windows author's `Node.PY` is
  invisible with no message. Left because goofi writes the extension itself at every door it owns,
  and because the real finding underneath is that four sites — `scan.rs`, `bridge/lib.rs`, two build
  scripts — each spell "is this a `.rs`" for themselves. One owner first, then the folding.
- **`Loop::open()` cannot fail on Windows**, so the "the display is gone" state is unreachable
  there; the agent e2e scenario is POSIX-only by construction; `patchfile.rs` writes an unquoted
  filename into `Content-Disposition` — now sanitized, but the header still has no `filename*`
  form, so a non-ASCII patch name reaches the browser mangled.

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
- **`shared()` caches a device failure for the process life**, and that is right: no adapter appears
  while goofi runs, and re-asking would pay the refusal at every node.
- **The embedded asset tables are sorted.** `embed_spa` sorts after its walk and `files_under`
  sorts before it returns; a finder read the `read_dir` inside and not the `sort` after.
