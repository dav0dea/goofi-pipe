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

## Fixed, third round

- **The tablet Playwright failure was never about the tablet.** `touch.spec.ts` turns the sole panel
  into a CONTROL panel and never hands it back, and a panel's type lives in the running patch — so
  the next spec on that worker booted into a page with no node editor, nothing subscribed, and
  `integrity`'s streaming poll timed out 30 s later on whichever project drew the short straw. Which
  project that is follows the worker count, which is why it read as a viewport. The panel type is the
  fifth leakable global in `expectPristineWorkspace` now, named in 2.1 s instead of 30, and
  `restorePanelType` is the `finally` half. Reproduced locally at `GOOFI_E2E_WORKERS=1`, both ways.
- **A cleared global source could still write the global.** A pick crosses a channel from the
  reducer thread to the follower, so one made while the binding stood was applied after it was
  cleared — over whatever the author typed next. `follow` knew the machine names and the value lock
  and not the OWNER, which is the whole of it. macOS found it as a redo that rebuilt every field but
  one; ten runs of the broken variant here never hit the window, so what stands behind the fix is
  that CI failure and the path, not a local reproduction.
- **A looped WAV file left a DC offset on the output forever.** `AudioPlayback` sets `ended` at the
  end of a file and lays one quiet chunk with it, because the DSP half HOLDS its last sample on an
  empty ring. A wrap under `loop` seeks back and left `ended` set, so the next real end laid no quiet
  and the engine held the file's last sample for good. macOS caught it and Linux did not, because
  what a take ends on depends on where a control landed inside a block: the scenario now records a
  SQUARE into the tail of its take, so the held sample is full scale and the step fails on any
  machine that has the defect. Verified both ways on Linux.

## Fixed, second round

- **A macOS VST3 plugin never got `InitModule()`.** `bundleEntry` took NULL, and the SDK's
  `macmain.cpp` runs `InitModule()` only inside `if (ref)` and returns `true` either way, so goofi
  read success and the plugin's own initialization never ran. It gets a real `CFBundleRef` now,
  from the bundle three levels above the binary.
- **`gpu.rs` asked one adapter and treated a refusal as final**, and never read `WGPU_BACKEND`.
  Windows is the platform with two backends, and a broken Vulkan ICD there took DX12 down with it.
  It reads the environment and falls back to the software adapter — WARP, on Windows.
- **`stop_all` did not wait, so Windows could not delete a retired mount.** It is gone: `reap_all`
  is the one teardown, and both callers get the bounded wait that exit already had.
- **A `.gfi` merged two workspace files into one** on a case-insensitive filesystem, and the load
  reported success. Both ends refuse the pair now, naming it.
- **A node's source rode an environment variable** against Windows's 32767-character block. It
  rides stdin, whose size is the pipe's problem.
- **`goofi-build` left a `.tmp.<pid>` behind on a lost rename**, which Windows also answers when
  the target is a mapped DLL. A lost race cleans up and reads as the success it is.
- **`STYLE` had no `WS_CLIPCHILDREN`**, so a parent paint flickered over an embedded VST3 editor.
- **`.gfi` and `.vst3` were matched case-sensitively**, so `Patch.GFI` was not a patch.
- **CI started three 240-minute jobs per push and cancelled none.** There is a `concurrency` group.

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
