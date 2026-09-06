# The VST3 scan is what a boot costs, and it pays twice for nothing

Measured 2026-09-06 on a Windows machine carrying ~150 VST3 bundles: **130 s cold, 62 s warm**, to
a live headless session. Neither number is the shared memory — `C:\Temp\iceoryx2` had 939 stranded
node directories at the time and reclaiming them changed nothing measurable, because a node's cache
entry is opened by name and never enumerated (`iceoryx2-windows-noise-and-leak.md` carries that
correction). Both numbers are the plugin scan, and they are two separate defects.

## The 68 s: a rebuild invalidates every plugin

`described()` in `backend/audio/goofi-audio/src/vst3/mod.rs` keys the cache on

```
Sha256(the plugin binary's bytes ++ stamp_bytes(scanner))
```

and `stamp_bytes` is the SCANNER's length and mtime. The scanner is goofi's own binary. So every
`cargo build` that relinks it mints a fresh key for all ~150 plugins, and the next boot rescans the
lot in subprocesses, one at a time.

The rationale beside it is sound and is not what is wrong: *what a host reads out of a plugin is the
host's answer as much as the plugin's*, so the host has to be in the key. mtime is the wrong spelling
of "the host", because it changes on every build of the same code. Measured consequence:
`$GOOFI_HOME/.goofi/build/vst3` held **5,099 entries, 180 MB** — about thirty-four generations of the
same hundred and fifty plugins, one per rebuild, and nothing ever collects them.

Open: what the host's identity should be instead. `CARGO_PKG_VERSION` is stable across rebuilds and
moves when a release does, which is right for a user and wrong for whoever is editing the scanner
that week. A content hash of the scanner binary would be exact, and is not stable on Windows, where
a fresh `.pdb` timestamp lands in the image. Both are one line; the choice is which lie is cheaper.

## The 62 s: a failure is never cached, so it is paid every boot

The same function writes the cache only on `Ok` — the `.part` is renamed in after a scan that
parsed. A plugin that crashes the scanner, times out or will not load writes nothing, so it is
retried in full on **every** boot, forever. Six of them on this machine:

| node | what the scanner did |
|---|---|
| `audio:GuitarRig7` | exited `-1073741819` — `0xC0000005`, access violation |
| `audio:Kontakt7` | the same crash |
| `audio:Kontakt8` | exited `-1073741811` — `0xC000000D` |
| `audio:WaveShellVST392X64` | did not answer in 20 s, which is the ceiling |
| `audio:AutoTuneSlice` | `LoadLibraryExW failed` |
| `audio:AutoTuneVocodist` | the same |

That is three crashes, two load failures and one full 20 s timeout on every start, and the timeout
alone is a third of the warm boot. A wedged scan can also survive its parent: one `vst3-scan` child
on Guitar Rig 7 was still resident **21 hours** after the session that spawned it, unkillable by
`taskkill /F` because it is blocked inside the plugin, and holding 127 s of CPU.

Open: whether a failure caches. Caching it under the same key is safe on the face of it — a vendor
update changes the binary's bytes, so a plugin that is fixed gets a new key and is retried once. What
it costs is that a plugin failing for a reason OUTSIDE its bytes, a missing runtime being the
obvious one, stays greyed until the cache is cleared by hand. A negative entry that records WHY, and
a `library refresh` that discards negatives, is the shape that answers both.

Not open: the ceiling stays. A plugin that hangs must not hang the boot, and 20 s is only expensive
because it is paid every time rather than once.
