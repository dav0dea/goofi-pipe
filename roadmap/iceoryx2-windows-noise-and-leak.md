# iceoryx2 on Windows: 101 MB of stderr a run, and node dirs that never go

Two upstream defects, both in `iceoryx2-pal-posix`, both reproduced against 0.9.3 AND `main`
(@5ece349). Nothing here is goofi's to fix in this tree; what is ours is knowing which is which,
so neither gets re-diagnosed from scratch a third time.

## What is measured

One `cargo test --workspace` on Windows emits **87,248** `< Win32 API error >` records at ~1,202
bytes each — **~101 MB of unbuffered stderr**, which is the whole of the run's log.

| site | error | records | what it is |
|---|---|---|---|
| `mman.rs:191` `FindNextFileA` | `18` NO_MORE_FILES | 79,731 | pure log noise |
| `unistd.rs:382` `RemoveDirectoryA` | `145` DIR_NOT_EMPTY | 7,458 | a real leak, failing |
| `stat.rs:72` `SetFileSecurityA` | `2` FILE_NOT_FOUND | 54 | a cleanup race |
| `fcntl.rs:306` `LockFileEx` | `33` LOCK_VIOLATION | 5 | — |

## Decisions already taken

- **The 91% is a missing `ignore` keyword, and nothing more.** `ERROR_NO_MORE_FILES` is
  `FindNextFileA`'s documented loop terminator. `win32call!` already takes an ignore list, and
  `dirent.rs:91` — the same call, in the same crate — passes it and emits ZERO records. That pair
  is the proof, and it is why "it is only noise" is the right reading of this one.
- **The 9% is NOT noise, and the DACL is why.** `.port_tag` files carry a PROTECTED DACL granting
  `BUILTIN\Users: Read, Synchronize` — so the owner cannot unlink its own file, and because the
  DACL is protected, the parent directory's `FILE_DELETE_CHILD` never reaches it either. POSIX
  `unlink` is governed by the directory; both routes are severed. This is upstream #1869.
  Measured here: 1,561 node directories stranded for seven days.
- **`ERROR_ACCESS_DENIED` never appears in the log**, and that is a trap. The unlink attempt is not
  routed through `win32call!` — only the downstream `rmdir` is. Reading the codes alone says
  "cleanup ordering"; reading the DACL says "permissions". Read the DACL.
- **Upgrading to `main` fixes neither.** Both call sites are still bare there. The #1808 fix
  (2026-07-10, already in 0.9.3) reached `dirent.rs` and missed the identical `mman.rs`.
- **Not patched locally.** Actively-developed dependency, upstream-tracked; a `[patch.crates-io]`
  here would be symptom-hiding and would rot. The write-up for upstream is prepared.

## The manual reclaim

Not a fix — for a clean measurement only, exactly as `/dev/shm/iox2_*` is on unix. The owner keeps
implicit `WRITE_DAC`, so:

```
icacls C:\Temp\iceoryx2 /grant "%USERNAME%":(OI)(CI)F /T
rmdir /s /q C:\Temp\iceoryx2
```

## The race now fails tests, not just logs

CI 2026-08-28 (run 33137431047, `subpatches` on windows-latest), two new signatures of the same
`stat.rs:72` cleanup race, under nothing more than routine node churn in one process:

- `PublishSubscribeCreateError(InternalFailure)` creating a node's `_sts` status service, right
  after `SetFileSecurityA` `[ 2 ]` — the DACL write raced the file it was for. The node never
  reports ready; the harness's `ready()` now fails fast wearing exactly this error, which is what
  turned the former 90-second silent wedge into a diagnosis.
- A hard PANIC in `iceoryx2-bb-posix` `Directory::new` — `"This should never happen!"`, dirfd
  invalid on `<root>/nodes/<id>` — unwinding through the caller's thread. Not goofi's sweep: all
  three automatic cleanup passes are off and `reclaim_stale_resources` had long finished; this is
  the PAL enumerating under its own concurrent create/remove.

Both are flake-grade (the identical commit passed the run before), so a red Windows job needs this
file read before anything local is "fixed".

CI 2026-08-28 again (runs 33187933365 and 33188420611, `bundles` on windows-latest): the bundle
sessions boot EIGHT subprocess nodes at once, each with its own iceoryx2 node and a probe beside
it, and that burst trips the race on nearly every run rather than one in several. Two more
signatures of the same family:

- `DeadNodeView` — `"Unable to acquire monitor cleaner since the Node is still alive"` — panicking
  inside a Python CHILD interpreter (`pyo3_runtime.PanicException`), which the parent reads only as
  `subprocess exited: exit code: 1` and the node wears as its error.
- `NodeCreationFailure::InternalError` creating the iceoryx2 node a test PROBE needs, and
  `PublishSubscribeOpenOrCreateError` opening a producer's output service.

- Once, on run 33188891274: `thread 'goofi-Buffer' has overflowed its stack` —
  `STATUS_STACK_OVERFLOW` in a NODE thread (a default 2 MB stack, and nothing of goofi's recurses)
  while the `dirent.rs:66` records were streaming. Not diagnosed: it needs a Windows machine and a
  backtrace, and a stack bump without one would be the symptom-hiding this file refuses.

The sessions are not the cause and are not thinned for it: a real patch boots this many nodes.
Until the upstream report lands, the Windows job is red on `bundles` and green on nothing less.

Not only Windows: on Linux (2026-08-28, local), `what_a_crash_left_behind_is_gone_by_the_next_start`
left the dead child's node directories standing ONCE in a full `transport` target run and never
when run alone — the sweep enumerating under a sibling test's concurrent node churn, which is the
same shape as the PAL race above with a quieter failure. Flake-grade; undiagnosed past that.

## Open

- Whether goofi should reclaim the leak itself at startup rather than wait for upstream. It
  already pre-creates `<root>/nodes` and `<root>/services`, so it has an opinion about that
  directory — but sweeping another library's bookkeeping is exactly the mirror this file's
  neighbours warn about, and a stale entry from a LIVE peer must never be swept.
- Whether the noise deserves any local mitigation before a release lands. Stderr filtering is
  ruled out: a pipe-based filter DEADLOCKS the Python subprocess tier, which was measured.
- Whether a bounded retry on service CREATE (any platform, no `cfg`) is boundary tolerance or
  symptom-hiding. It would absorb the first signature and cannot absorb the second (a panic has
  no retry), so it buys half a fix at most — parked until the upstream report lands an answer.
