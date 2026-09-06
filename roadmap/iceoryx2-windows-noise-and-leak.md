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

Not a fix — for a clean measurement only, exactly as `/dev/shm/iox2_*` is on unix, and never with
another goofi alive. The owner keeps implicit `WRITE_DAC`, so:

```
icacls C:\Temp\iceoryx2 /grant "%USERNAME%":(OI)(CI)F /T
rmdir /s /q C:\Temp\iceoryx2
```

**Measured 2026-09-06: that pair is not enough, and the reason is the word PROTECTED.** Against 939
stranded node directories it took 3,085 files down to 1,218 and stopped. `/T` grants an INHERITABLE
ace and walks containers; a `.node_monitor_context` carries a protected DACL, which is precisely a
DACL that refuses inheritance — so icacls reports "Failed processing 0 files" while the file's own
ace list still reads `SYSTEM:(F) Administrators:(F) Users:(R)` and the owner still cannot unlink it.
Naming the file DIRECTLY works, because that writes an explicit ace rather than an inherited one:
`icacls <file> /grant "%USERNAME%":(F)`. So the reclaim is per-file and recursive, not a `/T` walk —
in PowerShell, `[System.IO.File]::SetAccessControl` over a fresh `FileSecurity` carrying one
FullControl rule, which touches the Access section alone (`Set-Acl` fetches the SACL too and dies
on `SeSecurityPrivilege`). 622 files, then the tree removes.

That reclaim was NOT what made boots quick, and the record should say so: 939 stranded directories
cost nothing measurable, because a node's cache is opened by name and never enumerated. The boot
cost on that machine was the VST3 scan — `vst3-scan-boot-cost.md`.

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

Local, 2026-09-06, `goofi-tests --test audio` on Windows: the 64-node slab step cannot start its
nodes, each wearing `EventOpenError(ServiceInCorruptedState)` on its `_door` event service followed
by ten `ListenerCreateError::ResourceCreationFailed`. Same family, same `SetFileSecurityA [ 2 ]`
records beside it. Named here because it is the FIRST step of that scenario a Windows machine
reaches, so every later step — the watchdog's among them — is unreachable there; the way to judge a
change against those is to run the step alone and against the broken variant too.

The sessions are not the cause and are not thinned for it: a real patch boots this many nodes.
Until the upstream report lands, the Windows job is red on `bundles` and green on nothing less.

Not only Windows: on Linux (2026-08-28, local), `what_a_crash_left_behind_is_gone_by_the_next_start`
left the dead child's node directories standing ONCE in a full `transport` target run and never
when run alone — the sweep enumerating under a sibling test's concurrent node churn, which is the
same shape as the PAL race above with a quieter failure. Flake-grade; undiagnosed past that.

## The whole of the Windows red, 2026-09-06

Runs 34047760400 and 34048378434: EIGHT tests fail on `windows-latest` and every one of them is this
file. Nothing else on that job is red, and neither ubuntu nor macOS fails any of them.

| test | what it wore |
|---|---|
| `a_slot_feeds_more_consumers_than_the_iceoryx2_defaults_allow` | `PublisherCreateError::UnableToCreateDataSegment` |
| `a_multi_input_keeps_one_cell_per_wire_in_the_order_it_was_given` | the same |
| `the_analysis_nodes_read_a_known_sine_and_say_what_it_is` | `NodeCreationFailure::InternalError` |
| `a_patch_sounds_under_the_external_clock` | the `Directory::new` PANIC |
| four `signals` scenarios | `timed out waiting for …` |

The next run, on 2026-09-06 with the audio defect fixed, put the same eight up again with one
NEW signature that is worth more than the other seven: `a_complexity_node_reads_a_real_signal…`
failed on ARITHMETIC — Hjorth complexity of an 8 Hz sine read 1.2204651 against 0.9..1.1, where a
pure sine is exactly 1. A full 256-sample window that is not a sine is a window that LOST A BLOCK
and was stitched across the gap, and the PAL's `dirent` errors bracket that panic in its own
stdout. So on Windows this family is not only "a node will not start": it delivered quietly
wrong DATA, and it took a numeric oracle to see it. ONCE, though — the two Windows runs after it
ran that same test and passed, so it is flake-grade like the rest of the family rather than a
standing property. It is recorded because a lost block that nothing reports is a worse failure
than a node that refuses to start, not because it is common. The same run's `Directory::new` panic
landed on the audio test again, so the thrash theory below is answered: no.

The two transport ones are the burst again — 24 and 40 iceoryx2 nodes in a loop — and the four
timeouts are nodes whose services never came up, which is the create failure worn quietly.

Two things this adds to what is above:

- **The leak FEEDS the race.** Every stranded node directory stays in `<root>/nodes`, and that is
  the directory `FindNextFileA` walks and `Directory::new` opens. One run strands enough of them to
  make every later enumeration longer, so the window a concurrent create must miss widens as the run
  goes on — which is why the burst tests and the late targets are where it lands.
- **A panic under the graph lock takes the status thread with it.** `Directory::new` unwinds through
  a caller holding `state.graph`, and `lib.rs:421`'s `graph.lock().unwrap()` then dies on the
  `PoisonError`. Secondary, and it costs nothing while goofi does not panic — but it is why one
  upstream flake prints as two unrelated panics.

## The owner's call, 2026-09-06: ACCEPTED, and nothing here is taken

The iceoryx2 team knows and is working on a fix, so goofi waits for it. **Windows is expected red on
this family and nobody should re-diagnose it.** None of the levers below is taken — not the startup
reclaim, not the per-instance root, not the retry — because each one is goofi working around a
defect whose owner is already fixing it, and a workaround outlives the thing it works around. They
stay written down for the day the wait stops being the right answer.

The three PR branches and `rust-rewrite` are GREEN on ubuntu and macOS as of this date, so a red
Windows job on any of them is this file and needs no reading past this line.

## Open — parked on that call, not being worked

- Whether goofi should reclaim the leak itself at startup rather than wait for upstream. It
  already pre-creates `<root>/nodes` and `<root>/services`, so it has an opinion about that
  directory — but sweeping another library's bookkeeping is exactly the mirror this file's
  neighbours warn about, and a stale entry from a LIVE peer must never be swept.
- **Whether an instance gets a ROOT of its own**, which is the one lever nobody has tried and the
  only one that does not sweep: it isolates instead, so it has none of the failure mode that severed
  every other goofi on the machine. It would keep `<root>/nodes` down to one instance's entries, so
  the leak could not compound across a run. What it needs judging against is the rendezvous — the
  root is process-global to iceoryx2, and two goofis meant to see each other must share one — so it
  is a real decision about what an instance IS, not a test-only knob. The owner's call.
- Whether the noise deserves any local mitigation before a release lands. Stderr filtering is
  ruled out: a pipe-based filter DEADLOCKS the Python subprocess tier, which was measured.
- Whether a bounded retry on service CREATE (any platform, no `cfg`) is boundary tolerance or
  symptom-hiding. Still parked, but the 2026-09-06 run prices it: SEVEN of the eight failures wear a
  create-time `Err` a retry could absorb, and the eighth is the `panic!`, which nothing can. So it is
  not "half a fix" — it is most of the job, against the one signature it can never reach.
  It also cannot reach the spliced window above, which is the signature that matters most.
