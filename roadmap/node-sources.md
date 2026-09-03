# Node sources: what is still open after dynamic Rust nodes landed

Built 2026-09-02. A node is a file in `nodes_<engine>/` under a root, in either language; the
roots are the shipped tree, then each `--extra-nodes` directory, then the patch's own `workspace/`,
a later one winning a shared type name. A Rust node is one `.rs` file that `goofi-build` wraps in a
generated crate, builds through a nested cargo into one content-keyed cache under
`$GOOFI_HOME/.goofi/build/`, and the engine loads through the SDK's ABI. The shipped folders are
prebuilt at goofi's build time and embedded; a boot materialises them under
`.goofi/shipped/<version>/`, so a shipped node loads with no toolchain and `library get` answers
its file. `AGENTS.md` states the contract; the code holds the mechanism.

## Decisions taken

- **The signal ABI is the subprocess tier's codec.** One encode and one decode per side per run —
  a copy a compiled-in node never paid. It is the price of ONE seam for every out-of-crate signal
  node, and it is accepted: a node that needs the copy gone is a node that belongs in an engine,
  not in a folder. The audio ABI has no frame to encode: a block is the arena's own memory, and it
  crosses as descriptors of it.
- **A build is synchronous, off the graph lock.** `library refresh` and `session load` answer when
  every `.rs` under every root is built or has its failure memoised; only the caller who asked
  waits. A file that does not compile is a greyed type carrying rustc's words, and an instance
  built from the last good file runs on.
- **The cache is content-keyed and never re-checked.** The key is goofi's version, the SDK's hash,
  the SDK's allowlist and the source; a hit is trusted without opening cargo, and an artifact
  under its key is whole or absent, never rewritten under a process that has it mapped. A
  dependency allowlist per SDK is the whole envelope an authored node gets.
- **A failure is retried, never cached.** The prebuild runs cargo for every file with no artifact
  and keeps what it said for the process; the scan reads an artifact or that memo and can never
  reach cargo, so nothing transient — a signal, a full disk, no network — outlives the next
  refresh.
- **A stem outside the name rule is not a node**, in either language, the same as a `_` stem: the
  type name IS the stem, and a name the rule refuses would be one nothing can reference.
- **`nodes_audio/` and its SDK landed with the audio engine** (2026-09-02, `audio-engine.md`): a
  second entry in `goofi-build`'s sdk list, the same pipeline, the same three symbols; the audio
  ABI crosses the block as descriptors of the arena's regions rather than codec bytes. The two
  SDKs now share ONE boundary half (`goofi-node::abi`) — the version and describe symbols, the
  byte slice and the collector — because they had already drifted into two spellings of it.

## Open

- **The cache only grows.** Every source hash leaves an artifact behind, and nothing records which
  keys a loaded type or a saved patch still wants, so no sweep can tell a corpse from a hit.
- **A build in progress says nothing.** A node's lifecycle models "not ready yet"; a TYPE's does not,
  so a refresh that compiles for twenty seconds is twenty seconds of silence on every transport.
- What a Rust node buys over a Python one, measured through this boundary rather than assumed —
  `builtin-nodes.md` keeps that question.
