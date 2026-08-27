# A goofi config folder

`$GOOFI_HOME/.goofi/` exists now — `goofi_core::home` owns it, `config.toml` holds the agents
list, and `sessions/` holds the running servers' records. What remains here is what has not
moved into it yet.

## What still moves into it

- **The agent orientation.** `orientation.md` is `include_str!`'d into the binary and laid into
  each new workspace. It should be a file the user can edit, with the compiled-in copy as the
  default — the same seed-once pattern `config.toml` uses.
- **A skills corpus** the agent can be pointed at.
- **App defaults** — default `ufreq`, port, bind address.

## Decisions already taken

- The path is `$GOOFI_HOME/.goofi/`, `GOOFI_HOME` read per call and falling to the OS home.
- A missing config reads as the default without writing; only the serve path seeds the file, so
  the FILE stays the one owner and a test process never writes into a real home.
- A malformed config degrades to the default in memory and reports why; it never stops the app.
- The override is `GOOFI_HOME` alone, not a flag, and the path is the same on every platform —
  no XDG or AppData branch.

## Open

- Precedence when app defaults arrive: shipped default, then the config, then the flag.
