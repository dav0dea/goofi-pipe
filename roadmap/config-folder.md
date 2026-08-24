# A goofi config folder

A place for what belongs to the INSTALL rather than to a patch. It lives outside `workspace/`, so it
does not ride the `.gfi`.

## What moves into it

- **Harness declarations.** Today `static ADAPTERS` in `goofi-bridge/src/term.rs` — a compiled-in
  list of the agent CLIs goofi knows how to launch. A user cannot add one without a rebuild.
- **The agent orientation.** `orientation.md` is `include_str!`'d into the binary and laid into each
  new workspace. It should be a file the user can edit, with the compiled-in copy as the default.
- **A skills corpus** the harness can be pointed at.
- **App defaults** — default `ufreq`, port, bind address.

## Why the placement is already decided

The per-instance MCP config is written beside the workspace, and a `.gfi` packages the workspace
tree. Anything a patch should NOT carry between machines has to sit outside that boundary, and there
is no such place today.

## Needs

- The path, per platform, and what `--config DIR` overrides.
- Precedence: shipped default, then config folder, then the flag.
- What a missing or malformed config does. It degrades to the default and says so; it never stops
  the app.
