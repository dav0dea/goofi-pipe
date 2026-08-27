# `agent term`: the terminal in the shell

Attach a spawned agent's PTY interactively from a terminal: `goofi agent term <instance>` opens
the `/term/<instance>` socket, puts the local TTY in raw mode, and bridges bytes both ways —
the same stream the app's agent panel draws.

## Decisions already taken

- The phrase is RESERVED (`ops::RESERVED`) and rides the prefix-freedom invariant, so the
  namespace already holds its place; the CLI answers "not built yet" and points at the panel.
- It is client-side: it is ABOUT a connection the client owns, like `session list`.

## Open

- Raw-mode handling and restore on exit/panic; resize propagation into the seat arbitration.
- Whether a detach chord is needed, or Ctrl-C simply closes the attachment.
