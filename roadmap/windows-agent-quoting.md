# A quoted `[[agents]]` command line on Windows

An agent command that contains a double quote — `claude --append-system-prompt "…"`, and most
agent launchers eventually want one — reaches the child mangled on Windows. It runs on unix.

## What the cause is

`term::shell_command` launches the entry as `cmd /C <command>`, and portable-pty builds the
Windows command line with `CommandLineToArgvW` rules: the argument is wrapped in `"` and every
inner `"` is escaped `\"`. `cmd.exe` parses neither — a backslash is a literal to it, and each
quote toggles — so the child receives an unbalanced quote. Seen as `sh: -c: line 1: unexpected
EOF while looking for matching '"'`.

Nothing goofi writes is wrong. The two conventions are genuinely different, and portable-pty
0.9 exposes only the argv one — `CommandBuilder` has no raw-command-line door.

## Decisions already taken

- NOT worked around in the test fixture's spelling and left unrecorded. `goofi-tests`' `_deaf`
  entry states its trap handler as a shell FUNCTION so the line needs no `"` at all, which keeps
  one spelling running under both launchers — but that is the fixture dodging the defect, not
  the defect being fixed, and a user's config cannot be asked to dodge it.
- The launcher stays the user's own shell. `cmd` is what a Windows terminal runs; routing every
  platform through a POSIX shell would need one on Windows, which is not a given.

## Open

- Whether the fix is upstream (a raw-command-line constructor in portable-pty, which is where it
  belongs) or local (`cmd /S /C` with the argument pre-shaped so portable-pty's transform lands
  on what cmd wants — exact, but it encodes another crate's escaping rules in this one).
- Whether `%COMSPEC%` should name the launcher rather than a literal `cmd`, which is the same
  question one layer up: the config could name the shell per entry instead.
