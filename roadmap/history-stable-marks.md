# Stable marks in the command history

`compound` marks its rollback/coalesce span with `h.len()` and releases the history lock while
the steps run. Anything that removes entries from the middle of the shared vector in that window
shifts the mark: `apply`'s retain of another actor's undone entries always could, and the
reaper's `drop_actor` (an agent exiting mid-batch) now can too. The batch then coalesces or
rolls back fewer entries than it should.

## Decisions already taken

- The stack's lifetime follows its actor, dropped by the reaper — that stays; the fix is on the
  mark, not the drop.
- The fix direction: a monotonic per-entry sequence number, so a span survives removals — an
  index does not.

## Open

- Whether `apply`'s own retain-in-window shares the fix or the history lock should simply be
  held across a compound's steps (the steps take the graph lock, so ordering needs care).
