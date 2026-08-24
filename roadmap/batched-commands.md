# One batched command op

25 of the 34 ops carry `surface: Mcp`, and each is a tool in an agent's list. One op that takes a
LIST of commands replaces them.

## Why

- **It chains.** An agent that must add four nodes and wire them spends four round trips today, and
  each one is a chance to lose the thread.
- **It cuts the tool list.** 25 schemas is a standing token cost on every agent turn, paid whether
  or not the agent touches the graph.
- **A batch is one undo step.** `Command::Compound` already executes children in order and reverses
  them for the inverse, so the machinery exists; what is missing is a door to it.

## What it must not become

A second vocabulary. The batched op takes the SAME op names with the SAME args — it is a transport
for the vocabulary, never a dialect of it. A capability reachable only in a batch, or only outside
one, is the defect this whole design exists to prevent.

## Needs

- The arg schema for a list of `{op, args}`, and how a later entry names a uid an earlier one minted.
- What happens when entry three of five fails: the whole batch inverts, because a half-applied batch
  is a state nobody asked for.
- Which ops stay individually exposed. A read is cheap and an agent uses it to decide; a batch is
  for writes.
- Whether `/control` gets the same op. It should — one programmatic interface.
