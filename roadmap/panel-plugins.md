# Panel plugin system

Every panel becomes a **plugin**: a self-contained system that meets goofi at one interface. The
panels that ship today become the first plugins, through the same door an external one uses.

**The motivation is external developers.** Someone must be able to publish a plugin that runs
inside goofi's web interface — data-recording infrastructure is the canonical example — without a
fork and without a goofi release. Everything below follows from that.

## What a plugin declares

- **A frontend Svelte component**, rendered in its own panel and scoped to it. A plugin draws inside
  its panel and nowhere else.
- **A backend process**, optional, in Rust or Python, with its own lifecycle.
- **A set of control ops**, which join the ONE op vocabulary rather than opening a surface of their
  own.
- Its identity: a panel type id, a title, an icon, and whether it takes a bound node.

## What goofi exposes back

The plugin interface is the point of the whole design, and it must carry more than the graph:

- **Dialog popups** and **header-bar notifications**.
- **Info / warn / error reporting**, at a severity the app renders consistently. Today the console
  is the only reporting channel there is, and it is a node log.
- The graph, the document, the selection, and the `/data` stream plane.
- **A standardized panel-to-panel API**, so two plugins can cooperate without either importing the
  other.

## What is already in its favour

- **`registerPanel` exists.** Every built-in panel is registered through panelty's own seam already,
  with `id`, `title`, `icon`, `component`, `acceptsNode` and `confirmClose`. The shape a plugin
  declares is the shape today's panels declare.
- **The layout stores `panel_type` as a STRING**, not an enum. A plugin's panel type already rides
  the document, undo, a second tab and the `.gfi` with no backend change at all.
- **A panel already owns state.** Each layout panel node carries a `state` JSON value, and the
  document already owns it.
- **A panel with its own backend process already works.** The agent panel launches a harness from a
  declared adapter, binds it to the panel, streams it, and answers its own ✕. That is one worked
  example of the hardest part.
- **The op table already has a `surface` field.** A plugin's ops joining that table, marked for
  `/control` or for MCP, is an established shape rather than a new one.

## Locked decisions

These are not choices — they are what this codebase's rules already say.

- **The built-in panels go through the plugin door**, or the door is not proved. A capability only a
  shipped panel can reach is the defect.
- **A plugin's ops join the one vocabulary.** Same names, same args, same undo, same MCP mirroring.
  Never a second RPC surface.
- **The document stays the one owner of the panel tree.** A plugin holds no tree, exactly as panelty
  holds none — it raises intents and goofi turns each into one op.
- **Styling is a CONTRACT, not a shared `:root`.** The panelty token pattern is the precedent: goofi
  publishes tokens with defaults, a plugin reads them, and a plugin cannot restyle the app.

## The hard part, and it is genuinely hard

**The frontend bundle is compiled into the binary.** A failed frontend build is a failed build, and
there is exactly one artifact. A plugin's Svelte component cannot be compiled in, so it must arrive
as JavaScript loaded at run time — which crosses the two rules the build discipline rests on. This
is the decision the whole spec turns on, and it has to be made first.

Its consequences, all of them unresolved:

- **Trust.** A plugin's code runs in the app's origin with the control socket. A node package is
  arbitrary code on the machine; a plugin is arbitrary code in the UI. Neither has a sandbox today.
- What the CSP becomes, and whether a plugin gets its own origin.
- How a plugin is versioned against goofi's own interface, and what a mismatch does.
- Whether a plugin's Svelte version must match goofi's, or the boundary is plain DOM.

## Needs

- The spec. This item is not startable without one.
- Op namespacing, so a plugin's op cannot collide with a core op or with another plugin's.
- What panel-to-panel messages ARE: a manager-ordered op, or a peer channel. The first is
  replayable, undoable and testable through the one interface; the second is cheaper and weaker.
- The plugin package format, and how it relates to a node package (see `node-marketplace.md` — the
  two want the same answer for distribution and trust).
- What a plugin that fails to load does. It degrades legibly and names why, as an unavailable node
  does; it never takes the app down.
