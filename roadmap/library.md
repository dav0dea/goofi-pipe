# The library: node bundles and panel add-ons

One place to find, install, publish and update what goofi does not ship: **bundles** of nodes, and
**panel add-ons**. This item replaces `node-marketplace.md` and `panel-plugins.md`: distribution
and trust want one answer for both kinds of add-on, and the UI that browses them is one panel.

The model is the VCV Rack library. An author keeps the code in a git repo they host; the library
registers the repo, detects what is in it, and lets the author group the result into bundles that
anyone can install.

## What is already true and works in its favour

- Node discovery is a probe over a directory, and `--extra-nodes DIR` already adds directories to
  the scan, a later one winning a shared type name. **An installed bundle is a directory.**
- A node that cannot load is registered UNAVAILABLE with its missing dependency named, so a bundle
  with unmet requirements degrades legibly instead of vanishing.
- `$GOOFI_HOME/.goofi/` exists (`goofi_core::home`), which is where an installed bundle lands.
- The `library` op group exists — `list`, `get`, `refresh` — and every op is on every transport, so
  the CLI, an agent, the panel and a test reach the same door.
- `registerPanel` exists, and the layout stores `panel_type` as a STRING: a panel add-on's type
  already rides the document, undo, a second tab and the `.gfi` with no backend change.
- The agent panel is a worked example of a panel with its own backend process.

## Decisions

**The unit is a bundle.** A bundle is a directory with a `requirements.txt` for the Python
packages its nodes import — this exists today: checked against both interpreters at startup and
installed only on a yes in the terminal — and a manifest beside its files: a name, a version, a
description, and the goofi version it was built against. It holds `nodes_signal/` and
`nodes_audio/` — one folder per engine, `node-sources.md`'s rule — and, later, `panels/`.
A bundle is the only thing that is installed, published or updated. There is no per-node install.

**Installed bundles live in `$GOOFI_HOME/.goofi/bundles/<name>/`**, one directory each, and the
scan order becomes: the shipped tree, then each installed bundle, then this patch's own
`workspace/nodes_*/`. The precedence `node-sources.md` states is unchanged; this fills the middle
slot it left open. A bundle's name is the palette category its nodes appear under.

**The repo's own bundles live in `node-bundles/<name>/`**, and they publish through the same door a
third party's do — the shipped `nodes_*/` trees stay the shipped trees. A first-party bundle that
needs a private path is the defect that proves the door is not finished. Until the install half
exists, `--extra-nodes node-bundles/<name>` is how one is loaded.

**A source is a git repo, and publishing pins a commit.** An author registers a repo URL. A
published bundle names the repo, the commit and the paths it takes from it; an install fetches
exactly that; an update is a new pin. A bundle may take nodes from several of its author's repos,
or one repo may publish as one bundle — the latter is the default the author gets for nothing.

**Detection is ONE static scan, and the real probe stays on the user's machine.** The service
never imports a stranger's code. It parses a repo for `goofi.Node` subclasses and their declared
`INPUTS`, `OUTPUTS`, `PARAMS` and docstring — a scanner in `goofi-node`, beside `scan_nd_calls`,
that goofi runs on a local folder and the service runs on a linked repo, so the panel shows the
list the service will. Availability is decided at install by the probe every node already
passes: an import that fails names its missing package in the palette, as today.

**Everything is `library` ops, and the service speaks the remote half of them.** The vocabulary
has a local half and a remote half:

- local, against `.goofi/bundles/`: `library bundle list | install | update | remove`. Built first,
  against a path and a git URL, with no service at all.
- remote, goofi as a CLIENT of the service: `library login`, `library source add | list | remove`,
  `library search`, `library publish`. The service's HTTP API IS these phrases, so the panel's
  code calls `library source add` and does not know whether goofi forwards it or the website
  sends it directly.

An agent authors and publishes a bundle with exactly the CLI a human uses: log in, link a public
repo, define the bundle, publish.

**The panel is the website.** One Svelte panel — browse the catalog, see what is installed, and
for the logged-in author, the sources and the bundling — registered through `registerPanel` like
every panel. It is built inside goofi first, driven by the ops through the socket, and exported
to the website once it is solid, where the same code drives the same phrases over HTTP.

**Login is OAuth against the forge that hosts the repo** (GitHub, GitLab). The library holds no
passwords: the login exists to prove the author owns the repo they link, and the forge already
knows that.

**A patch names the bundles it uses.** The manifest records `name@version` for every bundle a
node on the canvas came from. A load on a machine without one offers the install and otherwise
leaves the node UNAVAILABLE with the bundle named — never a silent absence. A patch's own
`workspace/nodes_*/` is untouched by this: it still travels inside the `.gfi`.

**Trust is provenance, not a sandbox.** A bundle is arbitrary code on the user's machine, and a
panel add-on is arbitrary code in the app's origin with the control socket; neither has a sandbox
and adding one is a project of its own. What the library guarantees instead: a bundle publishes
only from a public repo, at a pinned commit, under an account the forge vouches for, and the
panel shows all three before an install. Whether anything more is needed is decided by use.

## Panel add-ons: the runtime door

A node bundle needs no new door — a directory of `.py` files is already a node source. A panel
add-on does, and the door is the hard part of this item.

What a panel add-on declares: a Svelte component rendered in its own panel and nowhere else; an
optional backend process, Rust or Python, with its own lifecycle; a set of control ops that JOIN
the one vocabulary; and its identity — a panel type id, a title, an icon, whether it takes a bound
node.

What goofi exposes back: dialogs and header-bar notifications; info/warn/error reporting at a
severity the app renders consistently; the graph, the document, the selection and the `/data`
plane; and a panel-to-panel API so two add-ons cooperate without either importing the other.

Locked, because the codebase's rules already say so:

- **The built-in panels go through the add-on door**, or the door is not proved.
- **An add-on's ops join the one vocabulary.** `plugin <name> <phrase…>` always exists as the
  explicit spelling (the prefix is RESERVED today); an add-on may also claim bare phrases, and the
  whole bare namespace — built-ins, the reserved client words and every add-on — is ONE
  prefix-free set checked at registration. The op table's rows are `&'static`, so an OWNED row
  form an add-on can register after construction is the first thing to build.
- **The document stays the one owner of the panel tree.** An add-on holds no tree, as panelty
  holds none: it raises intents and goofi turns each into one op.
- **Styling is a CONTRACT, not a shared `:root`.** The panelty token pattern is the precedent.
- **An add-on that fails to load degrades legibly and names why.** It never takes the app down.

The decision the door turns on: **the frontend bundle is compiled into the binary, and an
add-on's component cannot be.** It must arrive as JavaScript loaded at run time, which crosses the
two rules the build discipline rests on. Unresolved, and to be decided first when this half
starts: what the CSP becomes and whether an add-on gets its own origin; how an add-on is
versioned against goofi's interface and what a mismatch does; whether the boundary is Svelte or
plain DOM; and whether a panel-to-panel message is a manager-ordered op (replayable, undoable,
testable through the one interface) or a peer channel (cheaper, weaker).

## Order of work

1. **Done: `node-bundles/complexity` and `node-bundles/eeg`**, loaded with `--extra-nodes`, each
   naming its packages in a `requirements.txt` that provisioning installs and startup checks, and
   one scenario per bundle in `goofi-tests`.
2. **The bundle manifest and the local half**: `library bundle install <path | git url>` into
   `.goofi/bundles/`, the scan order, the palette category, `list`, `update`, `remove`. The
   `.gfi` records `name@version`.
3. **The static scanner** in `goofi-node`, and `library source` against a local folder, so an
   author previews the detection before any service exists.
4. **The panel**, inside goofi, against the local half.
5. **The service and the remote half**: accounts, sources, bundles, publish, search. Then the
   panel exports to the website.
6. **Panel add-ons**, once the door's first decision is taken.

## Open

- What the service is written in and where it is hosted, and what it stores in. It shares the
  scanner with goofi, so Rust is the default; nothing else is decided.
- A bundle carries a Rust audio node in `nodes_audio/` by `node-sources.md`'s rule; what a bundle
  must declare for the build (the allowlist is fixed, so nothing yet) is settled when the first
  one is published.
- Whether a shipped node is ever PROMOTED out of a bundle into `nodes_*/`, and what that does to the
  patches that named the bundle. `builtin-nodes.md` holds the other side of this line.
- A private repo as a source: the login could reach it, but "publishes only from a public repo"
  is what trust rests on above.
