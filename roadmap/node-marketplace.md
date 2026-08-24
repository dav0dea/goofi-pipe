# Node marketplace

Not designed. Recorded here because the shape of the node library and the shape of a marketplace
constrain each other, and deciding the library first without holding this in view would be a
decision made twice.

## The idea

A patch already carries its own node files in its workspace, and a `.gfi` already packages them —
so a patch that opens on another machine brings its nodes with it. A marketplace is the next step
of the same fact: a way to **find, install and update node packages** that are not the shipped
library and not this patch's own.

## What is already true and works in its favour

- A `.gfi` is a zip with a `workspace/` tree, so a node package is already a shippable unit.
- Node discovery is a probe over a directory, and `--extra-nodes DIR` already adds directories to
  the scan, later winning a shared type name. A marketplace install is a directory.
- A node that cannot load is registered UNAVAILABLE with its missing dependency named, so a
  package with unmet requirements degrades legibly instead of vanishing.
- Provenance already rides every palette row — builtin vs this patch's own. A third value is a
  small change.

## Needs

- A package format: what a node package is beyond a directory of `.py` files — a manifest, a
  version, declared dependencies, and how those dependencies reach the two interpreters.
- Versioning and update semantics for a patch that names a package it was authored against.
- Where packages live on disk, and how that interacts with the per-patch workspace.
- Trust. A node package is arbitrary code executed on the user's machine; there is no sandbox
  today and adding one is a project of its own. A panel plugin has the same problem one step worse
  (see `panel-plugins.md`), so distribution and trust want ONE answer for both.
- Discovery and distribution — whether that is a registry goofi hosts, a git URL, or a directory
  someone drops in.

## Open questions

- Is a marketplace node a Python node, a Rust plugin, or both? A Rust plugin needs a stable ABI
  or a rebuild, neither of which exists.
- Does the marketplace ship nodes, or whole patches? The two are the same artefact today.
- What the relationship is to the built-in library: does a marketplace node ever get promoted
  into the shipped set, and what does that mean for the patches that named it?
