# Persistence Phase 1 — Backend Save/Load + Full-FS Browser — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Save write to the backend filesystem by default with an in-app file browser ("Save As"), keep a "Save in browser" download option, and load patches via the same browser + an Examples menu — fixing the "downloads untitled" bug.

**Architecture:** The bridge's existing `save`/`load` control ops already write/read the backend filesystem; Phase 1 adds two **filesystem-browse** ops (`list_dir`, `list_examples`) and a non-writing `serialize` op, then rewires the frontend so Save calls the backend (silent overwrite when named, an `FsBrowser` modal "Save As" when not) and Load opens the same browser. No format change — this ships on **today's flat YAML format**. (The recursive v2 format, converter, and `member_uid` move to Phase 2, where sub-patches need them.)

**Tech Stack:** Python 3.12 + aiohttp (bridge), SvelteKit / Svelte 5 runes + TypeScript strict (frontend), pytest + Playwright (sync API) for e2e.

## Global Constraints

- **No filesystem jail / no allowlist.** The browse ops expose the whole backend filesystem over absolute paths. The LAN is trusted; bind stays `0.0.0.0`. Device auth is future work, out of scope. (Verbatim from spec decision-log item 4.)
- **No node source code in `.gfi`.** Not touched in Phase 1; flat format unchanged.
- **TypeScript strict, no `any` in app code.** (Project rule.)
- **Frontend test cycle is `npm run check` (svelte-check) + Python Playwright e2e** under `e2e/` — the repo has no TS unit-test harness for `.svelte` files, so UI tasks verify via typecheck + e2e, matching existing practice.
- **Run backend tests with:** `.venv/bin/python -m pytest tests/test_fsbrowse.py tests/test_manager.py -p no:cacheprovider -q`
- **Examples dir:** `Path(__file__).resolve().parents[3] / "examples"` from `src/goofi/bridge/*.py` (== `manager.py:625`'s `parents[2]` from `src/goofi/manager.py`). May be absent under a wheel — degrade to empty, never raise.

---

## File Structure

**Create:**
- `src/goofi/bridge/fsbrowse.py` — pure FS-listing helpers (`list_dir`, `list_examples`, `examples_dir`, `_roots`). One responsibility: turn a path into a JSON-safe directory listing. No aiohttp, no manager — unit-testable.
- `tests/test_fsbrowse.py` — unit tests for the above.
- `frontend/src/lib/fs/FsBrowser.svelte` — modal file browser (roots rail + editable path bar + entry list + save-mode filename field / load-mode Open). One responsibility: pick a backend path.
- `e2e/test_save_load.py` — Playwright e2e for the save/load UX.

**Modify:**
- `src/goofi/manager.py:466-522` — extract `serialize_patch()` from `save()` (so a "Save in browser" can get YAML without writing a file).
- `src/goofi/bridge/control.py:135-258` — add `list_dir`, `list_examples`, `serialize` ops to `_dispatch`.
- `frontend/src/lib/api/control.ts` — add `FsEntry`, `FsRoot`, `DirListing` types.
- `frontend/src/lib/stores/graph.svelte.ts:299-312` — add `listDir`, `listExamples`, `load`, `serialize` store methods.
- `frontend/src/lib/editor/TopBar.svelte` — Save split-button (Save / Save As… / Save in browser); Load opens browser.
- `frontend/src/lib/app/AppShell.svelte:46-76` — rewrite `triggerSave`/`triggerLoad`, add `saveBackend`/`saveAs`/`saveInBrowser`/`uploadLoad`, mount `FsBrowser`.

---

## Task 1: Backend FS-browse module

**Files:**
- Create: `src/goofi/bridge/fsbrowse.py`
- Test: `tests/test_fsbrowse.py`

**Interfaces:**
- Produces:
  - `list_dir(path: str | None) -> dict` → `{"path": str, "parent": str | None, "entries": list[dict], "roots": list[dict]}`; each entry `{"name","path","kind":"dir"|"file","is_gfi":bool,"hidden":bool,"size":int,"mtime":float}`; each root `{"label","path"}`.
  - `list_examples() -> dict` → `{"entries": list[dict]}` (same entry shape; empty if examples dir absent).
  - `examples_dir() -> pathlib.Path | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fsbrowse.py
"""Unit tests for the bridge filesystem-browse helpers."""
from pathlib import Path

from goofi.bridge import fsbrowse


def test_list_dir_lists_entries_dirs_first(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.gfi").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    res = fsbrowse.list_dir(str(tmp_path))
    assert res["path"] == str(tmp_path.resolve())
    names = [e["name"] for e in res["entries"]]
    # dirs sort before files
    assert names[0] == "sub"
    assert set(names) == {"sub", "a.gfi", "b.txt"}
    gfi = next(e for e in res["entries"] if e["name"] == "a.gfi")
    assert gfi["kind"] == "file" and gfi["is_gfi"] is True
    assert next(e for e in res["entries"] if e["name"] == "sub")["kind"] == "dir"
    assert "roots" in res and any(r["label"] == "Home" for r in res["roots"])


def test_list_dir_on_a_file_returns_its_parent(tmp_path: Path):
    f = tmp_path / "p.gfi"
    f.write_text("x")
    res = fsbrowse.list_dir(str(f))
    assert res["path"] == str(tmp_path.resolve())


def test_list_dir_parent_is_none_at_root():
    res = fsbrowse.list_dir("/")
    assert res["parent"] is None


def test_list_dir_none_defaults_to_home():
    res = fsbrowse.list_dir(None)
    assert res["path"] == str(Path.home().resolve())


def test_list_dir_flags_hidden(tmp_path: Path):
    (tmp_path / ".secret").write_text("x")
    res = fsbrowse.list_dir(str(tmp_path))
    assert next(e for e in res["entries"] if e["name"] == ".secret")["hidden"] is True


def test_list_examples_returns_gfi_entries():
    res = fsbrowse.list_examples()
    # In a checkout the examples dir exists and holds .gfi files.
    if fsbrowse.examples_dir() is not None:
        assert all(e["is_gfi"] for e in res["entries"])
        assert len(res["entries"]) > 0
    else:
        assert res["entries"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fsbrowse.py -p no:cacheprovider -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'goofi.bridge.fsbrowse'`

- [ ] **Step 3: Write the implementation**

```python
# src/goofi/bridge/fsbrowse.py
"""Filesystem-browse helpers for the bridge control plane.

Pure functions: turn a path into a JSON-safe directory listing the browser
renders as a file picker. NO jail — goofi-pipe is a trusted single-user local
app and the user explicitly wants full-filesystem access (see the persistence
design spec, decision-log item 4). Device auth is future work.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def examples_dir() -> Optional[Path]:
    """The repo `examples/` dir, or None under a wheel that doesn't ship it."""
    cand = Path(__file__).resolve().parents[3] / "examples"
    return cand if cand.is_dir() else None


def _entry(child: Path) -> Optional[Dict[str, Any]]:
    try:
        st = child.stat()
        is_dir = child.is_dir()
    except OSError:
        return None
    return {
        "name": child.name,
        "path": str(child),
        "kind": "dir" if is_dir else "file",
        "is_gfi": child.suffix == ".gfi",
        "hidden": child.name.startswith("."),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def _roots() -> List[Dict[str, str]]:
    roots = [{"label": "Home", "path": str(Path.home())}]
    ex = examples_dir()
    if ex is not None:
        roots.append({"label": "Examples", "path": str(ex)})
    roots.append({"label": "Working dir", "path": str(Path.cwd())})
    return roots


def list_dir(path: Optional[str]) -> Dict[str, Any]:
    """List one directory level. `path` None → home. A file path → its parent."""
    base = Path(path).expanduser() if path else Path.home()
    base = base.resolve()
    if base.is_file():
        base = base.parent

    entries: List[Dict[str, Any]] = []
    try:
        children = list(base.iterdir())
    except OSError:
        children = []
    # Dirs before files, each case-insensitively name-sorted.
    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
    for child in children:
        e = _entry(child)
        if e is not None:
            entries.append(e)

    parent = base.parent
    return {
        "path": str(base),
        "parent": str(parent) if parent != base else None,
        "entries": entries,
        "roots": _roots(),
    }


def list_examples() -> Dict[str, Any]:
    ex = examples_dir()
    if ex is None:
        return {"entries": []}
    entries: List[Dict[str, Any]] = []
    for child in sorted(ex.glob("*.gfi"), key=lambda p: p.name.lower()):
        e = _entry(child)
        if e is not None:
            entries.append(e)
    return {"entries": entries}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fsbrowse.py -p no:cacheprovider -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goofi/bridge/fsbrowse.py tests/test_fsbrowse.py
git commit -m "feat(bridge): filesystem-browse helpers (list_dir/list_examples)"
```

---

## Task 2: Extract `Manager.serialize_patch()`

So "Save in browser" can fetch the current patch YAML without writing a backend file. Pure refactor — `save()` keeps its behavior.

**Files:**
- Modify: `src/goofi/manager.py:466-522`
- Test: `tests/test_manager.py` (add one test)

**Interfaces:**
- Produces: `Manager.serialize_patch(self, timeout: float = 3.0) -> str` — the `.gfi` YAML for the current graph, no disk write. `Manager.save(...)` now calls it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_manager.py`:

```python
def test_serialize_patch_returns_yaml_without_writing(tmp_path):
    import yaml as _yaml
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _build_simple_graph(mgr)
        text = mgr.serialize_patch()
        doc = _yaml.load(text, Loader=_yaml.FullLoader)
        assert set(doc.keys()) >= {"nodes", "links"}
        assert len(doc["nodes"]) == 2
        # serialize must not have created a file or set save_path
        assert mgr.save_path is None
    finally:
        mgr.terminate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manager.py::test_serialize_patch_returns_yaml_without_writing -p no:cacheprovider -q`
Expected: FAIL — `AttributeError: 'Manager' object has no attribute 'serialize_patch'`

- [ ] **Step 3: Refactor `save()` to extract `serialize_patch()`**

In `src/goofi/manager.py`, replace the body of `save()` from the line `print("Saving manager state...")` through `manager_yaml = yaml.dump(patch, sort_keys=False)` (currently `manager.py:493-515`) so the serialization lives in a new method. The new method and the slimmed save:

```python
    def serialize_patch(self, timeout: float = 3.0) -> str:
        """Serialize the current graph to `.gfi` YAML text, without writing.

        Reads each node's pushed `serialized_state` directly (waiting briefly
        per node if it hasn't pushed yet) and merges its gui_kwargs. Shared by
        `save()` (writes to disk) and the bridge `serialize` op ("Save in
        browser" download)."""
        serialized_nodes: Dict[str, Any] = {}
        # Snapshot names first: a patch may still be spawning nodes on another
        # thread, and iterating the live container would raise.
        for name in list(self.nodes):
            ref = self.nodes[name]
            ref.wait_for_state(timeout=timeout)
            if ref.serialized_state is None:
                raise RuntimeError(f"Node {name} does not have a serialized state. Recreate the node and try again.")
            state = deepcopy(ref.serialized_state)
            state["gui_kwargs"] = ref.gui_kwargs
            # Drop output-subscriber bookkeeping — transient runtime state.
            state.pop("output_subscribers", None)
            serialized_nodes[name] = state

        patch: Dict[str, Any] = {"nodes": serialized_nodes, "links": list(self._links)}
        if self.layout is not None:
            patch["layout"] = self.layout
        return yaml.dump(patch, sort_keys=False)
```

Then in `save()`, replace the serialization block (everything from `print("Saving manager state...")` to the `with open(...)` write) with:

```python
        print("Saving manager state...")
        manager_yaml = self.serialize_patch(timeout=timeout)

        with open(filepath, "w") as f:
            f.write(manager_yaml)
```

Leave the path-resolution block above `print("Saving manager state...")` and the `print(f"Successfully saved...")` / `self.save_path = filepath` / `self.unsaved_changes = False` below it unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manager.py::test_serialize_patch_returns_yaml_without_writing -p no:cacheprovider -q`
Expected: PASS

Then run the save-related regression: `.venv/bin/python -m pytest "tests/test_manager.py" -k "save or load" -p no:cacheprovider -q`
Expected: PASS (existing save/load tests still green)

- [ ] **Step 5: Commit**

```bash
git add src/goofi/manager.py tests/test_manager.py
git commit -m "refactor(manager): extract serialize_patch() from save()"
```

---

## Task 3: Wire `list_dir` / `list_examples` / `serialize` control ops

**Files:**
- Modify: `src/goofi/bridge/control.py` (imports near line 28; `_dispatch` near line 255, before the `if op == "ping"` branch)
- Test: `tests/test_control_ops.py` (Create)

**Interfaces:**
- Consumes: `fsbrowse.list_dir`, `fsbrowse.list_examples` (Task 1); `Manager.serialize_patch` (Task 2).
- Produces: control ops `list_dir {path?} -> DirListing`, `list_examples {} -> {entries}`, `serialize {} -> {yaml}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control_ops.py
"""The bridge control ops added for Phase-1 save/load (no live socket)."""
import asyncio
import types
from pathlib import Path

from goofi.bridge.control import ControlHub


def _hub(manager=None) -> ControlHub:
    # ControlHub only needs `server.manager` for these ops; no real server.
    return ControlHub(types.SimpleNamespace(manager=manager))


def test_list_dir_op(tmp_path: Path):
    (tmp_path / "x.gfi").write_text("x")
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_dir", {"path": str(tmp_path)}))
    assert res["path"] == str(tmp_path.resolve())
    assert any(e["name"] == "x.gfi" for e in res["entries"])


def test_list_examples_op():
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_examples", {}))
    assert "entries" in res


def test_serialize_op_calls_manager():
    class FakeMgr:
        def serialize_patch(self):
            return "nodes: {}\nlinks: []\n"
    hub = _hub(FakeMgr())
    res = asyncio.run(hub._dispatch("serialize", {}))
    assert res["yaml"].startswith("nodes:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_control_ops.py -p no:cacheprovider -q`
Expected: FAIL — `ValueError: unknown op: list_dir`

- [ ] **Step 3: Implement the ops**

In `src/goofi/bridge/control.py`, add the import alongside the existing schema import (after line 32):

```python
from goofi.bridge import fsbrowse
```

Then in `_dispatch`, immediately before `if op == "ping":` (currently `control.py:255`), insert:

```python
        if op == "list_dir":
            return await self._call_manager(fsbrowse.list_dir, payload.get("path"))
        if op == "list_examples":
            return fsbrowse.list_examples()
        if op == "serialize":
            yaml_text = await self._call_manager(manager.serialize_patch)
            return {"yaml": yaml_text}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_control_ops.py -p no:cacheprovider -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goofi/bridge/control.py tests/test_control_ops.py
git commit -m "feat(bridge): list_dir/list_examples/serialize control ops"
```

---

## Task 4: Frontend types + store methods

**Files:**
- Modify: `frontend/src/lib/api/control.ts` (after the `LinkInfo` block, ~line 46)
- Modify: `frontend/src/lib/stores/graph.svelte.ts` (after `loadText`, ~line 312)

**Interfaces:**
- Produces (control.ts):
  ```ts
  export interface FsEntry { name: string; path: string; kind: 'dir' | 'file'; is_gfi: boolean; hidden: boolean; size: number; mtime: number; }
  export interface FsRoot { label: string; path: string; }
  export interface DirListing { path: string; parent: string | null; entries: FsEntry[]; roots: FsRoot[]; }
  ```
- Produces (graph store): `listDir(path?: string): Promise<DirListing>`, `listExamples(): Promise<{ entries: FsEntry[] }>`, `load(path: string): Promise<void>`, `serialize(): Promise<{ yaml: string }>`.

- [ ] **Step 1: Add the types to `control.ts`**

After the `sameLink` function (line 62), add:

```ts
export interface FsEntry {
	name: string;
	path: string;
	kind: 'dir' | 'file';
	is_gfi: boolean;
	hidden: boolean;
	size: number;
	mtime: number;
}

export interface FsRoot {
	label: string;
	path: string;
}

export interface DirListing {
	path: string;
	parent: string | null;
	entries: FsEntry[];
	roots: FsRoot[];
}
```

- [ ] **Step 2: Add the store methods to `graph.svelte.ts`**

Update the import from `$lib/api/control` (line 8-17) to add `DirListing` and `FsEntry`:

```ts
import {
	getControl,
	paramValues,
	sameLink,
	type ControlEvent,
	type DirListing,
	type FsEntry,
	type GraphSnapshot,
	type LinkInfo,
	type NodeInstanceInfo,
	type NodeTypeInfo
} from '$lib/api/control';
```

Then, immediately after the `loadText` method (line 310-312), add:

```ts
	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return getControl().call<DirListing>('list_dir', { path });
	}

	/** The bundled example patches (empty under a wheel without examples/). */
	async listExamples(): Promise<{ entries: FsEntry[] }> {
		return getControl().call<{ entries: FsEntry[] }>('list_examples');
	}

	/** Load a patch from a BACKEND filesystem path (destructive — replaces the graph). */
	async load(path: string): Promise<void> {
		await getControl().call('load', { path });
	}

	/** Current patch as `.gfi` YAML, without writing to disk (for browser download). */
	async serialize(): Promise<{ yaml: string }> {
		return getControl().call<{ yaml: string }>('serialize');
	}
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npm run check`
Expected: PASS (0 errors). If `npm run check` reports unused-import warnings for `FsEntry`/`DirListing`, they are consumed by Tasks 5/6; leave them — they resolve once those land. (If the project's check fails on unused imports, defer adding the import line until first use; note this and proceed.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api/control.ts frontend/src/lib/stores/graph.svelte.ts
git commit -m "feat(frontend): FS-browse types + listDir/listExamples/load/serialize store methods"
```

---

## Task 5: `FsBrowser.svelte` modal

**Files:**
- Create: `frontend/src/lib/fs/FsBrowser.svelte`

**Interfaces:**
- Consumes: `graph().listDir`, `FsEntry`, `FsRoot` (Task 4).
- Produces: a component with props
  ```ts
  { mode: 'save' | 'load'; initialPath?: string | null; suggestedName?: string;
    onPick: (path: string) => void; onClose: () => void; onUpload?: () => void; }
  ```
  `onPick` receives a full absolute path (save mode: `dir/name.gfi`; load mode: the selected file).

- [ ] **Step 1: Write the component**

```svelte
<!--
  Backend filesystem browser modal. Two modes: 'save' (pick a directory + type a
  filename) and 'load' (pick an existing .gfi). Full-FS, no jail (trusted LAN).
  Renders in the top-level modal band above all other chrome.
-->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import type { FsEntry, FsRoot } from '$lib/api/control';
	import { onMount, tick } from 'svelte';

	type Props = {
		mode: 'save' | 'load';
		initialPath?: string | null;
		suggestedName?: string;
		onPick: (path: string) => void;
		onClose: () => void;
		onUpload?: () => void;
	};
	const { mode, initialPath = null, suggestedName = '', onPick, onClose, onUpload }: Props = $props();

	const g = graph();
	let cwd = $state('');
	let parent = $state<string | null>(null);
	let entries = $state<FsEntry[]>([]);
	let roots = $state<FsRoot[]>([]);
	let pathDraft = $state('');
	let filename = $state(suggestedName);
	let selected = $state<string | null>(null);
	let error = $state<string | null>(null);
	let firstInput = $state<HTMLInputElement | null>(null);

	async function go(path?: string | null): Promise<void> {
		error = null;
		selected = null;
		try {
			const res = await g.listDir(path ?? undefined);
			cwd = res.path;
			pathDraft = res.path;
			parent = res.parent;
			entries = res.entries;
			roots = res.roots;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function openEntry(entry: FsEntry): void {
		if (entry.kind === 'dir') {
			void go(entry.path);
		} else if (entry.is_gfi) {
			if (mode === 'load') onPick(entry.path);
			else filename = entry.name.replace(/\.gfi$/, '');
		}
	}

	function clickEntry(entry: FsEntry): void {
		if (entry.kind === 'dir') return; // single click on a dir just highlights
		if (entry.is_gfi) {
			selected = entry.path;
			if (mode === 'save') filename = entry.name.replace(/\.gfi$/, '');
		}
	}

	function confirmSave(): void {
		const name = filename.trim();
		if (!name) return;
		const dir = cwd.replace(/\/+$/, '');
		const full = `${dir}/${name.endsWith('.gfi') ? name : name + '.gfi'}`;
		onPick(full);
	}

	function confirmOpen(): void {
		if (selected) onPick(selected);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			onClose();
		}
	}

	onMount(() => {
		void go(initialPath);
		tick().then(() => firstInput?.focus());
	});

	const visible = $derived(entries.filter((e) => !e.hidden));
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fs-backdrop" onclick={onClose} role="presentation"></div>
<div
	class="fs-modal"
	role="dialog"
	aria-label={mode === 'save' ? 'Save patch' : 'Load patch'}
	data-testid="fs-browser"
>
	<header>
		<span class="title">{mode === 'save' ? 'Save patch' : 'Load patch'}</span>
		<button class="x" onclick={onClose} aria-label="Close">✕</button>
	</header>

	<div class="body">
		<nav class="roots">
			{#each roots as r (r.path)}
				<button class="root" class:active={cwd === r.path} onclick={() => go(r.path)}>{r.label}</button>
			{/each}
		</nav>

		<section class="files">
			<div class="pathbar">
				<button class="up" disabled={!parent} onclick={() => go(parent)} title="Up one level">↑</button>
				<input
					bind:this={firstInput}
					bind:value={pathDraft}
					onkeydown={(e) => {
						if (e.key === 'Enter') void go(pathDraft);
					}}
					spellcheck="false"
					autocomplete="off"
					data-testid="fs-path-input"
				/>
			</div>

			{#if error}
				<div class="err" data-testid="fs-error">{error}</div>
			{/if}

			<ul class="list" data-testid="fs-list">
				{#each visible as entry (entry.path)}
					<li>
						<button
							class="entry"
							class:dir={entry.kind === 'dir'}
							class:gfi={entry.is_gfi}
							class:sel={selected === entry.path}
							onclick={() => clickEntry(entry)}
							ondblclick={() => openEntry(entry)}
							data-testid="fs-entry"
						>
							<span class="ico">{entry.kind === 'dir' ? '📁' : entry.is_gfi ? '◆' : '·'}</span>
							<span class="nm">{entry.name}</span>
						</button>
					</li>
				{/each}
				{#if visible.length === 0}
					<li class="empty">Empty folder.</li>
				{/if}
			</ul>
		</section>
	</div>

	<footer>
		{#if mode === 'save'}
			<input class="fname" bind:value={filename} placeholder="patch name" data-testid="fs-filename" />
			<span class="ext">.gfi</span>
			<div class="spacer"></div>
			<button class="ghost" onclick={onClose}>Cancel</button>
			<button class="primary" onclick={confirmSave} data-testid="fs-save">Save</button>
		{:else}
			{#if onUpload}
				<button class="ghost" onclick={onUpload} data-testid="fs-upload">Upload from this computer…</button>
			{/if}
			<div class="spacer"></div>
			<button class="ghost" onclick={onClose}>Cancel</button>
			<button class="primary" disabled={!selected} onclick={confirmOpen} data-testid="fs-open">Open</button>
		{/if}
	</footer>
</div>

<style>
	.fs-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.45);
		z-index: 1000;
	}
	.fs-modal {
		position: fixed;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: min(720px, 92vw);
		height: min(560px, 86vh);
		display: flex;
		flex-direction: column;
		background: var(--bg-elev-1);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-2);
		z-index: 1001;
		font-size: 12px;
		overflow: hidden;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		font-weight: 600;
	}
	header .x {
		background: transparent;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		font-size: 13px;
	}
	.body {
		flex: 1;
		display: flex;
		min-height: 0;
	}
	.roots {
		flex: 0 0 140px;
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		padding: 8px 6px;
		gap: 2px;
	}
	.root {
		background: transparent;
		border: none;
		color: var(--text);
		text-align: left;
		padding: 6px 8px;
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.root.active,
	.root:hover {
		background: color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.files {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.pathbar {
		display: flex;
		gap: 6px;
		padding: 8px;
		border-bottom: 1px solid var(--border);
	}
	.pathbar .up {
		flex: 0 0 auto;
		background: var(--bg-elev-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
		padding: 0 8px;
	}
	.pathbar input {
		flex: 1;
		min-width: 0;
		background: var(--bg-elev-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		padding: 4px 8px;
		font-family: var(--font-mono);
	}
	.err {
		color: var(--warning);
		padding: 6px 10px;
		font-family: var(--font-mono);
	}
	.list {
		flex: 1;
		overflow-y: auto;
		list-style: none;
		margin: 0;
		padding: 4px 0;
	}
	.entry {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		text-align: left;
		background: transparent;
		border: none;
		color: var(--text);
		padding: 5px 12px;
		cursor: pointer;
		font-family: var(--font-mono);
	}
	.entry:hover {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.entry.sel {
		background: color-mix(in srgb, var(--accent) 18%, transparent);
	}
	.entry .ico {
		flex: 0 0 auto;
		width: 14px;
		text-align: center;
	}
	.entry.gfi .ico {
		color: var(--accent);
	}
	.empty {
		color: var(--text-faint);
		padding: 12px;
		text-align: center;
		list-style: none;
	}
	footer {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 12px;
		border-top: 1px solid var(--border);
	}
	footer .spacer {
		flex: 1;
	}
	.fname {
		background: var(--bg-elev-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		padding: 4px 8px;
		font-family: var(--font-mono);
	}
	.ext {
		color: var(--text-faint);
		font-family: var(--font-mono);
	}
	button.ghost,
	button.primary {
		border-radius: var(--radius-sm);
		padding: 5px 12px;
		cursor: pointer;
		font-size: 12px;
	}
	button.ghost {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text);
	}
	button.primary {
		background: var(--accent);
		border: 1px solid var(--accent);
		color: var(--bg-elev-1);
	}
	button.primary:disabled {
		opacity: 0.5;
		cursor: default;
	}
</style>
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npm run check`
Expected: PASS (0 errors). CSS variables (`--bg-elev-1`, `--accent`, `--radius-md`, `--shadow-2`, etc.) are defined in `frontend/src/app.css` — reuse them; do not hardcode colors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/fs/FsBrowser.svelte
git commit -m "feat(frontend): FsBrowser modal (backend save/load file picker)"
```

---

## Task 6: TopBar split-button + AppShell wiring

**Files:**
- Modify: `frontend/src/lib/editor/TopBar.svelte`
- Modify: `frontend/src/lib/app/AppShell.svelte`

**Interfaces:**
- Consumes: `FsBrowser` (Task 5); `graph().save/load/serialize/listDir` (Tasks 2–4).
- TopBar new props: `onSave`, `onSaveAs`, `onSaveInBrowser`, `onLoad`, plus existing `onAddNode`/`onFitView`/`tabs`.

- [ ] **Step 1: Rewrite TopBar's actions with a Save split-button**

Replace the `<script>` Props type + destructure (TopBar.svelte:5-15) with:

```ts
	type Props = {
		onAddNode: () => void;
		onSave: () => void;
		onSaveAs: () => void;
		onSaveInBrowser: () => void;
		onLoad: () => void;
		onFitView: () => void;
		tabs?: Snippet;
	};

	const { onAddNode, onSave, onSaveAs, onSaveInBrowser, onLoad, onFitView, tabs }: Props =
		$props();

	const g = graph();
	let saveMenuOpen = $state(false);

	function pick(fn: () => void): void {
		saveMenuOpen = false;
		fn();
	}
```

Replace the `.actions` block (TopBar.svelte:40-45) with:

```svelte
	<div class="actions">
		<button class="ghost" data-testid="topbar-add" onclick={onAddNode}>＋ Add node</button>
		<button class="ghost" data-testid="topbar-fit" onclick={onFitView}>Fit</button>
		<div class="split">
			<button class="ghost main" data-testid="topbar-save" onclick={onSave}>Save</button>
			<button
				class="ghost caret"
				data-testid="topbar-save-caret"
				aria-label="Save options"
				onclick={() => (saveMenuOpen = !saveMenuOpen)}>▾</button
			>
			{#if saveMenuOpen}
				<div class="menu" data-testid="topbar-save-menu">
					<button onclick={() => pick(onSaveAs)} data-testid="topbar-save-as">Save As…</button>
					<button onclick={() => pick(onSaveInBrowser)} data-testid="topbar-save-browser"
						>Save in browser</button
					>
				</div>
			{/if}
		</div>
		<button class="ghost" data-testid="topbar-load" onclick={onLoad}>Load…</button>
	</div>
```

Add to TopBar's `<style>`:

```css
	.split {
		position: relative;
		display: flex;
		align-items: center;
	}
	.split .main {
		border-top-right-radius: 0;
		border-bottom-right-radius: 0;
	}
	.split .caret {
		padding: 0 5px;
		border-top-left-radius: 0;
		border-bottom-left-radius: 0;
	}
	.split .menu {
		position: absolute;
		top: calc(100% + 4px);
		right: 0;
		background: var(--bg-elev-1);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		display: flex;
		flex-direction: column;
		min-width: 160px;
		z-index: 50;
	}
	.split .menu button {
		background: transparent;
		border: none;
		color: var(--text);
		text-align: left;
		padding: 8px 12px;
		cursor: pointer;
		font-size: 12px;
	}
	.split .menu button:hover {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
```

- [ ] **Step 2: Rewrite AppShell save/load handlers + mount FsBrowser**

In `frontend/src/lib/app/AppShell.svelte`, add the import (after line 13):

```ts
	import FsBrowser from '$lib/fs/FsBrowser.svelte';
```

Replace `triggerSave` and `triggerLoad` (AppShell.svelte:46-76) with:

```ts
	// Backend file browser state — null = closed.
	let fsMode = $state<null | 'save' | 'load'>(null);

	function dirOf(p: string | null): string | null {
		if (!p) return null;
		const i = p.lastIndexOf('/');
		return i > 0 ? p.slice(0, i) : null;
	}

	async function saveBackend(path?: string): Promise<void> {
		const { path: saved } = await g.save(path ?? g.savePath ?? undefined, true, ws.serialize());
		g.savePath = saved; // backend also broadcasts save_path_changed; set now for immediacy
	}

	// Default Save: silent overwrite when the patch is named, else "Save As".
	function triggerSave(): void {
		if (g.savePath) {
			void saveBackend().catch((e) => console.error('save failed', e));
		} else {
			fsMode = 'save';
		}
	}

	function saveAs(): void {
		fsMode = 'save';
	}

	// "Save in browser": download the patch YAML to the user's computer; no backend write.
	async function saveInBrowser(): Promise<void> {
		try {
			const { yaml } = await g.serialize();
			const blob = new Blob([yaml], { type: 'application/x-yaml' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			const base = g.savePath ? (g.savePath.split('/').pop() ?? '') : '';
			a.href = url;
			a.download = base || `${(window.prompt('Name this patch', 'patch') ?? 'patch').replace(/\.gfi$/, '')}.gfi`;
			a.click();
			setTimeout(() => URL.revokeObjectURL(url), 1000);
		} catch (e) {
			console.error('browser save failed', e);
		}
	}

	function triggerLoad(): void {
		fsMode = 'load';
	}

	// Secondary load path: upload a .gfi from the frontend computer.
	function uploadLoad(): void {
		fsMode = null;
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.gfi,.yaml,.yml';
		input.onchange = async () => {
			const f = input.files?.[0];
			if (!f) return;
			try {
				await g.loadText(await f.text());
			} catch (e) {
				console.error('load failed', e);
			}
		};
		input.click();
	}

	async function onFsPick(pickedPath: string): Promise<void> {
		const mode = fsMode;
		fsMode = null;
		try {
			if (mode === 'save') await saveBackend(pickedPath);
			else if (mode === 'load') await g.load(pickedPath);
		} catch (e) {
			console.error(`${mode} failed`, e);
		}
	}
```

Update the `<TopBar ...>` props (AppShell.svelte:129-134) to:

```svelte
	<TopBar
		onAddNode={addNode}
		onFitView={fitView}
		onSave={triggerSave}
		onSaveAs={saveAs}
		onSaveInBrowser={saveInBrowser}
		onLoad={triggerLoad}
	>
```

Add the FsBrowser mount just before the closing `</div>` of `.app-root` (after the `.main` div, AppShell.svelte:142):

```svelte
	{#if fsMode}
		<FsBrowser
			mode={fsMode}
			initialPath={dirOf(g.savePath)}
			suggestedName={g.savePath ? (g.savePath.split('/').pop() ?? '').replace(/\.gfi$/, '') : ''}
			onPick={onFsPick}
			onClose={() => (fsMode = null)}
			onUpload={uploadLoad}
		/>
	{/if}
```

The existing `onKeydown` already calls `triggerSave`/`triggerLoad` (Ctrl+S/Ctrl+O) — no change needed.

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npm run check`
Expected: PASS (0 errors).

- [ ] **Step 4: Build the SPA (needed for e2e)**

Run: `cd frontend && npm run build`
Expected: build completes, `frontend/build/index.html` exists.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/editor/TopBar.svelte frontend/src/lib/app/AppShell.svelte
git commit -m "feat(frontend): backend-default Save with Save-As browser + Save-in-browser dropdown"
```

---

## Task 7: End-to-end test + screenshots

**Files:**
- Create: `e2e/test_save_load.py`

**Interfaces:**
- Consumes: the `page` + `shots` fixtures and `open_menu_and_pick` helper from `e2e/conftest.py`.

- [ ] **Step 1: Write the e2e test**

```python
# e2e/test_save_load.py
"""E2E: backend-default save, Save-As file browser, browser download, load."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page

from .conftest import assert_no_console_errors, open_menu_and_pick


def test_save_as_writes_backend_file_without_download(page: Page, tmp_path: Path, shots: Path):
    open_menu_and_pick(page, "Oscillator")

    downloads: list = []
    page.on("download", lambda d: downloads.append(d))

    # Unnamed patch → clicking Save opens the FsBrowser in save mode.
    page.click('[data-testid="topbar-save"]')
    page.wait_for_selector('[data-testid="fs-browser"]', timeout=3000)
    page.screenshot(path=str(shots / "save-as-browser.png"))

    # Point it at the temp dir and name the patch.
    page.fill('[data-testid="fs-path-input"]', str(tmp_path))
    page.keyboard.press("Enter")
    page.fill('[data-testid="fs-filename"]', "run1")
    page.click('[data-testid="fs-save"]')

    # Backend wrote the file; nothing downloaded.
    saved = tmp_path / "run1.gfi"
    for _ in range(100):
        if saved.exists():
            break
        page.wait_for_timeout(50)
    assert saved.exists(), "backend did not write the patch file"
    assert downloads == [], "backend save must not trigger a browser download"

    # TopBar now shows the saved filename.
    page.wait_for_selector('text=run1.gfi', timeout=3000)
    assert_no_console_errors(page)


def test_named_save_is_silent(page: Page, tmp_path: Path):
    open_menu_and_pick(page, "Oscillator")
    # First save via the browser to name it.
    page.click('[data-testid="topbar-save"]')
    page.wait_for_selector('[data-testid="fs-browser"]')
    page.fill('[data-testid="fs-path-input"]', str(tmp_path))
    page.keyboard.press("Enter")
    page.fill('[data-testid="fs-filename"]', "run2")
    page.click('[data-testid="fs-save"]')
    saved = tmp_path / "run2.gfi"
    for _ in range(100):
        if saved.exists():
            break
        page.wait_for_timeout(50)
    assert saved.exists()
    first_mtime = saved.stat().st_mtime_ns

    # A second Save on a named patch must NOT open the browser; it overwrites silently.
    page.wait_for_timeout(50)
    page.click('[data-testid="topbar-save"]')
    page.wait_for_timeout(300)
    assert page.locator('[data-testid="fs-browser"]').count() == 0
    for _ in range(100):
        if saved.stat().st_mtime_ns != first_mtime:
            break
        page.wait_for_timeout(50)
    assert saved.stat().st_mtime_ns != first_mtime, "silent re-save did not rewrite the file"
    assert_no_console_errors(page)


def test_save_in_browser_downloads(page: Page, shots: Path):
    open_menu_and_pick(page, "Oscillator")
    page.click('[data-testid="topbar-save-caret"]')
    page.wait_for_selector('[data-testid="topbar-save-menu"]', timeout=2000)
    page.screenshot(path=str(shots / "save-dropdown.png"))
    with page.expect_download() as dl:
        page.click('[data-testid="topbar-save-browser"]')
    assert dl.value.suggested_filename.endswith(".gfi")
    assert_no_console_errors(page)


def test_load_via_browser_restores_graph(page: Page, tmp_path: Path):
    # Build + save a 2-node patch.
    open_menu_and_pick(page, "Oscillator")
    open_menu_and_pick(page, "Buffer")
    page.click('[data-testid="topbar-save"]')
    page.wait_for_selector('[data-testid="fs-browser"]')
    page.fill('[data-testid="fs-path-input"]', str(tmp_path))
    page.keyboard.press("Enter")
    page.fill('[data-testid="fs-filename"]', "two")
    page.click('[data-testid="fs-save"]')
    saved = tmp_path / "two.gfi"
    for _ in range(100):
        if saved.exists():
            break
        page.wait_for_timeout(50)
    assert saved.exists()

    before = page.locator(".svelte-flow__node").count()
    assert before == 2

    # Open the load browser, select the file, Open → graph replaced.
    page.click('[data-testid="topbar-load"]')
    page.wait_for_selector('[data-testid="fs-browser"]')
    page.fill('[data-testid="fs-path-input"]', str(tmp_path))
    page.keyboard.press("Enter")
    page.locator('[data-testid="fs-entry"]', has_text="two.gfi").first.click()
    page.click('[data-testid="fs-open"]')
    page.wait_for_function("document.querySelectorAll('.svelte-flow__node').length === 2", timeout=8000)
    assert_no_console_errors(page)
```

- [ ] **Step 2: Run the e2e suite**

Run: `.venv/bin/python -m pytest e2e/test_save_load.py -p no:cacheprovider -q`
Expected: PASS (4 passed). (Requires `frontend/build` from Task 6 Step 4 and Playwright Chromium installed — `.venv/bin/python -m playwright install chromium` if missing.)

- [ ] **Step 3: Review screenshots**

Open `e2e/screenshots/save-as-browser.png` and `e2e/screenshots/save-dropdown.png`. Verify: the modal is centered above the canvas with no clipping, the dropdown sits below the Save button without overlap, text is legible. (Spec validation: pay attention to z-layering.)

- [ ] **Step 4: Commit**

```bash
git add e2e/test_save_load.py
git commit -m "test(e2e): save-as, silent save, browser download, load round-trip"
```

---

## Self-Review

**Spec coverage (Phase-1 rows of the design spec §Phase 1 + decision-log):**
- Backend save by default → Task 6 `triggerSave`/`saveBackend`. ✓
- Save-As in-app full-FS browser + name field → Tasks 1,5,6. ✓
- Save dropdown: Save / Save As… / Save in browser → Task 6 TopBar. ✓
- "Save in browser" download without backend write → Tasks 2 (`serialize_patch`), 3 (`serialize` op), 6 (`saveInBrowser`). ✓
- Load via full-FS browser + Examples menu + upload secondary → Tasks 1 (`list_examples` + Examples root), 5 (load mode + upload button), 6 (`triggerLoad`/`uploadLoad`). ✓
- Fix "downloads untitled" bug → Task 6 (Save no longer Blob-downloads; passes a real path). ✓
- No FS jail, bind stays 0.0.0.0 → Task 1 (no jail), nothing touches the bind. ✓
- Unsaved indicator / save_path wiring → unchanged backend broadcasts (`manager.py` save_path_changed / unsaved_changes) already drive the store; Task 6 also sets `g.savePath` from the return. ✓
- **Deferred to Phase 2 (intentional, flagged):** v2 envelope, one-shot converter, `member_uid`/`membership`, `flat_view` guard. Not in any Phase-1 task by design.

**Placeholder scan:** none — every step has full code/commands.

**Type consistency:** `DirListing`/`FsEntry`/`FsRoot` defined in Task 4, consumed identically in Tasks 5–6; store methods `listDir`/`listExamples`/`load`/`serialize` defined Task 4, called in Tasks 5–6; `serialize_patch` defined Task 2, called Task 3; control ops `list_dir`/`list_examples`/`serialize` defined Task 3, called via store Task 4. TopBar props `onSave`/`onSaveAs`/`onSaveInBrowser`/`onLoad` defined Task 6 and supplied by AppShell same task. ✓

**Note for the implementer:** after all tasks, run the full backend suite once — `.venv/bin/python -m pytest tests/ -p no:cacheprovider -q` — to confirm no regression (goal-condition: the existing Python tests still pass). The `test_multiproc_create` test is known-flaky on fork+iceoryx2; re-run it alone if it fails before treating it as a real regression.
