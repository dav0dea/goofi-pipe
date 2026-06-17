<!--
  Application shell — the only constant chrome. A header (TopBar) over a strip
  of workspace tabs, over the customizable panel workspace. The selection
  inspector and the floating error chip overlay the workspace. App-global
  shortcuts (Ctrl+S / Ctrl+O) and the unsaved-changes guard live here; the
  per-editor shortcuts live in NodeEditorPanel.

  Panel types are registered before the workspace renders so the saved/default
  layout resolves its content immediately.
-->
<script lang="ts">
	import TopBar from '$lib/editor/TopBar.svelte';
	import FsBrowser from '$lib/fs/FsBrowser.svelte';
	import ErrorPanel from '$lib/editor/ErrorPanel.svelte';
	import WorkspaceTabs from '$lib/workspace/WorkspaceTabs.svelte';
	import WorkspaceView from '$lib/workspace/WorkspaceView.svelte';
	import { registerBuiltinPanels } from '$lib/workspace/panels';
	import { registerAppPanels } from '$lib/panels/register';
	import { editorFor } from '$lib/panels/editorCommands';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { exposeAgentApi } from '$lib/agent';
	import { onMount } from 'svelte';

	// Populate the panel registry before any panel renders.
	registerBuiltinPanels();
	registerAppPanels();
	// Publish window.goofi so the agent panel / Playwright can drive the app.
	exposeAgentApi();

	const g = graph();
	const ws = workspace();

	// TopBar "Add node" / "Fit" drive whichever editor panel is active.
	function addNode(): void {
		editorFor(ws.activePanelId)?.openAddMenu();
	}
	function fitView(): void {
		editorFor(ws.activePanelId)?.fitView();
	}

	// Focus an errored node in the editor the user last touched.
	function focusError(name: string): void {
		editorFor(ws.activePanelId)?.focusNode(name);
	}

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
			a.download =
				base || `${(window.prompt('Name this patch', 'patch') ?? 'patch').replace(/\.gfi$/, '')}.gfi`;
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

	function onKeydown(e: KeyboardEvent): void {
		const meta = e.ctrlKey || e.metaKey;
		if (meta && e.key.toLowerCase() === 's') {
			e.preventDefault();
			void triggerSave();
		} else if (meta && e.key.toLowerCase() === 'o') {
			e.preventDefault();
			triggerLoad();
		}
	}

	function onBeforeUnload(e: BeforeUnloadEvent): void {
		// Flush a pending layout push so a quick reload keeps the arrangement.
		if (pushTimer) {
			clearTimeout(pushTimer);
			pushTimer = null;
			void g.setLayout(ws.serialize());
		}
		if (!g.unsavedChanges) return;
		e.preventDefault();
		e.returnValue = '';
	}

	// Push layout changes into the running patch (debounced — a resize drag or
	// rapid splits collapse into one push). Only after the initial sync, so we
	// don't echo before the patch's own layout has arrived.
	let pushTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		void ws.state; // track: every layout mutation replaces this reference
		if (!g.hadHello) return;
		if (pushTimer) clearTimeout(pushTimer);
		pushTimer = setTimeout(() => void g.setLayout(ws.serialize()), 400);
	});

	onMount(() => {
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('beforeunload', onBeforeUnload);
		return () => {
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('beforeunload', onBeforeUnload);
			if (pushTimer) clearTimeout(pushTimer);
		};
	});
</script>

<svelte:head>
	<title>{g.unsavedChanges ? '● ' : ''}{g.savePath ? g.savePath.split('/').pop() : 'goofi-pipe'}</title
	>
</svelte:head>

<div class="app-root">
	<TopBar
		onAddNode={addNode}
		onFitView={fitView}
		onSave={triggerSave}
		onSaveAs={saveAs}
		onSaveInBrowser={saveInBrowser}
		onLoad={triggerLoad}
	>
		{#snippet tabs()}
			<WorkspaceTabs />
		{/snippet}
	</TopBar>
	<div class="main">
		<WorkspaceView />
		<ErrorPanel onFocus={focusError} />
	</div>
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
</div>

<style>
	.app-root {
		position: fixed;
		inset: 0;
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
	}
	.main {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
	}
</style>
