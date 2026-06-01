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
	import ErrorPanel from '$lib/editor/ErrorPanel.svelte';
	import WorkspaceTabs from '$lib/workspace/WorkspaceTabs.svelte';
	import WorkspaceView from '$lib/workspace/Workspace.svelte';
	import AutoSidePanel from './AutoSidePanel.svelte';
	import { registerBuiltinPanels } from '$lib/workspace/panels';
	import { registerAppPanels } from '$lib/panels/register';
	import { editorFor } from '$lib/panels/editorCommands';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { onMount } from 'svelte';

	// Populate the panel registry before any panel renders.
	registerBuiltinPanels();
	registerAppPanels();

	const g = graph();
	const ws = workspace();
	const sel = selection();

	let sidePanelEnabled = $state(readInspectorPref());

	function readInspectorPref(): boolean {
		try {
			return localStorage.getItem('goofi.inspectorOn') !== '0';
		} catch {
			return true;
		}
	}
	function toggleSidePanel(): void {
		sidePanelEnabled = !sidePanelEnabled;
		try {
			localStorage.setItem('goofi.inspectorOn', sidePanelEnabled ? '1' : '0');
		} catch {
			/* best-effort */
		}
	}

	// TopBar "Add node" / "Fit" drive whichever editor panel is active.
	function addNode(): void {
		editorFor(ws.activePanelId)?.openAddMenu();
	}
	function fitView(): void {
		editorFor(ws.activePanelId)?.fitView();
	}

	async function triggerSave(): Promise<void> {
		try {
			const { yaml, path } = await g.save(undefined, true);
			const blob = new Blob([yaml], { type: 'application/x-yaml' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = path.split('/').pop() ?? 'patch.gfi';
			a.click();
			setTimeout(() => URL.revokeObjectURL(url), 1000);
		} catch (e) {
			console.error('save failed', e);
		}
	}

	function triggerLoad(): void {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.gfi,.yaml,.yml';
		input.onchange = async () => {
			const f = input.files?.[0];
			if (!f) return;
			const content = await f.text();
			try {
				await g.loadText(content);
			} catch (e) {
				console.error('load failed', e);
			}
		};
		input.click();
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
		if (!g.unsavedChanges) return;
		e.preventDefault();
		e.returnValue = '';
	}

	onMount(() => {
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('beforeunload', onBeforeUnload);
		return () => {
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('beforeunload', onBeforeUnload);
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
		onLoad={triggerLoad}
		onToggleSidePanel={toggleSidePanel}
		sidePanelOn={sidePanelEnabled}
	/>
	<WorkspaceTabs />
	<div class="main">
		<WorkspaceView />
		<AutoSidePanel enabled={sidePanelEnabled} />
		<ErrorPanel mode="chip" onFocus={(name) => sel.selectNodes([name])} />
	</div>
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
