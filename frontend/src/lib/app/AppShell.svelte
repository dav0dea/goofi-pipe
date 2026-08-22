<script lang="ts">
	import TopBar from '$lib/editor/TopBar.svelte';
	import FsBrowser from '$lib/fs/FsBrowser.svelte';
	import { uploadPatch } from '$lib/api/patchFile';
	import ErrorPanel from '$lib/editor/ErrorPanel.svelte';
	import Toast from '$lib/app/Toast.svelte';
	import AgentClose from '$lib/app/AgentClose.svelte';
	import TitleTip from '$lib/app/TitleTip.svelte';
	import { Tabs as WorkspaceTabs } from 'panelty';
	import { Panels as WorkspaceView } from 'panelty';
	import { registerAppPanels } from '$lib/panels/register';
	import { editorFor } from '$lib/panels/editorCommands';
	import { workspace } from 'panelty';
	import { layoutHost } from '$lib/stores/layoutHost';
	import { DEFAULT_PANEL_TYPE } from '$lib/api/vocab';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { notify } from '$lib/stores/notify.svelte';
	import { undoKeyAction } from '$lib/app/undoKeys';
	import { isTextEditingTarget } from '$lib/ui';
	import { exposeAgentApi } from '$lib/agent';
	import { Button } from '$lib/ui';
	import { getControl } from '$lib/api/control';
	import { onMount } from 'svelte';

	let protocolMismatch = $state(false);

	// Before any panel renders; the pre-sync frame is the manager's own first-mint spelling, so the
	// editor mounts once.
	registerAppPanels();
	workspace().configureHost(layoutHost(), {
		kind: 'stack',
		id: 'stack-1',
		children: [{ kind: 'panel', id: 'panel-2', panelType: DEFAULT_PANEL_TYPE }]
	});
	exposeAgentApi();

	const g = graph();
	const ws = workspace();

	function focusError(uid: string): void {
		editorFor(ws.activePanelId)?.focusNode(uid);
	}

	let fsMode = $state<null | 'save' | 'load'>(null);

	function dirOf(p: string | null): string | null {
		if (!p) return null;
		const i = p.lastIndexOf('/');
		return i > 0 ? p.slice(0, i) : null;
	}

	function triggerSave(): void {
		const path = g.savePath;
		if (path) {
			// The one save with no dialog in front of it, so a rejection has no other surface.
			void g.save(path).catch((e) => notify().failure('Save', e));
		} else {
			fsMode = 'save';
		}
	}

	function saveAs(): void {
		fsMode = 'save';
	}

	function triggerLoad(): void {
		fsMode = 'load';
	}

	async function onFsPick(pickedPath: string): Promise<void> {
		const mode = fsMode;
		fsMode = null;
		try {
			if (mode === 'save') await g.save(pickedPath);
			else if (mode === 'load') await g.load(pickedPath);
		} catch (e) {
			notify().failure(mode === 'save' ? 'Save' : 'Load', e);
		}
	}

	/** Upload a `.gfi` from the user's own machine, for what the backend's browser cannot reach. */
	async function onFsFilePick(file: File): Promise<void> {
		fsMode = null;
		try {
			await uploadPatch(file);
		} catch (e) {
			notify().failure('Open', e);
		}
	}

	function onKeydown(e: KeyboardEvent): void {
		const standdown = ui().modalOpen;
		const meta = e.ctrlKey || e.metaKey;
		const key = e.key.toLowerCase();
		if (meta && (key === 's' || key === 'o')) {
			// Claimed even when standing down: that is what keeps Chrome's own Save/Open off screen.
			e.preventDefault();
			if (standdown) return;
			if (key === 's') void triggerSave();
			else triggerLoad();
			return;
		}
		const undoRedo = undoKeyAction(
			{
				key: e.key,
				ctrlKey: e.ctrlKey,
				metaKey: e.metaKey,
				shiftKey: e.shiftKey,
				editing: isTextEditingTarget(e.target)
			},
			standdown
		);
		if (undoRedo === 'undo') {
			e.preventDefault();
			void history().undo();
		} else if (undoRedo === 'redo') {
			e.preventDefault();
			void history().redo();
		}
	}

	function onBeforeUnload(e: BeforeUnloadEvent): void {
		if (pushTimer) {
			clearTimeout(pushTimer);
			pushTimer = null;
			void g.setViewpoint(ws.viewpoint());
		}
		if (!g.unsavedChanges) return;
		e.preventDefault();
		e.returnValue = '';
	}

	// The viewpoint is this client's alone: stored, never converged, and it cannot dirty the patch.
	let pushTimer: ReturnType<typeof setTimeout> | null = null;

	$effect(() => {
		void ws.viewpointEpoch; // track: bumped by every viewpoint change
		// A fresh client must not overwrite the stored viewpoint with its own default.
		if (!g.hadHello) return;
		if (pushTimer) clearTimeout(pushTimer);
		pushTimer = setTimeout(() => {
			pushTimer = null;
			void g.setViewpoint(ws.viewpoint());
		}, 400);
	});

	onMount(() => {
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('beforeunload', onBeforeUnload);
		const offProto = getControl().onProtocolMismatch(() => (protocolMismatch = true));
		return () => {
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('beforeunload', onBeforeUnload);
			offProto();
			if (pushTimer) clearTimeout(pushTimer);
		};
	});
</script>

<svelte:head>
	<title>{g.unsavedChanges ? '● ' : ''}{g.savePath ? g.savePath.split('/').pop() : 'goofi-pipe'}</title
	>
</svelte:head>

<div class="app-root">
	{#if protocolMismatch}
		<div class="proto-banner" role="alert">
			<span>This page is out of date with the backend. Reload to continue.</span>
			<Button size="sm" onclick={() => location.reload()}>Reload</Button>
		</div>
	{/if}
	<TopBar onSave={triggerSave} onSaveAs={saveAs} onLoad={triggerLoad}>
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
			onFilePick={onFsFilePick}
			onClose={() => (fsMode = null)}
		/>
	{/if}
	<Toast />
	<!-- Shell chrome, not a panel: it asks about an INSTANCE, and asking must not dirty the patch. -->
	<AgentClose />
	<!-- One layer, mounted once, so every `title=` below is reachable without hover. -->
	<TitleTip />
	{#if g.disconnected}
		<!-- An overlay, not a ring on the shell: panels make their own stacking contexts and would
		     paint over one. -->
		<div class="net-frame" data-testid="net-frame" aria-hidden="true"></div>
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
		/* On the shell, not `body`: this box is fixed, so an ancestor's padding cannot reach it. */
		padding: var(--safe-top) var(--safe-right) var(--safe-bottom) var(--safe-left);
	}
	/* An `inset` shadow because an outline would land outside the fixed box and clip to nothing. */
	.net-frame {
		position: fixed;
		inset: 0;
		pointer-events: none;
		box-shadow: inset 0 0 0 3px var(--warning);
		z-index: var(--z-toast);
	}
	.main {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
	}
	.proto-banner {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-6);
		padding: var(--space-3) var(--space-6);
		background: var(--danger);
		color: var(--on-danger);
		font-size: var(--fs-small);
	}
</style>
