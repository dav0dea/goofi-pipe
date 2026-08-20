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

	// A protocol-version mismatch means this built SPA can't safely talk to the
	// running backend (a stale frontend/build/ against a newer manager). Prompt a
	// hard reload rather than letting the event-reconciled UI diverge silently.
	let protocolMismatch = $state(false);

	// Populate the panel registry before any panel renders, and give the panel system the host its
	// gestures go through — until one is installed it draws and refuses. The pre-sync frame is the
	// MANAGER's own first-mint spelling, so what is on screen before the snapshot lands and what is
	// on screen after it are the same panel, and the editor mounts once.
	registerAppPanels();
	workspace().configureHost(layoutHost(), [
		{ id: 'tab-1', name: 'Tab 1', root: { kind: 'panel', id: 'panel-2', panelType: DEFAULT_PANEL_TYPE } }
	]);
	// Publish window.goofi so the agent panel / Playwright can drive the app.
	exposeAgentApi();

	const g = graph();
	const ws = workspace();

	// Focus an errored node in the editor the user last touched (keyed by uid).
	function focusError(uid: string): void {
		editorFor(ws.activePanelId)?.focusNode(uid);
	}

	// Backend file browser state — null = closed.
	let fsMode = $state<null | 'save' | 'load'>(null);

	function dirOf(p: string | null): string | null {
		if (!p) return null;
		const i = p.lastIndexOf('/');
		return i > 0 ? p.slice(0, i) : null;
	}

	// Default Save: silent overwrite when the patch is named, else "Save As". The name comes from
	// the MANAGER now (the snapshot and `save_path_changed`), so it survives a reload and reaches
	// every open tab — which is exactly what makes the failure below reachable.
	function triggerSave(): void {
		const path = g.savePath;
		if (path) {
			// A silent overwrite onto a remembered path is the one save with no dialog in front of
			// it, so a rejection here has no other surface: the file may have been deleted, moved
			// or made read-only since the manager learned the path. It used to be a console.error.
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
			// The browser has already closed by here, so the rejection has nowhere else to land.
			notify().failure(mode === 'save' ? 'Save' : 'Load', e);
		}
	}

	/** A `.gfi` the user picked on their OWN machine, for locations the backend's browser cannot
	 *  reach — in a container, anything that was not bind-mounted. The modal closes first: the
	 *  upload replaces the whole patch, so leaving a file list from the outgoing one on screen
	 *  reads as if nothing happened. Failure notifies here for the same reason `onFsPick` does —
	 *  the browser is already gone, so a rejection has nowhere else to land. */
	async function onFsFilePick(file: File): Promise<void> {
		fsMode = null;
		try {
			await uploadPatch(file);
		} catch (e) {
			notify().failure('Open', e);
		}
	}

	function onKeydown(e: KeyboardEvent): void {
		// A modal (the file browser) or an in-panel expression editor owns the keyboard while it is
		// up, so EVERY app-global chord stands down — not just undo/redo. Ctrl+S used to re-enter
		// triggerSave() with the browser already open, flipping a Load browser into Save mode (and,
		// on a named patch, writing the file behind it).
		const standdown = ui().modalOpen;
		const meta = e.ctrlKey || e.metaKey;
		const key = e.key.toLowerCase();
		if (meta && (key === 's' || key === 'o')) {
			// The chord is ALWAYS the app's, standdown or not: claiming it is what keeps Chrome's own
			// "Save page as…" / "Open file…" off the screen. Standing down means not acting on it.
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
		// Flush a pending viewpoint push so a quick reload lands where we left off.
		if (pushTimer) {
			clearTimeout(pushTimer);
			pushTimer = null;
			void g.setViewpoint(ws.viewpoint());
		}
		if (!g.unsavedChanges) return;
		e.preventDefault();
		e.returnValue = '';
	}

	// The ARRANGEMENT is not pushed from here any more — every gesture is a layout command, and the
	// manager holds it. What is left is the VIEWPOINT: which page is in front, which panel is
	// focused, how deep each editor has navigated. It is this client's alone, so it is stored
	// (debounced — tab-hopping collapses into one push) and never converged, and it cannot dirty
	// the patch. Only after the initial sync, so a fresh client does not overwrite the stored
	// viewpoint with its own default before that viewpoint has arrived.
	let pushTimer: ReturnType<typeof setTimeout> | null = null;

	$effect(() => {
		void ws.viewpointEpoch; // track: bumped by every viewpoint change
		if (!g.hadHello) return;
		if (pushTimer) clearTimeout(pushTimer);
		// Cleared as it fires, so `onBeforeUnload` flushes only a push that is genuinely pending.
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
			<!-- The banner IS the danger surface, so the action inside it takes the neutral variant:
			     a `danger` fill on a --danger banner would be the same colour twice. -->
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
	<!-- The detach-or-kill question. Shell chrome because it is about an INSTANCE, not a view: both
	     doors onto it (a panel header's ✕ and the TopBar's badge) raise the same one, and neither
	     has to open a panel — asking must not dirty the patch. -->
	<AgentClose />
	<!-- The coarse-pointer door onto every `title=` in the app. One layer, mounted once, so a
	     tooltip anywhere below is reachable without hover. -->
	<TitleTip />
	{#if g.disconnected}
		<!-- The loud half of the alarm (the chip in the TopBar is the quiet half): the WHOLE window
		     wears the warning ring, because a lost backend is the app's problem, not the header's.
		     A fixed, pointer-transparent overlay rather than an outline on any one element — panels
		     establish their own stacking contexts and would paint over an inset ring on the shell,
		     and an overlay can sit above them all without costing a pixel of layout or a single
		     pointer event. -->
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
		/* The safe area belongs on the shell, not on `body`: this box is laid out against the
		   initial containing block, so an ancestor's padding cannot reach it (pinned in
		   device-stamp.spec.ts). Padding rather than `inset`, so the app still paints edge to edge
		   under a notch and only its CHROME steps clear. Zero on a desktop. */
		padding: var(--safe-top) var(--safe-right) var(--safe-bottom) var(--safe-left);
	}
	/* The disconnection ring. `inset` box-shadow so it draws INWARD from the viewport edges —
	   an outline would land outside the fixed box and clip to nothing. --z-toast: the alarm and a
	   toast are the two things allowed above every panel, and the frame takes no events, so the
	   toast (later in the DOM) still wins where they overlap. */
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
