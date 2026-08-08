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
	import Toast from '$lib/app/Toast.svelte';
	import TitleTip from '$lib/app/TitleTip.svelte';
	import WorkspaceTabs from '$lib/workspace/WorkspaceTabs.svelte';
	import WorkspaceView from '$lib/workspace/WorkspaceView.svelte';
	import { registerBuiltinPanels } from '$lib/workspace/panels';
	import { registerAppPanels } from '$lib/panels/register';
	import { editorFor } from '$lib/panels/editorCommands';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { history } from '$lib/stores/history.svelte';
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

	// Populate the panel registry before any panel renders.
	registerBuiltinPanels();
	registerAppPanels();
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

	// `g.save` remembers the path it wrote to — the store is where BOTH doors onto a save learn the
	// patch has a home (this one and `window.goofi.commands.save`), since the manager keeps no
	// save-path state and its `save` arm broadcasts nothing.
	async function saveBackend(path?: string): Promise<void> {
		await g.save(path ?? g.savePath ?? undefined);
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
		// Flush a pending layout push so a quick reload keeps the arrangement.
		if (pushTimer) {
			clearTimeout(pushTimer);
			pushTimer = null;
			pushLayout();
		}
		if (!g.unsavedChanges) return;
		e.preventDefault();
		e.returnValue = '';
	}

	// Push layout changes into the running patch (debounced — a resize drag or
	// rapid splits collapse into one push). Only after the initial sync, so we
	// don't echo before the patch's own layout has arrived. The push carries the
	// window's folded dirty classification (`takeLayoutIntent`), so navigating
	// persists without marking the patch unsaved.
	let pushTimer: ReturnType<typeof setTimeout> | null = null;

	// …with one exception: an arrangement a PEER authored is already what the manager holds, so
	// pushing it back is a round trip that buys nothing and would overwrite the manager's copy with
	// THIS client's navigation fields. The latch is spent on read, and any local write inside the
	// same debounce window clears it, so a real edit still gets through.
	function pushLayout(): void {
		if (ws.takeRemoteApplied()) return;
		void g.setLayout(ws.serialize(), ws.takeLayoutIntent());
	}

	$effect(() => {
		void ws.state; // track: every layout mutation replaces this reference
		if (!g.hadHello) return;
		if (pushTimer) clearTimeout(pushTimer);
		// Cleared as it fires, so `onBeforeUnload` flushes only a push that is genuinely still
		// PENDING — a spent handle used to make it re-push, which after a remote apply meant
		// echoing the peer's own arrangement back at the manager on the way out.
		pushTimer = setTimeout(() => {
			pushTimer = null;
			pushLayout();
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
	<TopBar
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
	<Toast />
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
