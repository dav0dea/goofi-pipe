<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import type { Snippet } from 'svelte';
	import type { MenuItem } from '$lib/workspace/menu';
	import ContextMenu from '$lib/workspace/ContextMenu.svelte';
	import PerfHud from './PerfHud.svelte';
	import { Button, IconButton, Badge } from '$lib/ui';

	type Props = {
		onAddNode: () => void;
		onSave: () => void;
		onSaveAs: () => void;
		onSaveInBrowser: () => void;
		onLoad: () => void;
		onFitView: () => void;
		/** Workspace tab strip, rendered in the header's central gap between the
		 * filename and the action buttons. */
		tabs?: Snippet;
	};

	const { onAddNode, onSave, onSaveAs, onSaveInBrowser, onLoad, onFitView, tabs }: Props = $props();

	const g = graph();
	const h = history();

	// Save split-button dropdown — opened via the shared ContextMenu, which
	// portals to <body> at --z-menu so it stacks above side panels.
	let saveMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	function openSaveMenu(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		saveMenu = {
			x: Math.max(6, r.right - 180),
			y: r.bottom + 4,
			items: [
				{ label: 'Save As…', action: onSaveAs },
				{ label: 'Save in browser', action: onSaveInBrowser }
			]
		};
	}
</script>

<div class="topbar">
	<div class="brand">
		<span class="logo">⟁</span>
		<span class="name">goofi-pipe</span>
		<Badge tone={g.connected ? 'success' : 'warning'}>
			{g.connected ? 'connected' : 'connecting…'}
		</Badge>
		<PerfHud />
		{#if g.savePath}
			<span class="path" title={g.savePath}
				>{g.unsavedChanges ? '● ' : ''}{g.savePath.split('/').pop()}</span
			>
		{:else if g.unsavedChanges}
			<span class="path">● untitled</span>
		{/if}
	</div>

	{#if tabs}
		<div class="tabslot">{@render tabs()}</div>
	{/if}

	<div class="actions">
		<IconButton
			variant="ghost"
			data-testid="topbar-undo"
			disabled={!h.canUndo}
			title={h.undoLabel ? `Undo ${h.undoLabel}` : 'Nothing to undo'}
			label="Undo"
			onclick={() => void h.undo()}>↶</IconButton
		>
		<IconButton
			variant="ghost"
			data-testid="topbar-redo"
			disabled={!h.canRedo}
			title={h.redoLabel ? `Redo ${h.redoLabel}` : 'Nothing to redo'}
			label="Redo"
			onclick={() => void h.redo()}>↷</IconButton
		>
		<Button variant="ghost" data-testid="topbar-add" onclick={onAddNode}>＋ Add node</Button>
		<Button variant="ghost" data-testid="topbar-fit" onclick={onFitView}>Fit</Button>
		<div class="split">
			<Button variant="ghost" class="seg-main" data-testid="topbar-save" onclick={onSave}
				>Save</Button
			>
			<IconButton
				variant="ghost"
				class="seg-caret"
				data-testid="topbar-save-caret"
				label="Save options"
				onclick={openSaveMenu}>▾</IconButton
			>
		</div>
		<Button variant="ghost" data-testid="topbar-load" onclick={onLoad}>Load…</Button>
	</div>
</div>

{#if saveMenu}
	<ContextMenu
		x={saveMenu.x}
		y={saveMenu.y}
		items={saveMenu.items}
		onClose={() => (saveMenu = null)}
	/>
{/if}

<style>
	.topbar {
		display: flex;
		align-items: center;
		gap: var(--space-7);
		padding: 0 var(--space-6);
		background: var(--surface-1);
		border-bottom: 1px solid var(--border);
		/* Frozen: 44px is the coarse --hit floor, so the bar already fits a touch-sized
		   control without growing — a rem height would break that flush relationship. */
		height: 44px;
		font-size: var(--fs-body);
		z-index: 10;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: var(--space-6);
		flex: 0 0 auto;
	}
	/* The tab strip fills the slack between the filename and the actions. */
	.tabslot {
		flex: 1 1 auto;
		min-width: 0;
		align-self: stretch;
		display: flex;
		align-items: stretch;
	}
	.logo {
		font-size: var(--fs-title);
		color: var(--text);
		font-weight: 600;
	}
	.name {
		font-weight: 600;
	}
	.path {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.actions {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		flex: 0 0 auto;
	}
	/* Save split — two adjacent Buttons sharing a seam: the touching corners are
	   squared so the main action + its caret read as one segmented control. */
	.split {
		position: relative;
		display: inline-flex;
		align-items: stretch;
	}
	.split :global(.seg-main) {
		border-top-right-radius: 0;
		border-bottom-right-radius: 0;
	}
	.split :global(.seg-caret) {
		border-top-left-radius: 0;
		border-bottom-left-radius: 0;
	}
</style>
