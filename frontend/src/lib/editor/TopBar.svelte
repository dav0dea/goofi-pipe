<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import type { Snippet } from 'svelte';
	import type { MenuItem } from '$lib/workspace/menu';
	import ContextMenu from '$lib/workspace/ContextMenu.svelte';
	import PerfHud from './PerfHud.svelte';

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
		<span class="status" class:online={g.connected} class:offline={!g.connected}>
			{g.connected ? 'connected' : 'connecting…'}
		</span>
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
		<button
			class="ghost icon"
			data-testid="topbar-undo"
			disabled={!h.canUndo}
			title={h.undoLabel ? `Undo ${h.undoLabel}` : 'Nothing to undo'}
			aria-label="Undo"
			onclick={() => void h.undo()}>↶</button
		>
		<button
			class="ghost icon"
			data-testid="topbar-redo"
			disabled={!h.canRedo}
			title={h.redoLabel ? `Redo ${h.redoLabel}` : 'Nothing to redo'}
			aria-label="Redo"
			onclick={() => void h.redo()}>↷</button
		>
		<button class="ghost" data-testid="topbar-add" onclick={onAddNode}>＋ Add node</button>
		<button class="ghost" data-testid="topbar-fit" onclick={onFitView}>Fit</button>
		<div class="split">
			<button class="ghost main" data-testid="topbar-save" onclick={onSave}>Save</button>
			<button
				class="ghost caret"
				data-testid="topbar-save-caret"
				aria-label="Save options"
				onclick={openSaveMenu}>▾</button
			>
		</div>
		<button class="ghost" data-testid="topbar-load" onclick={onLoad}>Load…</button>
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
		gap: 14px;
		padding: 0 12px;
		background: var(--bg-elev-1);
		border-bottom: 1px solid var(--border);
		height: 44px;
		font-size: 12px;
		z-index: 10;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
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
		font-size: 18px;
		color: var(--text);
		font-weight: 600;
	}
	.name {
		font-weight: 600;
	}
	.status {
		font-size: 10px;
		padding: 2px 6px;
		border-radius: 4px;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.status.online {
		background: color-mix(in srgb, var(--success) 20%, transparent);
		color: var(--success);
	}
	.status.offline {
		background: color-mix(in srgb, var(--warning) 20%, transparent);
		color: var(--warning);
	}
	.path {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 11px;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: 4px;
		flex: 0 0 auto;
	}
	.icon {
		font-size: 15px;
		line-height: 1;
		padding: 4px 7px;
	}
	.icon:disabled {
		opacity: 0.35;
		cursor: default;
	}
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
</style>
