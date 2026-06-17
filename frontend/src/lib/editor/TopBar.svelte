<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import type { Snippet } from 'svelte';

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
	let saveMenuOpen = $state(false);

	function pick(fn: () => void): void {
		saveMenuOpen = false;
		fn();
	}
</script>

<div class="topbar">
	<div class="brand">
		<span class="logo">⟁</span>
		<span class="name">goofi-pipe</span>
		<span class="status" class:online={g.connected} class:offline={!g.connected}>
			{g.connected ? 'connected' : 'connecting…'}
		</span>
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
</div>

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
		background: linear-gradient(120deg, var(--accent), var(--cat-array));
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
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
</style>
