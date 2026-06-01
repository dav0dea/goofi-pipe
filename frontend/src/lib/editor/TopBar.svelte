<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';

	type Props = {
		onAddNode: () => void;
		onSave: () => void;
		onLoad: () => void;
		onFitView: () => void;
		onToggleSidePanel: () => void;
		sidePanelOn: boolean;
	};

	const { onAddNode, onSave, onLoad, onFitView, onToggleSidePanel, sidePanelOn }: Props = $props();

	const g = graph();
	let examplesOpen = $state(false);
	let examples = $state<{ name: string; size: number }[]>([]);

	async function toggleExamples(): Promise<void> {
		examplesOpen = !examplesOpen;
		if (examplesOpen && examples.length === 0) {
			try {
				examples = (await g.listExamples()).examples;
			} catch (e) {
				console.warn('list_examples failed', e);
			}
		}
	}

	async function loadExample(name: string): Promise<void> {
		examplesOpen = false;
		try {
			await g.loadExample(name);
		} catch (e) {
			console.error('load_example failed', e);
		}
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

	<div class="actions">
		<button class="ghost" data-testid="topbar-add" onclick={onAddNode}>＋ Add node</button>
		<button class="ghost" data-testid="topbar-fit" onclick={onFitView}>Fit</button>
		<button class="ghost" data-testid="topbar-save" onclick={onSave}>Save</button>
		<button class="ghost" data-testid="topbar-load" onclick={onLoad}>Load…</button>
		<button
			class="ghost"
			class:active={sidePanelOn}
			data-testid="topbar-inspector"
			title="Toggle the selection inspector overlay"
			onclick={onToggleSidePanel}>Inspector</button
		>
		<div class="examples-host">
			<button class="ghost" onclick={toggleExamples} data-testid="topbar-examples">
				Examples ▾
			</button>
			{#if examplesOpen}
				<div class="examples-pop" role="menu">
					{#each examples as ex}
						<button class="ex-item" onclick={() => loadExample(ex.name)}>{ex.name}</button>
					{/each}
					{#if examples.length === 0}
						<div class="ex-empty">no examples</div>
					{/if}
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
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
	}
	.actions button.active {
		color: var(--accent);
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.examples-host {
		position: relative;
	}
	.examples-pop {
		position: absolute;
		right: 0;
		top: 100%;
		margin-top: 4px;
		background: var(--bg-elev-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		min-width: 240px;
		max-height: 320px;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		box-shadow: var(--shadow-2);
		z-index: var(--z-menu);
	}
	.ex-item {
		background: transparent;
		border: none;
		border-radius: 0;
		text-align: left;
		padding: 6px 10px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text);
		cursor: pointer;
	}
	.ex-item:hover {
		background: var(--bg-elev-3);
	}
	.ex-empty {
		padding: 12px;
		color: var(--text-faint);
		text-align: center;
	}
</style>
