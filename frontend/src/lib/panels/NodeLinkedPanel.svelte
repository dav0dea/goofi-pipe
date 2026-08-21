<!-- Shared chrome for the node-linked panels (Parameters / Viewer / Metadata). -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { linkedNodeName } from 'panelty';
	import { workspace } from 'panelty';
	import { Bar, Icon, IconButton, StatusDot, EmptyState } from '$lib/ui';
	import { nodeHealth } from '$lib/editor/nodeHealth';
	import NodeSelect from './NodeSelect.svelte';
	import type { Snippet } from 'svelte';

	let {
		panelId,
		state: linkState,
		label,
		content,
		controls
	}: PanelProps & {
		label: string;
		content: Snippet<[NodeInstanceInfo]>;
		/** Panel-specific controls rendered inline in the header bar. */
		controls?: Snippet<[NodeInstanceInfo]>;
	} = $props();

	const g = graph();
	const uiStore = ui();
	const ws = workspace();

	const linkedName = $derived(linkedNodeName(linkState));
	const node = $derived(linkedName ? g.nodeById(linkedName) : null);
	const dragActive = $derived(uiStore.nodeDrag !== null);
	const over = $derived(uiStore.nodeDragTarget === panelId);

	// Through the store, not the opaque `setState`: an unrecorded edit is destroyed by the next undo.
	function unlink(): void {
		ws.unlinkNodeFromPanel(panelId);
	}
</script>

<div class="linked" role="group" data-testid="node-linked-panel">
	<Bar>
		{#snippet start()}
			{#if node}
				{@const health = nodeHealth(node)}
				<StatusDot
					tone={health.tone}
					size="sm"
					pulse={health.kind === 'dead'}
					title={health.title}
				/>
			{/if}
			<NodeSelect {panelId} state={linkState} emptyLabel="No node" />
			{#if node && controls}
				<div class="controls thin-scrollbar">{@render controls(node)}</div>
			{/if}
		{/snippet}
		{#snippet end()}
			{#if node}
				<IconButton variant="ghost" density="chrome" label="Unlink node" onclick={unlink}
					><Icon name="x" /></IconButton
				>
			{/if}
		{/snippet}
	</Bar>
	{#if node}
		<div class="body">
			{@render content(node)}
		</div>
	{:else}
		<div class="empty">
			<EmptyState>
				{#snippet title()}No node bound{/snippet}
				{#snippet hint()}Pick one above, or drag a node here, to show its {label}{/snippet}
			</EmptyState>
		</div>
	{/if}

	{#if dragActive}
		<div class="node-drop-hint" class:active={over} data-testid="node-drop-hint"></div>
	{/if}
</div>

<style>
	.linked {
		position: relative;
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	/* `gap: inherit` — the bar group's own gap, so the picker and these read as one row of siblings. */
	.controls {
		flex: 0 1 auto;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: inherit;
		overflow-x: auto;
		overflow-y: hidden;
	}
	.body {
		flex: 1;
		min-height: 0;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}
	.body > :global(*) {
		flex: 1;
		min-height: 0;
	}
	.empty {
		flex: 1;
		display: grid;
		place-items: center;
		padding: var(--space-7);
		text-align: center;
		color: var(--text-muted);
	}
</style>
