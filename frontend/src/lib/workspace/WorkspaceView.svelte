<!--
  Renders the active workspace tab's layout tree. When a panel is maximized it
  takes over the whole area; otherwise the recursive `LayoutNode` renders the
  split tree. The active tab is keyed so switching tabs swaps the whole tree.
-->
<script lang="ts">
	import { findPanel } from './model';
	import { workspace } from './workspace.svelte';
	import LayoutNode from './LayoutNode.svelte';
	import Panel from './Panel.svelte';

	const ws = workspace();
	const active = $derived(ws.active);
	const maximized = $derived(
		ws.maximizedPanelId ? findPanel(active.root, ws.maximizedPanelId) : null
	);
</script>

<div class="workspace" data-testid="workspace">
	{#key active.id}
		{#if maximized}
			<Panel node={maximized} />
		{:else}
			<LayoutNode node={active.root} />
		{/if}
	{/key}
</div>

<style>
	.workspace {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
		padding: var(--space-2);
		gap: 0;
	}
	.workspace :global(> *) {
		flex: 1;
		min-width: 0;
		min-height: 0;
	}
</style>
