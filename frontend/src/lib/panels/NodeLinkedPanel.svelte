<!--
  Shared chrome for the node-linked panels (Parameters / Viewer / Metadata).
  A node is linked either by picking it from the bar's dropdown (`NodeSelect`, the same control the
  Console wears) or by dragging it (by its grip) from any editor into the panel; the link (node uid —
  stable across rename + save/load) is stored in the panel state, so it persists in the .gfi and
  across selection changes — independent of which editor is focused.

  The bar is drawn either way — an unbound panel is one the picker can still bind, which matters most
  where there is no editor to drag from (another layout page, a phone). Deleting the linked node
  empties the panel again (the store clears stale refs on node removal). The `content` snippet
  renders the panel's body once a node is resolved.
-->
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
		/** Panel-specific controls rendered inline in the header bar next to the
		 * node name (group tabs / slot picker / viewer selector). */
		controls?: Snippet<[NodeInstanceInfo]>;
	} = $props();

	const g = graph();
	const uiStore = ui();
	const ws = workspace();

	const linkedName = $derived(linkedNodeName(linkState));
	const node = $derived(linkedName ? g.nodeById(linkedName) : null);
	// A node is being dragged from an editor (the editor drives the link on
	// release); `over` is true when it's currently over this panel.
	const dragActive = $derived(uiStore.nodeDrag !== null);
	const over = $derived(uiStore.nodeDragTarget === panelId);

	// Through the store, not the opaque `setState`: unlinking is an edit, and a layout undo
	// restores a whole snapshot — so an unrecorded one is destroyed by the next Ctrl+Z.
	function unlink(): void {
		ws.unlinkNodeFromPanel(panelId);
	}
</script>

<div class="linked" role="group" data-testid="node-linked-panel">
	<!-- The bar is here whether or not anything is bound: the picker in it is how an unbound panel
	     gets bound without a drag, which is the only door a phone has. -->
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
	/* Host for the panel's own controls (slot picker / viewer selector). It exists for ONE reason —
	   it fills the slack between the node picker and the unlink ✕ and scrolls horizontally when its
	   contents overflow, which they do on a phone (the viewer bar's controls are 215px in 184px of
	   slack) — so it must not also be a spacing decision. `inherit` takes the bar group's own gap,
	   so the picker and the controls beside it read as one row of siblings rather than two clusters
	   (they were the bar's gap PLUS a margin apart, and then closer to each other than to it). */
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
