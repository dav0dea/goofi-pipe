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
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { linkedNodeName } from '$lib/workspace/panelState';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { Bar, Icon, IconButton, StatusDot, EmptyState } from '$lib/ui';
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
			{#if node}<StatusDot tone={node.error ? 'error' : 'ok'} size="sm" />{/if}
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
	/* Host for the panel's own controls (group tabs / slot picker / viewer
	   selector). Fills the slack between the node name and the unlink button and
	   scrolls horizontally if its contents overflow. The left margin keeps the
	   controls visually separated from the node name. */
	.controls {
		flex: 0 1 auto;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-left: var(--space-6);
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
