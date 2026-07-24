<!--
  Shared chrome for the node-linked panels (Parameters / Viewer / Metadata).
  A node is linked by dragging it (by its grip) from any editor into the panel;
  the link (node uid — stable across rename + save/load) is stored in the panel
  state, so it persists in the .gfi and across selection changes — independent of
  which editor is focused.

  Empty until something is dropped; deleting the linked node empties it again
  (the store clears stale refs on node removal). The `content` snippet renders
  the panel's body once a node is resolved.
-->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { linkedNodeName, withLinkedNode } from '$lib/workspace/panelState';
	import type { Snippet } from 'svelte';

	let {
		panelId,
		state: linkState,
		setState,
		active,
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

	const linkedName = $derived(linkedNodeName(linkState));
	const node = $derived(linkedName ? g.nodeById(linkedName) : null);
	// A node is being dragged from an editor (the editor drives the link on
	// release); `over` is true when it's currently over this panel.
	const dragActive = $derived(uiStore.nodeDrag !== null);
	const over = $derived(uiStore.nodeDragTarget === panelId);

	function unlink(): void {
		setState(withLinkedNode(linkState, null));
	}
</script>

<div
	class="linked"
	class:dragging={dragActive}
	class:over
	class:active
	role="group"
	data-testid="node-linked-panel"
>
	{#if node}
		<div class="linkbar">
			<span class="dot" class:err={node.error}></span>
			<span class="ln" title={node.type}>{node.name}</span>
			{#if controls}
				<div class="controls">{@render controls(node)}</div>
			{/if}
			<button class="unlink" title="Unlink node" aria-label="Unlink node" onclick={unlink}>✕</button>
		</div>
		<div class="body">
			{@render content(node)}
		</div>
	{:else}
		<div class="empty">
			<span class="hint">Drag a node here<br /><small>to show its {label}</small></span>
		</div>
	{/if}

	{#if dragActive}
		<div class="drop-hint" class:active={over} data-testid="node-drop-hint"></div>
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
	.linkbar {
		display: flex;
		align-items: center;
		gap: 6px;
		flex: 0 0 auto;
		padding: 4px 8px;
		background: var(--surface-1);
		border-bottom: 1px solid var(--border);
		font-size: 0.78rem;
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--success);
		flex: 0 0 auto;
	}
	.dot.err {
		background: var(--danger);
	}
	.ln {
		font-family: var(--font-mono);
		color: var(--text);
		flex: 0 1 auto;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* Host for the panel's own controls (group tabs / slot picker / viewer
	   selector). Fills the slack between the node name and the unlink button and
	   scrolls horizontally if its contents overflow. The left margin keeps the
	   controls visually separated from the node name. */
	.controls {
		flex: 1 1 auto;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: 6px;
		margin-left: 10px;
		overflow-x: auto;
		overflow-y: hidden;
		scrollbar-width: thin;
	}
	.unlink {
		width: 18px;
		height: 18px;
		display: grid;
		place-items: center;
		padding: 0;
		font-size: 0.7rem;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
	}
	.unlink:hover {
		color: var(--danger);
		background: var(--surface-2);
	}
	.body {
		position: relative;
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
		position: relative;
		flex: 1;
		display: grid;
		place-items: center;
		padding: 16px;
		text-align: center;
		color: var(--text-muted);
	}
	/* Active-panel accent — framed around just the content below the inner
	   header bar (not the bars themselves). Drawn as an overlay so it stays
	   visible over opaque viewer content. */
	.linked.active .body::after,
	.linked.active .empty::after {
		content: '';
		position: absolute;
		inset: 0;
		pointer-events: none;
		border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
		/* Square at the top (meets the inner header bar), rounded at the panel's
		   bottom corners. */
		border-radius: 0 0 var(--radius-sm) var(--radius-sm);
		z-index: 4;
	}
	.hint small {
		color: var(--text-dim);
	}
	/* Dashed outline marks the panel as a drop target while a node is dragged;
	   fills in when the cursor is over it. */
	.drop-hint {
		position: absolute;
		inset: 4px;
		pointer-events: none;
		border: 2px dashed color-mix(in srgb, var(--accent) 55%, transparent);
		border-radius: var(--radius-sm);
		z-index: var(--z-drag-ghost);
	}
	.drop-hint.active {
		border-style: solid;
		background: color-mix(in srgb, var(--accent) 16%, transparent);
	}
</style>
