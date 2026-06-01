<!--
  Shared chrome for the node-linked panels (Parameters / Viewer / Metadata).
  A node is linked by dragging it (by its grip) from any editor into the panel;
  the link (node name) is stored in the panel state, so it persists in the .gfi
  and across selection changes — independent of which editor is focused.

  Empty until something is dropped; deleting the linked node empties it again
  (the store clears stale refs on node removal). The `content` snippet renders
  the panel's body once a node is resolved.
-->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import type { Snippet } from 'svelte';

	let {
		state: linkState,
		setState,
		label,
		content
	}: PanelProps & { label: string; content: Snippet<[NodeInstanceInfo]> } = $props();

	const g = graph();
	const uiStore = ui();

	const linkedName = $derived(
		typeof linkState === 'object' && linkState !== null
			? ((linkState as { node?: string | null }).node ?? null)
			: null
	);
	const node = $derived(linkedName ? (g.nodes.find((n) => n.name === linkedName) ?? null) : null);
	const dragActive = $derived(uiStore.nodeDrag !== null);
	let over = $state(false);

	function base(): Record<string, unknown> {
		return typeof linkState === 'object' && linkState !== null
			? (linkState as Record<string, unknown>)
			: {};
	}
	function onDragOver(e: DragEvent): void {
		if (!dragActive) return;
		e.preventDefault();
		over = true;
	}
	function onDragLeave(e: DragEvent): void {
		if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) over = false;
	}
	function onDrop(e: DragEvent): void {
		over = false;
		const name = uiStore.nodeDrag;
		if (!name) return;
		e.preventDefault();
		e.stopPropagation();
		setState({ ...base(), node: name });
	}
	function unlink(): void {
		setState({ ...base(), node: null });
	}
</script>

<div
	class="linked"
	class:dragging={dragActive}
	class:over
	ondragover={onDragOver}
	ondragleave={onDragLeave}
	ondrop={onDrop}
	role="group"
	data-testid="node-linked-panel"
>
	{#if node}
		<div class="linkbar">
			<span class="dot" class:err={node.error}></span>
			<span class="ln" title={node.type}>{node.name}</span>
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
		background: var(--bg-elev-1);
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
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
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
		color: var(--text-faint);
		cursor: pointer;
	}
	.unlink:hover {
		color: var(--danger);
		background: var(--bg-elev-2);
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
		padding: 16px;
		text-align: center;
		color: var(--text-faint);
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
