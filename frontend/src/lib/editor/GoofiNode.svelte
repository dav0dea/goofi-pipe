<script lang="ts">
	import { Handle, Position, useNodes, type NodeProps } from '@xyflow/svelte';
	import { categoryColor, formatName } from './categoryColor';
	import { graph } from '$lib/stores/graph.svelte';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const expanded = $derived(Boolean(data.expanded));

	function toggleExpanded(e: MouseEvent) {
		e.stopPropagation();
		const nodes = useNodes();
		nodes.update((arr) =>
			arr.map((n) =>
				n.id === node.name ? { ...n, data: { ...n.data, expanded: !expanded } } : n
			)
		);
	}

	const accent = $derived(categoryColor(node?.category));
	const hasError = $derived(Boolean(node?.error));
</script>

<div
	class="goofi-node"
	class:selected
	class:has-error={hasError}
	style="--accent: {accent};"
	title={node?.doc ?? ''}
>
	<div class="header" onclick={toggleExpanded} role="button" tabindex="0">
		<span class="dot" style="background: {accent};"></span>
		<span class="title">{formatName(node?.type ?? node?.name)}</span>
		<span class="name">{node?.name}</span>
		<button class="ghost expand" onclick={toggleExpanded} aria-label="toggle viewers">
			{expanded ? '▾' : '▸'}
		</button>
	</div>

	<div class="body">
		<div class="ports inputs">
			{#each inputs as slot (slot)}
				<div class="port-row">
					<Handle id={slot} type="target" position={Position.Left} />
					<span class="port-label">{slot}</span>
					<span class="port-dtype">{node.input_slots[slot]}</span>
				</div>
			{/each}
		</div>
		<div class="ports outputs">
			{#each outputs as slot (slot)}
				<div class="port-row out">
					<span class="port-dtype">{node.output_slots[slot]}</span>
					<span class="port-label">{slot}</span>
					<Handle id={slot} type="source" position={Position.Right} />
				</div>
			{/each}
		</div>
	</div>

	{#if expanded}
		<div class="viewers">
			{#each outputs as slot (slot)}
				<SlotViewer node={node.name} {slot} dtype={node.output_slots[slot]} />
			{/each}
		</div>
	{/if}

	{#if hasError}
		<div class="error-banner" title={node.error ?? ''}>error</div>
	{/if}
</div>

<style>
	.goofi-node {
		min-width: 200px;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		overflow: hidden;
		box-shadow: var(--shadow-1);
		transition:
			border-color 80ms ease,
			box-shadow 80ms ease;
		position: relative;
	}
	.goofi-node.selected {
		border-color: var(--accent);
		box-shadow: var(--shadow-2);
	}
	.goofi-node.has-error {
		border-color: var(--danger);
	}
	.header {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 10px;
		background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 18%, transparent), transparent);
		border-bottom: 1px solid var(--border);
		cursor: pointer;
		user-select: none;
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.title {
		font-weight: 600;
		font-size: 12px;
	}
	.name {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		margin-left: auto;
		opacity: 0.7;
	}
	.expand {
		margin-left: 4px;
		padding: 0 4px;
		min-width: 14px;
		color: var(--text-dim);
	}
	.body {
		padding: 6px 8px;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 4px;
		font-size: 11px;
	}
	.ports {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.port-row {
		display: flex;
		align-items: center;
		gap: 4px;
		min-height: 16px;
	}
	.port-row.out {
		justify-content: flex-end;
	}
	.port-label {
		color: var(--text);
	}
	.port-dtype {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--text-faint);
		text-transform: lowercase;
	}
	.viewers {
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px;
		background: var(--bg-elev-2);
	}
	.error-banner {
		position: absolute;
		top: -8px;
		right: 8px;
		background: var(--danger);
		color: #1a0709;
		font-size: 9px;
		font-weight: 700;
		padding: 1px 6px;
		border-radius: 4px;
		text-transform: uppercase;
		pointer-events: auto;
	}
</style>
