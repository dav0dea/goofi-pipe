<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { categoryColor, dtypeColor, formatName } from './categoryColor';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { onMount, tick } from 'svelte';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const uiStore = ui();
	const g = graph();
	const expanded = $derived(uiStore.isExpanded(node?.name ?? ''));

	let nodeEl: HTMLDivElement | null = $state(null);
	// Per-slot Y coordinates (centered on the corresponding label row) relative
	// to the outer node element. Handles are rendered as absolutely-positioned
	// children of the node and need exact Y per slot so they line up with the
	// label and so the inside-clipping doesn't eat their hover/click area.
	let inputYs = $state<number[]>([]);
	let outputYs = $state<number[]>([]);

	function measureRows(): void {
		if (!nodeEl) return;
		const nodeRect = nodeEl.getBoundingClientRect();
		const inRows = nodeEl.querySelectorAll<HTMLElement>('.port-row.in');
		const outRows = nodeEl.querySelectorAll<HTMLElement>('.port-row.out');
		const nextIn: number[] = [];
		const nextOut: number[] = [];
		inRows.forEach((r) => {
			const rect = r.getBoundingClientRect();
			nextIn.push(rect.top + rect.height / 2 - nodeRect.top);
		});
		outRows.forEach((r) => {
			const rect = r.getBoundingClientRect();
			nextOut.push(rect.top + rect.height / 2 - nodeRect.top);
		});
		inputYs = nextIn;
		outputYs = nextOut;
	}

	// Re-measure whenever the dependency surface that affects row positions
	// changes — slot lists, the expanded flag, and (after mount) the node's
	// actual rendered size via ResizeObserver.
	$effect(() => {
		// Subscribe to reactivity; the actual measurement happens after the
		// DOM updates so we use tick() to defer.
		void inputs.length;
		void outputs.length;
		void expanded;
		void tick().then(measureRows);
	});

	onMount(() => {
		measureRows();
		if (!nodeEl) return;
		const obs = new ResizeObserver(() => measureRows());
		obs.observe(nodeEl);
		nodeEl.querySelectorAll('.port-row').forEach((r) => obs.observe(r));
		return () => obs.disconnect();
	});

	function toggleExpanded(e: MouseEvent) {
		e.stopPropagation();
		uiStore.toggleExpanded(node.name);
	}

	function isConnected(slot: string, side: 'source' | 'target'): boolean {
		for (const l of g.links) {
			if (side === 'source' && l.node_out === node.name && l.slot_out === slot) return true;
			if (side === 'target' && l.node_in === node.name && l.slot_in === slot) return true;
		}
		return false;
	}

	function onSlotClick(e: MouseEvent, slot: string, dtype: string, side: 'source' | 'target') {
		if (isConnected(slot, side)) return; // let the click reach SvelteFlow
		e.stopPropagation();
		uiStore.requestSlotClick({
			node: node.name,
			slot,
			dtype,
			side,
			clientX: e.clientX,
			clientY: e.clientY
		});
	}

	const accent = $derived(categoryColor(node?.category));
	const hasError = $derived(Boolean(node?.error));
	const healthColor = $derived(hasError ? 'var(--danger)' : 'var(--success)');
</script>

<div
	bind:this={nodeEl}
	class="goofi-node"
	class:selected
	class:has-error={hasError}
	style="--accent: {accent};"
	title={node?.doc ?? ''}
>
	<!-- Inner clip: everything that uses the rounded silhouette + the
	     1px border lives here. The Handles are OUTSIDE this so they can
	     overhang the node's edge without being clipped. -->
	<div class="node-clip">
		<div class="header">
			<span class="dot" style="background: {accent};"></span>
			<span class="title">{formatName(node?.type ?? node?.name)}</span>
			<span class="health" style="background: {healthColor};" title={node?.error ?? 'running'}></span>
			<span class="name">{node?.name}</span>
			<button class="ghost expand" onclick={toggleExpanded} aria-label="toggle viewers">
				{expanded ? '▾' : '▸'}
			</button>
		</div>

		<div class="body">
			<div class="ports inputs">
				{#each inputs as slot (slot)}
					<div
						class="port-row in clickable"
						style="--dtype: {dtypeColor(node.input_slots[slot])};"
						onclick={(e) => onSlotClick(e, slot, node.input_slots[slot], 'target')}
						role="button"
						tabindex="0"
						data-testid="slot-input"
					>
						<span class="port-label">{slot}</span>
						<span class="port-dtype">{node.input_slots[slot].toLowerCase()}</span>
					</div>
				{/each}
			</div>
			<div class="ports outputs">
				{#each outputs as slot (slot)}
					<div
						class="port-row out clickable"
						style="--dtype: {dtypeColor(node.output_slots[slot])};"
						onclick={(e) => onSlotClick(e, slot, node.output_slots[slot], 'source')}
						role="button"
						tabindex="0"
						data-testid="slot-output"
					>
						<span class="port-dtype">{node.output_slots[slot].toLowerCase()}</span>
						<span class="port-label">{slot}</span>
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
	</div>

	<!-- Handles: rendered as positioned siblings of .node-clip so they can
	     overflow the node and stay clickable beyond its silhouette. Each
	     handle's Y is set to the measured center of its label row, and the
	     default `.svelte-flow__handle-left/-right` rules (`top: 50%` and a
	     translate(-50%, -50%)) are overridden below so only the X half-shift
	     remains. -->
	{#each inputs as slot, i (slot)}
		<Handle
			id={slot}
			type="target"
			position={Position.Left}
			style="top: {inputYs[i] ?? 24}px; --dtype: {dtypeColor(node.input_slots[slot])};"
		/>
	{/each}
	{#each outputs as slot, i (slot)}
		<Handle
			id={slot}
			type="source"
			position={Position.Right}
			style="top: {outputYs[i] ?? 24}px; --dtype: {dtypeColor(node.output_slots[slot])};"
		/>
	{/each}
</div>

<style>
	.goofi-node {
		min-width: 200px;
		color: var(--text);
		/* overflow:visible so the handle wrappers can overhang the silhouette.
		   The 1px border + rounded corners live on .node-clip below. */
		overflow: visible;
		box-shadow: var(--shadow-1);
		transition: box-shadow 80ms ease;
		position: relative;
	}
	.node-clip {
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
		transition: border-color 80ms ease;
	}
	.goofi-node.selected .node-clip {
		border-color: var(--accent);
	}
	.goofi-node.selected {
		box-shadow: var(--shadow-2);
	}
	.goofi-node.has-error .node-clip {
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
	.health {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
		box-shadow: 0 0 4px currentColor;
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
		min-height: 18px;
		border-radius: 4px;
		padding: 2px 4px;
	}
	.port-row.out {
		justify-content: flex-end;
	}
	.port-row.clickable {
		cursor: pointer;
		transition: background 80ms ease;
	}
	.port-row.clickable:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 15%, transparent);
	}
	.port-label {
		color: var(--text);
	}
	.port-dtype {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--dtype, var(--text-faint));
		text-transform: lowercase;
		opacity: 0.9;
	}
	/* Handle positioning — direct children of .goofi-node, positioned via
	   inline `top: <measuredY>` per slot. We override the framework's
	   default `top: 50%` (which would pin to node-center) and the Y half
	   of its translate(-50%, -50%) (which would shift the handle up by
	   half its size on top of the inline top). Only the X half-shift
	   remains so the handle still overhangs the node edge horizontally. */
	:global(.svelte-flow__node) .goofi-node :global(.svelte-flow__handle) {
		background: var(--dtype, var(--bg-elev-3));
		border-color: var(--dtype, var(--border-strong));
		width: 9px;
		height: 9px;
		z-index: 4;
	}
	:global(.svelte-flow__node) .goofi-node :global(.svelte-flow__handle-left) {
		transform: translate(-50%, 0);
	}
	:global(.svelte-flow__node) .goofi-node :global(.svelte-flow__handle-right) {
		transform: translate(50%, 0);
	}
	.viewers {
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px;
		background: var(--bg-elev-2);
	}
</style>
