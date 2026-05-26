<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { categoryColor, dtypeColor, formatName } from './categoryColor';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const uiStore = ui();
	const g = graph();

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
	const snap = $derived(uiStore.dragSnap[node?.name ?? '']);
	const snapStyle = $derived(
		snap ? `--snap-dx: ${snap.dx}px; --snap-dy: ${snap.dy}px;` : ''
	);
</script>

<div
	class="goofi-node"
	class:selected
	class:has-error={hasError}
	class:snapped={Boolean(snap)}
	style="--accent: {accent}; {snapStyle}"
	title={node?.doc ?? ''}
>
	<div class="header">
		<span class="health" style="background: {healthColor};" title={node?.error ?? 'running'}></span>
		<span class="name" title={formatName(node?.type ?? node?.name)}>{node?.name}</span>
	</div>

	<div class="body">
		<div class="ports inputs">
			{#each inputs as slot (slot)}
				<div
					class="port-row clickable"
					style="--dtype: {dtypeColor(node.input_slots[slot])};"
					onclick={(e) => onSlotClick(e, slot, node.input_slots[slot], 'target')}
					role="button"
					tabindex="0"
					data-testid="slot-input"
				>
					<Handle id={slot} type="target" position={Position.Left} />
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
					<Handle id={slot} type="source" position={Position.Right} />
				</div>
			{/each}
		</div>
	</div>

	{#if outputs.length > 0}
		<div class="viewers">
			{#each outputs as slot (slot)}
				<SlotViewer node={node.name} {slot} dtype={node.output_slots[slot]} />
			{/each}
		</div>
	{/if}
</div>

<style>
	.goofi-node {
		min-width: 200px;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		/* Visible — otherwise the overhanging slot handles get clipped at
		   the node's bounding box. Inner sections that rely on the rounded
		   border-radius (header gradient, viewers panel background) carry
		   their own matching corner radii below. */
		overflow: visible;
		box-shadow: var(--shadow-1);
		transition:
			border-color 80ms ease,
			box-shadow 80ms ease;
		position: relative;
		/* Live preview-snap offset applied during a snapped drag. Editor.svelte
		   writes (--snap-dx, --snap-dy) onto the dragged nodes via the ui
		   store; the transform sits on top of SvelteFlow's mouse-tracking
		   transform on the outer .svelte-flow__node wrapper. */
		transform: translate(var(--snap-dx, 0px), var(--snap-dy, 0px));
	}
	.goofi-node.snapped {
		/* Briefly cushion the snap engagement with an easing window — uPlot
		   refreshes still need to land within the next frame, so keep this
		   short. */
		transition:
			border-color 80ms ease,
			box-shadow 80ms ease,
			transform 60ms ease-out;
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
		/* Match the node's rounded top corners so the gradient doesn't
		   square off now that the node uses overflow: visible. */
		border-radius: var(--radius-md) var(--radius-md) 0 0;
		cursor: pointer;
		user-select: none;
	}
	.health {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
		box-shadow: 0 0 5px currentColor;
	}
	.name {
		font-family: var(--font-mono);
		font-weight: 600;
		font-size: 12px;
		color: var(--text);
		/* Keep the font's natural line-height (don't force 1) so capital
		   glyphs aren't clipped at the top, and add a single-pixel optical
		   nudge: JetBrains Mono's geometric centre sits slightly above the
		   x-height midline, so the text-box centre measured by flex lands
		   the glyphs about 1px too high relative to the round health dot. */
		line-height: 1.15;
		padding-top: 1px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1 1 auto;
		min-width: 0;
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
		border-radius: 4px;
		padding: 1px 2px;
		/* Establish a positioning context so Svelte Flow's
		   .svelte-flow__handle-{left,right} { top: 50% } anchors to THIS row,
		   not to the whole node — otherwise every handle stacks at the node's
		   vertical centre. */
		position: relative;
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
	:global(.svelte-flow__node) .port-row :global(.svelte-flow__handle) {
		background: var(--dtype, var(--bg-elev-3));
		border-color: var(--dtype, var(--border-strong));
	}
	/* Overhang the handle half-outside the node's body padding so the pin
	   sits on the node's outer border, matching goofi3's slot look. The
	   .body has 8px horizontal padding; SF's translate(±50%, -50%) already
	   gives ~5px of overhang, so we nudge another 3px outward.            */
	.port-row :global(.svelte-flow__handle-left) {
		left: -8px;
	}
	.port-row :global(.svelte-flow__handle-right) {
		right: -8px;
	}
	.viewers {
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px;
		background: var(--bg-elev-2);
		/* Round the bottom corners to match the (now overflow: visible)
		   node frame. */
		border-radius: 0 0 var(--radius-md) var(--radius-md);
	}
</style>
