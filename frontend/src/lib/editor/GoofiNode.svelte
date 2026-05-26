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

	function onInputClick(e: MouseEvent, slot: string, dtype: string) {
		// Connected slots fall through to SvelteFlow's drag-start. Output
		// slots have their own click handler in SlotViewer.svelte.
		if (g.links.some((l) => l.node_in === node.name && l.slot_in === slot)) return;
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.name, slot, dtype, side: 'target', clientX: e.clientX, clientY: e.clientY });
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

	{#if inputs.length > 0}
		<div class="body">
			{#each inputs as slot (slot)}
				<div
					class="port-row"
					style="--dtype: {dtypeColor(node.input_slots[slot])};"
					onclick={(e) => onInputClick(e, slot, node.input_slots[slot])}
					role="button"
					tabindex="0"
					data-testid="slot-input"
					title={node.input_slots[slot].toLowerCase()}
				>
					<Handle id={slot} type="target" position={Position.Left} />
					<span class="port-label">{slot}</span>
				</div>
			{/each}
		</div>
	{/if}

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
		/* baseline (not center) is what actually lands the mono glyphs
		   visually centered with the round health dot — center on the line
		   box leaves the ascender bias intact. */
		align-items: baseline;
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
		/* `normal` line-height + parent's `align-items: baseline` puts the
		   text baseline on the same line as the health dot. Trying to force
		   line-height to 1 leaves the optical glyph centre above where the
		   flex centring expects it; baseline alignment side-steps the
		   problem entirely. */
		line-height: normal;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1 1 auto;
		min-width: 0;
	}
	.body {
		padding: 6px 8px;
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
		font-size: 11px;
	}
	.port-row {
		/* `position: relative` + width-to-content puts the row's bounding
		   box on the left edge of the body, so SF's
		   `.svelte-flow__handle-left { top: 50% }` anchors per-row and the
		   handle hit-area stays tight around the slot name. */
		position: relative;
		display: flex;
		align-items: center;
		gap: 4px;
		min-height: 16px;
		padding: 1px 4px;
		width: max-content;
		max-width: 100%;
		border-radius: 4px;
		cursor: pointer;
		transition: background 80ms ease;
		font-family: var(--font-mono);
		color: var(--dtype, var(--text));
	}
	.port-row:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 15%, transparent);
	}
	.port-row :global(.svelte-flow__handle) {
		background: var(--dtype, var(--bg-elev-3));
		border-color: var(--dtype, var(--border-strong));
	}
	/* Overhang the pin past the node's body padding (8px) so it sits on
	   the outer border. SF's translate(-50%, -50%) already gives ~5px of
	   overhang; nudge another 3px outward. Output handles live in
	   SlotViewer.svelte. */
	.port-row :global(.svelte-flow__handle-left) {
		left: -8px;
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
