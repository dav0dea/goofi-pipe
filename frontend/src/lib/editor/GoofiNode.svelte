<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { categoryColor, dtypeColor } from './categoryColor';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { flash } from '$lib/stores/flash.svelte';
	import { NODE } from './nodeMetrics';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	// Inside a sub-patch the node shows its local (un-namespaced) name; elsewhere
	// its full graph name. The id stays the full name — only the label differs.
	const label = $derived((data.label as string | undefined) ?? node?.name);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const uiStore = ui();

	// A sub-patch instance is a node too: same surface, slots (its wired
	// boundaries), and output viewers — plus a glyph, a member-count badge, and
	// enter/expand controls in the header. Everything else is shared with a
	// regular node, so the two never drift.
	const sub = $derived(node?.subpatch ?? null);
	const enterSub = data.onEnter as ((id: string) => void) | undefined;
	const expandSub = data.onExpand as ((id: string) => void) | undefined;

	function onInputClick(e: MouseEvent, slot: string, dtype: string): void {
		// Clicking an input opens the add-node menu seeded for this slot — whether
		// or not a cable is attached. Picking a node REPLACES the existing input
		// connection (inputs are single-source); see autoLink in NodeEditorPanel.
		// A drag from the handle is still a drag, not a click, so reconnect by
		// dragging keeps working.
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.name, slot, dtype, side: 'target', clientX: e.clientX, clientY: e.clientY });
	}

	function onOutputClick(e: MouseEvent, slot: string, dtype: string): void {
		// Mirror of the slot-name click in SlotViewer: outputs fan out, so this
		// seeds a new downstream node without disconnecting anything. Dragging the
		// pill still starts a connection.
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.name, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	const accent = $derived(categoryColor(node?.category));
	const hasError = $derived(Boolean(node?.error));
	// Brief "this just changed" pulse after an undo/redo reorients here (#19).
	const flashing = $derived(flash().active(node?.name));
	const healthColor = $derived(hasError ? 'var(--danger)' : 'var(--success)');

	// Inputs are bare connectors on the left edge, one per slot unit from the top
	// (so their count never balloons the node). The node only needs to be tall
	// enough to host them — viewers grow it past that when there are more.
	const minBody = $derived(Math.max(inputs.length, 1));

	// Output ports live in the unclipped overlay, so we position them ourselves by
	// walking the slot stack: header first, then each slot — one unit tall when
	// collapsed, plus the viewer plot when open. This mirrors how the SlotViewers
	// lay out inside the surface, so every pill lands on its slot's header bar.
	const outPorts = $derived.by(() => {
		let y = NODE.border + NODE.header;
		return outputs.map((slot) => {
			const top = y + NODE.unit / 2;
			y += uiStore.isSlotExpanded(node.name, slot) ? NODE.unit + NODE.viewer : NODE.unit;
			return { slot, dtype: node.output_slots[slot], top };
		});
	});
</script>

<div
	class="goofi-node"
	class:selected
	class:has-error={hasError}
	class:undo-flash={flashing}
	class:subpatch={Boolean(sub)}
	style="--accent: {accent}; min-height: calc(var(--node-header) + {minBody} * var(--node-u));"
	data-testid={sub ? 'subpatch-node' : undefined}
>
	<!-- Visual surface: clipped to the rounded node shape, so the header's top
	     corners and the last slot's bottom corners are rounded once, here, no
	     matter how the slots stack. Nothing inside it needs to round itself. -->
	<div class="surface">
		<div class="header">
			<span class="health" style="background: {healthColor};" title={node?.error ?? 'running'}></span>
			{#if sub}
				<span class="glyph" class:shared={sub.shared} title={sub.shared ? 'shared (strict mirror)' : 'unique sub-patch'}>{sub.shared ? '⇄' : '▣'}</span>
			{/if}
			<span class="name">{label}</span>
			{#if sub}
				<span class="count" title="{sub.memberCount} nodes">{sub.memberCount}</span>
				<button
					class="sp-btn"
					title="Enter sub-patch (edit inside)"
					aria-label="Enter sub-patch"
					data-testid="subpatch-enter"
					onclick={(e) => {
						e.stopPropagation();
						enterSub?.(sub.instId);
					}}>⮕</button
				>
				<button
					class="sp-btn"
					title="Expand sub-patch (dissolve into its nodes)"
					aria-label="Expand sub-patch"
					data-testid="subpatch-expand"
					onclick={(e) => {
						e.stopPropagation();
						expandSub?.(sub.instId);
					}}>⤢</button
				>
			{/if}
		</div>

		{#if outputs.length > 0}
			<div class="viewers">
				{#each outputs as slot (slot)}
					<SlotViewer node={node.name} {slot} dtype={node.output_slots[slot]} />
				{/each}
			</div>
		{/if}
	</div>

	<!-- Connector overlay: sits outside the clip so the pills can overhang the
	     left/right edges. Pointer-transparent except for each connector box. -->
	<div class="ports">
		{#each inputs as slot, i (slot)}
			<div
				class="conn in"
				style="top: calc({NODE.border}px + var(--node-header) + var(--node-u) * {i + 0.5}); --dtype: {dtypeColor(
					node.input_slots[slot]
				)};"
				onclick={(e) => onInputClick(e, slot, node.input_slots[slot])}
				role="button"
				tabindex="0"
				data-testid="slot-input"
				title={node.input_slots[slot].toLowerCase()}
			>
				<Handle id={slot} type="target" position={Position.Left} />
				<span class="conn-label">{slot}</span>
			</div>
		{/each}

		{#each outPorts as port (port.slot)}
			<div
				class="conn out"
				style="top: {port.top}px; --dtype: {dtypeColor(port.dtype)};"
				onclick={(e) => onOutputClick(e, port.slot, port.dtype)}
				role="button"
				tabindex="0"
				data-testid="slot-output-pin"
				title={port.dtype.toLowerCase()}
			>
				<Handle id={port.slot} type="source" position={Position.Right} />
			</div>
		{/each}
	</div>
</div>

<style>
	.goofi-node {
		position: relative;
		display: flex;
		flex-direction: column;
		width: var(--node-w);
		color: var(--text);
		font-family: var(--font-mono);
	}
	/* The painted node. It carries the flow height (header + viewers), and the
	   min-height on .goofi-node grows it to fit the inputs; flex:1 stretches it to
	   fill that. Clipping it rounds every inner corner uniformly. */
	.surface {
		flex: 1 1 auto;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-1);
		transition:
			border-color 80ms ease,
			box-shadow 80ms ease;
	}
	.goofi-node.selected .surface {
		border-color: var(--accent);
		box-shadow: var(--shadow-2);
	}
	.goofi-node.has-error .surface {
		border-color: var(--danger);
	}
	/* Undo/redo just reoriented here — a one-shot ring pulse to catch the eye
	   (#19). The class is removed after the window, so the animation re-fires on
	   the next undo/redo that lands on this node. */
	.goofi-node.undo-flash .surface {
		animation: undo-flash 0.7s ease-out;
	}
	@keyframes undo-flash {
		0% {
			box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 80%, transparent);
		}
		100% {
			box-shadow: 0 0 0 10px color-mix(in srgb, var(--accent) 0%, transparent);
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.goofi-node.undo-flash .surface {
			animation: none;
		}
	}
	.header {
		flex: 0 0 auto;
		height: var(--node-header);
		box-sizing: border-box;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 0 10px;
		background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 18%, transparent), transparent);
		border-bottom: 1px solid var(--border);
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
		font-weight: 600;
		font-size: 12px;
		color: var(--text);
		line-height: normal;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1 1 auto;
		min-width: 0;
	}
	/* A collapsed sub-patch reads as a node with a slightly stronger border + a
	   marker glyph, so it's recognizable without a separate component. */
	.goofi-node.subpatch .surface {
		background: var(--bg-elev-2);
		border-color: var(--border-strong);
	}
	.glyph {
		flex: 0 0 auto;
		color: var(--accent);
		font-size: 12px;
	}
	.glyph.shared {
		color: var(--cat-array, var(--accent));
	}
	.count {
		flex: 0 0 auto;
		font-size: 10px;
		color: var(--text-faint);
		background: color-mix(in srgb, var(--accent) 16%, transparent);
		border-radius: 8px;
		padding: 1px 7px;
	}
	.sp-btn {
		flex: 0 0 auto;
		background: transparent;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		font-size: 11px;
		line-height: 1;
		padding: 2px 5px;
	}
	.sp-btn:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.viewers {
		display: flex;
		flex-direction: column;
	}

	/* Connector overlay — pointer-transparent layer over the whole node; each
	   .conn re-enables pointer events. Inputs step down the left edge by one unit;
	   outputs are placed (in px) on their slot's header bar by outPorts above. */
	.ports {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.conn {
		position: absolute;
		display: grid;
		place-items: center;
		height: var(--node-u);
		pointer-events: auto;
		cursor: pointer;
	}
	.conn.in {
		left: 0;
		width: 22px;
		transform: translate(-50%, -50%);
	}
	.conn.out {
		right: 0;
		width: 16px;
		transform: translate(50%, -50%);
	}
	/* Centre each handle within its connector box (the box hugs the border), so the
	   pill — and the cable anchor SvelteFlow measures from it — lands on the edge. */
	.conn.in :global(.svelte-flow__handle-left) {
		left: 50%;
	}
	.conn.out :global(.svelte-flow__handle-right) {
		right: 50%;
	}
	.conn-label {
		position: absolute;
		right: calc(100% + 6px);
		top: 50%;
		transform: translateY(-50%);
		font-size: 9px;
		line-height: 1;
		color: var(--text-dim);
		background: var(--surface-glass);
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 2px 5px;
		white-space: nowrap;
		pointer-events: none;
		opacity: 0;
		transition: opacity 90ms ease;
	}
	.conn.in:hover .conn-label {
		opacity: 1;
	}
</style>
