<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { dtypeColor } from './categoryColor';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { flash } from '$lib/stores/flash.svelte';
	import { NODE, inputPorts, inputUnits } from './nodeMetrics';
	import { nodeHealth } from './nodeHealth';
	import { formatUpdateRate } from './nodeStats';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	// Inside a sub-patch the node shows its local (un-namespaced) name; elsewhere
	// its display name. The id stays the uid (the universal identity) — only the
	// label differs.
	const label = $derived((data.label as string | undefined) ?? node?.name);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const uiStore = ui();

	// A sub-patch instance is rendered by THIS component with no special-casing: its
	// wired boundaries are its slots, its output viewers stream its inner members.
	// Enter it by double-click; its sharing/expand controls live in the inspector
	// (SubPatchInspector) when selected — so the node body never diverges from a
	// regular node's.

	function onInputClick(e: MouseEvent, slot: string, dtype: string): void {
		// Clicking an input opens the add-node menu seeded for this slot — whether
		// or not a cable is attached. Picking a node REPLACES the existing input
		// connection (inputs are single-source); see autoLink in NodeEditorPanel.
		// A drag from the handle is still a drag, not a click, so reconnect by
		// dragging keeps working.
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.uid, slot, dtype, side: 'target', clientX: e.clientX, clientY: e.clientY });
	}

	function onOutputClick(e: MouseEvent, slot: string, dtype: string): void {
		// Mirror of the slot-name click in SlotViewer: outputs fan out, so this
		// seeds a new downstream node without disconnecting anything. Dragging the
		// pill still starts a connection.
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.uid, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	// Health: ok / error (code, red border) / booting.
	const health = $derived(nodeHealth(node));
	const isError = $derived(health.kind === 'error');
	// Still coming up (child importing its implementation / running setup()) —
	// spinner + stage label instead of the status dot, body slightly dimmed.
	const isBooting = $derived(health.kind === 'booting');
	// Brief "this just changed" pulse after an undo/redo reorients here (#19).
	const flashing = $derived(flash().active(node?.uid));
	const healthColor = $derived(isError ? 'var(--danger)' : 'var(--success)');
	// Glanceable update rate at the right of the header — faint by default, brought
	// forward on hover. Null until the node's first NODE_STATS push. Adds no height.
	const rateLabel = $derived(formatUpdateRate(node?.stats));

	// Inputs are bare connectors on the left edge, stepping down from the top. A
	// single slot is one unit tall; a MULTI (list) slot is 2× tall — the affordance
	// that it accepts an arbitrary number of cables. `input_multi` is static type
	// shape from the backend (which slots are multi), read-only here.
	const multiInputs = $derived(new Set(node?.input_multi ?? []));
	// Plain closure over the reactive set — read at call time, no $derived needed.
	const isMulti = (slot: string) => multiInputs.has(slot);
	const inPorts = $derived(
		inputPorts(inputs, isMulti).map((p) => ({
			...p,
			dtype: node.input_slots[p.slot],
			multi: p.units === 2 // multi ⇔ 2 units (see inputPorts)
		}))
	);
	// The node must be tall enough to host every input block (viewers grow it past
	// that when there are more outputs).
	const minBody = $derived(inputUnits(inputs, isMulti));

	// Output ports live in the unclipped overlay, so we position them ourselves by
	// walking the slot stack: header first, then each slot — one unit tall when
	// collapsed, plus the viewer plot when open. This mirrors how the SlotViewers
	// lay out inside the surface, so every pill lands on its slot's header bar.
	const outPorts = $derived.by(() => {
		let y = NODE.border + NODE.header;
		return outputs.map((slot) => {
			const top = y + NODE.unit / 2;
			y += uiStore.isSlotExpanded(node.uid, slot) ? NODE.unit + NODE.viewer : NODE.unit;
			return { slot, dtype: node.output_slots[slot], top };
		});
	});
</script>

<div
	class="goofi-node"
	class:selected
	class:has-error={isError}
	class:booting={isBooting}
	class:undo-flash={flashing}
	style="min-height: calc(var(--node-header) + {minBody} * var(--node-u));"
	data-testid={node?.subpatch ? 'subpatch-node' : undefined}
>
	<!-- Visual surface: clipped to the rounded node shape, so the header's top
	     corners and the last slot's bottom corners are rounded once, here, no
	     matter how the slots stack. Nothing inside it needs to round itself. -->
	<div class="surface">
		<div class="header">
			{#if isBooting}
				<span class="boot-spinner" title={health.title} data-testid="boot-spinner"></span>
			{:else}
				<span class="health" style="background: {healthColor};" title={health.title}></span>
			{/if}
			<span class="name">{label}</span>
			{#if isBooting}
				<span class="boot-label" data-testid="boot-label">{health.label}</span>
			{:else if rateLabel}
				<span class="rate" title="update rate">{rateLabel}</span>
			{/if}
		</div>

		{#if outputs.length > 0}
			<div class="viewers">
				{#each outputs as slot (slot)}
					<SlotViewer node={node.uid} {slot} dtype={node.output_slots[slot]} label={node.slot_labels?.[slot]} />
				{/each}
			</div>
		{/if}
	</div>

	<!-- Connector overlay: sits outside the clip so the pills can overhang the
	     left/right edges. Pointer-transparent except for each connector box. -->
	<div class="ports">
		{#each inPorts as port (port.slot)}
			<!-- Pointer-only connector pill: opens the add-node menu at the cursor;
			     keyboard users reach "add node" via the topbar + menu. A multi (list)
			     slot renders 2× tall with a bar-shaped handle. -->
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<div
				class="conn in"
				class:multi={port.multi}
				style="top: {port.top}px; height: calc(var(--node-u) * {port.units}); --dtype: {dtypeColor(
					port.dtype
				)};"
				onclick={(e) => onInputClick(e, port.slot, port.dtype)}
				role="button"
				tabindex="0"
				data-testid="slot-input"
				data-multi={port.multi ? 'true' : undefined}
				title={port.multi ? `${port.dtype.toLowerCase()} · list (multi-input)` : port.dtype.toLowerCase()}
			>
				<Handle id={port.slot} type="target" position={Position.Left} />
				<span class="conn-label">{node.slot_labels?.[port.slot] ?? port.slot}</span>
			</div>
		{/each}

		{#each outPorts as port (port.slot)}
			<!-- Pointer-only connector pill: opens the add-node menu at the cursor;
			     keyboard users reach "add node" via the topbar + menu. -->
			<!-- svelte-ignore a11y_click_events_have_key_events -->
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
	/* A booting node (child importing / running setup): dimmed body + a small
	   spinner and stage label in the header. Fully shaped from spec metadata —
	   slots and params render normally — just not live yet. */
	.goofi-node.booting .surface {
		opacity: 0.75;
	}
	.boot-spinner {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
		border: 1.5px solid color-mix(in srgb, var(--accent) 40%, transparent);
		border-top-color: var(--accent);
		animation: boot-spin 0.8s linear infinite;
		box-sizing: border-box;
	}
	@keyframes boot-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.boot-label {
		flex: 0 0 auto;
		font-size: 9px;
		color: var(--text-muted, #8a8f98);
		font-style: italic;
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
		.boot-spinner {
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
		background: var(--surface-3);
		border-bottom: 1px solid var(--border);
		cursor: pointer;
		user-select: none;
	}
	.health {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
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
	/* Glanceable update rate, right-aligned. Faint at rest so it doesn't compete
	   with the name; brought forward when the node is hovered. Tabular so the ~1 Hz
	   number updates don't jitter the layout. */
	.rate {
		flex: 0 0 auto;
		font-size: 9px;
		color: var(--text-muted, #8a8f98);
		font-variant-numeric: tabular-nums;
		opacity: 0.3;
		transition: opacity 120ms ease;
	}
	.goofi-node:hover .rate {
		opacity: 0.85;
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
	/* A multi (list) input: a taller, bar-shaped handle spanning the 2× box, so it
	   reads as "accepts a stack of cables" rather than a single connector. */
	.conn.in.multi :global(.svelte-flow__handle-left) {
		height: calc(var(--node-u) * 1.15);
		border-radius: 3px;
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
