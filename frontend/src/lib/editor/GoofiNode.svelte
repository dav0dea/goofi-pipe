<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { dtypeColor } from './categoryColor';
	import SlotViewer from '$lib/viewers/SlotViewer.svelte';
	import { isSlotExpanded } from '$lib/viewers/inlineView';
	import { ui } from '$lib/stores/ui.svelte';
	import { flash } from '$lib/stores/flash.svelte';
	import { NODE, inputPorts, inputUnits } from './nodeMetrics';
	import { nodeHealth } from './nodeHealth';
	import { StatusDot } from '$lib/ui';
	import { formatUpdateRate } from './nodeStats';
	import { graph } from '$lib/stores/graph.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const node = $derived(data.node as NodeInstanceInfo);
	const label = $derived((data.label as string | undefined) ?? node?.name);
	const inputs = $derived(Object.keys(node?.input_slots ?? {}));
	const outputs = $derived(Object.keys(node?.output_slots ?? {}));
	const uiStore = ui();
	// Reached HERE, never inside a derived: a lazy singleton first called in a tracking scope
	// creates its `$state` there, where an outside `pulse()` can never re-run it.
	const flashStore = flash();

	function onInputClick(e: MouseEvent, slot: string, dtype: string): void {
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.uid, slot, dtype, side: 'target', clientX: e.clientX, clientY: e.clientY });
	}

	function onOutputClick(e: MouseEvent, slot: string, dtype: string): void {
		e.stopPropagation();
		uiStore.requestSlotClick({ node: node.uid, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	/** Open the plugin's own editor. It appears on the machine goofi runs on, never in the page. */
	function onEditorClick(e: MouseEvent): void {
		e.stopPropagation();
		void graph()
			.showNodeEditor(node.uid, true)
			.catch((err) => console.warn('editor failed', err));
	}

	const health = $derived(nodeHealth(node));
	const isError = $derived(health.kind === 'error' || health.kind === 'dead');
	const isBooting = $derived(health.kind === 'booting');
	const flashing = $derived(flashStore.active(node?.uid));
	const rateLabel = $derived(formatUpdateRate(node?.stats));

	const multiInputs = $derived(new Set(node?.input_multi ?? []));
	const isMulti = (slot: string) => multiInputs.has(slot);
	const inPorts = $derived(
		inputPorts(inputs, isMulti).map((p) => ({
			...p,
			dtype: node.input_slots[p.slot],
			multi: p.units === 2
		}))
	);
	const minBody = $derived(inputUnits(inputs, isMulti));

	// The overlay is unclipped, so it walks the slot stack itself to place each output pill.
	const outPorts = $derived.by(() => {
		let y = NODE.border + NODE.header;
		return outputs.map((slot) => {
			const top = y + NODE.unit / 2;
			y += isSlotExpanded(node, slot) ? NODE.unit + NODE.viewer : NODE.unit;
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
	<!-- Clipped to the rounded node shape, so nothing inside it needs to round itself. -->
	<div class="surface">
		<div class="header">
			<!-- The card is sized in fixed px, so the dot is told its diameter. -->
			<StatusDot
				tone={health.tone}
				pulse={health.kind === 'dead'}
				title={health.title}
				style="--status-dot-size: 8px"
				data-testid="node-health"
			/>
			<span class="name">{label}</span>
			{#if isBooting}
				<span class="boot-label" data-testid="boot-label">{health.label}</span>
			{:else if rateLabel}
				<span class="rate" title="update rate">{rateLabel}</span>
			{/if}
			<!-- `nodrag` so the press opens the editor rather than starting a node drag. -->
			{#if node?.editor}
				<button
					class="editor nodrag"
					onclick={onEditorClick}
					title="Open this plugin's own editor window"
					aria-label="Open plugin editor"
					data-testid="node-editor">▤</button
				>
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

	<!-- Connector overlay: outside the clip, so the pills can overhang the edges. -->
	<div class="ports">
		{#each inPorts as port (port.slot)}
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<div
				class="conn in"
				class:multi={port.multi}
				class:cable-near={uiStore.isCableNear(node.uid, port.slot)}
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
	.surface {
		flex: 1 1 auto;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		transition:
			border-color var(--dur-fast) var(--ease),
			box-shadow var(--dur-fast) var(--ease);
	}
	.goofi-node.selected .surface {
		border-color: var(--accent);
		box-shadow: var(--shadow-2);
	}
	.goofi-node.has-error .surface {
		border-color: var(--danger);
	}
	.goofi-node.booting .surface {
		opacity: 0.75;
	}
	.boot-label {
		flex: 0 0 auto;
		font-size: 9px;
		color: var(--text-muted);
		font-style: italic;
	}
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
	.editor {
		margin-left: auto;
		flex: 0 0 auto;
		display: grid;
		place-items: center;
		/* The one hit target on a node header, so it carries the app's touch floor. */
		min-width: var(--hit);
		min-height: var(--hit);
		padding: 0;
		border: 0;
		background: none;
		color: var(--text-2);
		font-size: 12px;
		line-height: 1;
		cursor: pointer;
	}

	.editor:hover {
		color: var(--text-1);
	}

	.header {
		flex: 0 0 auto;
		height: var(--node-header);
		box-sizing: border-box;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 0 10px;
		background: var(--surface-2);
		border-bottom: 1px solid var(--border);
		cursor: pointer;
		user-select: none;
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
	.rate {
		flex: 0 0 auto;
		font-size: 9px;
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
		opacity: 0.3;
		transition: opacity var(--dur-slow) var(--ease);
	}
	.goofi-node:hover .rate,
	.goofi-node.selected .rate {
		opacity: 0.85;
	}
	/* No hover to resolve the faint resting state, so the rate rests where hover would leave it. */
	@media (hover: none) and (pointer: coarse) {
		.rate {
			opacity: 0.85;
		}
	}
	.viewers {
		display: flex;
		flex-direction: column;
	}

	/* These connectors sit BELOW `--hit` deliberately: they tile the left edge at `--node-u`, so a
	   44px target would cover its neighbours and make the WRONG slot answer a tap. */
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
	/* Centre the handle in its box, so the cable anchor SvelteFlow measures lands on the edge. */
	.conn.in :global(.svelte-flow__handle-left) {
		left: 50%;
	}
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
		transition: opacity var(--dur-fast) var(--ease);
	}
	.conn.in:hover .conn-label,
	.conn.in:focus-visible .conn-label,
	/* The touch door: while a cable is in flight, the inputs it nears name themselves. */
	.conn.in.cable-near .conn-label {
		opacity: 1;
	}
</style>
