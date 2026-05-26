<script lang="ts">
	import { ViewportPortal, useSvelteFlow } from '@xyflow/svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { categoryColor, dtypeColor } from './categoryColor';
	import {
		computeSnapDelta,
		makeBounds,
		DEFAULT_NODE_W,
		DEFAULT_NODE_H,
		type Bounds
	} from './snap';
	import type { NodeTypeInfo } from '$lib/api/control';

	interface Props {
		typeInfo: NodeTypeInfo;
		/** Initial mouse position in client coords — used before the first
		 * mousemove so the ghost spawns at the user's cursor rather than (0,0). */
		initialClient: { x: number; y: number };
		/** Measured widths/heights of currently-rendered nodes keyed by name. */
		measurements: Map<string, { width: number; height: number }>;
		onCommit: (pos: [number, number]) => void;
		onCancel: () => void;
	}

	let { typeInfo, initialClient, measurements, onCommit, onCancel }: Props = $props();

	const { screenToFlowPosition } = useSvelteFlow();
	const g = graph();
	const accent = $derived(categoryColor(typeInfo.category));

	let mouseClient = $state<{ x: number; y: number }>({ x: initialClient.x, y: initialClient.y });
	let altKey = $state(false);
	let ghostW = $state(DEFAULT_NODE_W);
	let ghostH = $state(DEFAULT_NODE_H);

	const flowPos = $derived(screenToFlowPosition({ x: mouseClient.x, y: mouseClient.y }));

	const snap = $derived.by(() => {
		const dragged = [makeBounds(flowPos.x, flowPos.y, ghostW, ghostH)];
		const targets: Bounds[] = [];
		for (const n of g.nodes) {
			const m = measurements.get(n.name);
			const w = m?.width ?? DEFAULT_NODE_W;
			const h = m?.height ?? DEFAULT_NODE_H;
			targets.push(makeBounds(n.pos[0], n.pos[1], w, h));
		}
		return computeSnapDelta(dragged, targets, altKey);
	});

	const snappedX = $derived(flowPos.x + snap.dx);
	const snappedY = $derived(flowPos.y + snap.dy);

	const inputs = $derived(Object.entries(typeInfo.input_slots));
	const outputs = $derived(Object.entries(typeInfo.output_slots));

	function onMouseMove(e: MouseEvent): void {
		mouseClient = { x: e.clientX, y: e.clientY };
		altKey = e.altKey;
	}

	function onKeyDown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			onCancel();
		}
	}

	function inCanvas(target: EventTarget | null): boolean {
		return Boolean((target as HTMLElement | null)?.closest?.('.canvas-wrap'));
	}

	function onWindowClick(e: MouseEvent): void {
		if (e.button !== 0) return;
		if (!inCanvas(e.target)) {
			onCancel();
			return;
		}
		// Block SF's pane-click / node-click from also firing.
		e.stopPropagation();
		e.preventDefault();
		onCommit([Math.round(snappedX), Math.round(snappedY)]);
	}

	function onWindowMouseDown(e: MouseEvent): void {
		if (e.button !== 0) return;
		if (!inCanvas(e.target)) return;
		// Stop SF from starting a pan/select/drag while placement is pending.
		e.stopPropagation();
		e.preventDefault();
	}

	$effect(() => {
		window.addEventListener('mousemove', onMouseMove);
		window.addEventListener('keydown', onKeyDown, true);
		window.addEventListener('click', onWindowClick, true);
		window.addEventListener('mousedown', onWindowMouseDown, true);
		return () => {
			window.removeEventListener('mousemove', onMouseMove);
			window.removeEventListener('keydown', onKeyDown, true);
			window.removeEventListener('click', onWindowClick, true);
			window.removeEventListener('mousedown', onWindowMouseDown, true);
		};
	});
</script>

<ViewportPortal target="front">
	<div
		class="ghost"
		bind:offsetWidth={ghostW}
		bind:offsetHeight={ghostH}
		style="transform: translate({snappedX}px, {snappedY}px); --accent: {accent};"
		data-testid="placement-ghost"
	>
		<div class="header">
			<span class="health"></span>
			<span class="name">{typeInfo.type}</span>
		</div>
		{#if inputs.length > 0}
			<div class="body">
				{#each inputs as [slot, dtype] (slot)}
					<div class="port-row" style="--dtype: {dtypeColor(dtype)};">
						<span class="pin in"></span>
						<span class="port-label">{slot}</span>
					</div>
				{/each}
			</div>
		{/if}
		{#if outputs.length > 0}
			<div class="viewers">
				{#each outputs as [slot, dtype] (slot)}
					<div class="port-row out" style="--dtype: {dtypeColor(dtype)};">
						<span class="port-label">{slot}</span>
						<span class="pin out"></span>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	{#if snap.guides.length > 0}
		<svg class="snap-guides" data-testid="placement-snap-guides">
			{#each snap.guides as gd, i (i)}
				{#if gd.x !== undefined}
					<line
						x1={gd.x}
						x2={gd.x}
						y1={-5000}
						y2={5000}
						stroke="var(--accent)"
						stroke-width="1"
						stroke-opacity={gd.opacity}
					/>
				{:else if gd.y !== undefined}
					<line
						x1={-5000}
						x2={5000}
						y1={gd.y}
						y2={gd.y}
						stroke="var(--accent)"
						stroke-width="1"
						stroke-opacity={gd.opacity}
					/>
				{/if}
			{/each}
		</svg>
	{/if}
</ViewportPortal>

<style>
	.ghost {
		position: absolute;
		left: 0;
		top: 0;
		min-width: 200px;
		background: var(--bg-elev-1);
		border: 1.5px dashed var(--accent);
		border-radius: var(--radius-md);
		color: var(--text);
		box-shadow: var(--shadow-1);
		opacity: 0.85;
		pointer-events: none;
		/* The ghost is purely visual — all input is handled by the window-level
		 * listeners in PlacementPreview itself. */
		user-select: none;
	}
	.header {
		display: flex;
		align-items: baseline;
		gap: 8px;
		padding: 6px 10px;
		background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 18%, transparent), transparent);
		border-bottom: 1px solid var(--border);
		border-radius: var(--radius-md) var(--radius-md) 0 0;
	}
	.health {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--accent);
		flex-shrink: 0;
		box-shadow: 0 0 5px currentColor;
	}
	.name {
		font-family: var(--font-mono);
		font-weight: 600;
		font-size: 12px;
		line-height: normal;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
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
		position: relative;
		display: flex;
		align-items: center;
		gap: 4px;
		min-height: 16px;
		padding: 1px 4px;
		font-family: var(--font-mono);
		color: var(--dtype, var(--text));
	}
	.port-row.out {
		justify-content: flex-end;
	}
	.pin {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--dtype, var(--bg-elev-3));
		border: 1px solid var(--dtype, var(--border-strong));
		flex-shrink: 0;
	}
	.pin.in {
		margin-left: -8px;
	}
	.pin.out {
		margin-right: -8px;
	}
	.viewers {
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 6px;
		background: var(--bg-elev-2);
		border-radius: 0 0 var(--radius-md) var(--radius-md);
	}
	.snap-guides {
		position: absolute;
		left: 0;
		top: 0;
		width: 1px;
		height: 1px;
		overflow: visible;
		pointer-events: none;
	}
</style>
