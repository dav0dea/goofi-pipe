<script lang="ts">
	import { untrack } from 'svelte';
	import { ViewportPortal, useSvelteFlow } from '@xyflow/svelte';
	import { dtypeColor } from './categoryColor';
	import SnapGuides from './SnapGuides.svelte';
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
		/** Snap-target bounds (flow coords) of every node currently on screen — the
		 * SAME set the drag snap uses, so a new node snaps to real nodes, sub-patch
		 * instances, and boundary pills alike (not just the real nodes in g.nodes). */
		targets: Bounds[];
		onCommit: (pos: [number, number]) => void;
		onCancel: () => void;
	}

	let { typeInfo, initialClient, targets, onCommit, onCancel }: Props = $props();

	const { screenToFlowPosition } = useSvelteFlow();

	let mouseClient = $state<{ x: number; y: number }>(untrack(() => ({ x: initialClient.x, y: initialClient.y })));
	let altKey = $state(false);
	let ghostW = $state(DEFAULT_NODE_W);
	let ghostH = $state(DEFAULT_NODE_H);

	const flowPos = $derived(screenToFlowPosition({ x: mouseClient.x, y: mouseClient.y }));

	const snap = $derived.by(() => {
		const dragged = [makeBounds(flowPos.x, flowPos.y, ghostW, ghostH)];
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
		style="transform: translate({snappedX}px, {snappedY}px); min-height: calc(var(--node-header) + {Math.max(inputs.length, 1)} * var(--node-u));"
		data-testid="placement-ghost"
	>
		<!-- Mirror of GoofiNode: a clipped surface plus an unclipped pin overlay. -->
		<div class="surface">
			<div class="header">
				<span class="health"></span>
				<span class="name">{typeInfo.type}</span>
			</div>
			{#if outputs.length > 0}
				<div class="viewers">
					{#each outputs as [slot, dtype] (slot)}
						<div class="slot-row" style="--dtype: {dtypeColor(dtype)};">
							<span class="slot-name">{slot}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
		<div class="ports">
			{#each inputs as [slot, dtype], i (slot)}
				<span
					class="pin in"
					style="top: calc(1px + var(--node-header) + var(--node-u) * {i + 0.5}); --dtype: {dtypeColor(dtype)};"
				></span>
			{/each}
			{#each outputs as [slot, dtype], i (slot)}
				<span
					class="pin out"
					style="top: calc(1px + var(--node-header) + var(--node-u) * {i + 0.5}); --dtype: {dtypeColor(dtype)};"
				></span>
			{/each}
		</div>
	</div>

	{#if snap.guides.length > 0}
		<SnapGuides guides={snap.guides} testid="placement-snap-guides" />
	{/if}
</ViewportPortal>

<style>
	.ghost {
		position: absolute;
		left: 0;
		top: 0;
		display: flex;
		flex-direction: column;
		width: var(--node-w);
		color: var(--text);
		opacity: 0.9;
		pointer-events: none;
		font-family: var(--font-mono);
		/* The ghost is purely visual — all input is handled by the window-level
		 * listeners in PlacementPreview itself. */
		user-select: none;
	}
	.surface {
		flex: 1 1 auto;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		background: var(--surface-1);
		border: 1.5px dashed var(--accent);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-1);
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
	}
	.health {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--accent);
		flex-shrink: 0;
	}
	.name {
		font-weight: 600;
		font-size: 12px;
		line-height: normal;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.viewers {
		display: flex;
		flex-direction: column;
	}
	.slot-row {
		height: var(--node-u);
		box-sizing: border-box;
		border-top: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: flex-end;
		padding: 0 12px 0 8px;
		font-size: 10px;
		color: var(--dtype, var(--text-dim));
	}
	.viewers .slot-row:first-child {
		border-top: none;
	}
	.ports {
		position: absolute;
		inset: 0;
	}
	.pin {
		position: absolute;
		width: 14px;
		height: 9px;
		border-radius: 999px;
		background: var(--dtype, var(--border-strong));
		box-shadow: 0 0 0 2px var(--surface-1);
	}
	.pin.in {
		left: 0;
		transform: translate(-50%, -50%);
	}
	.pin.out {
		right: 0;
		transform: translate(50%, -50%);
	}
</style>
