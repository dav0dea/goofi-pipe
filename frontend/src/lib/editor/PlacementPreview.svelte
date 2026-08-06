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
	import { createTouchPlacement, ghostOrigin, type GhostAnchor } from './touchPlacement';
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
	/** Which point of the ghost the input driving it is holding — asked of the EVENT, below, so a
	 *  hybrid device answers separately for its trackpad and its screen. */
	let anchor = $state<GhostAnchor>('top-left');

	const flowPos = $derived(screenToFlowPosition({ x: mouseClient.x, y: mouseClient.y }));
	// The anchor is spent HERE, once, on the flow position — everything downstream (the snap bounds,
	// the transform, the commit) reads `origin` and so cannot disagree about where the ghost is.
	const origin = $derived(ghostOrigin(flowPos, { w: ghostW, h: ghostH }, anchor));

	const snap = $derived.by(() => {
		const dragged = [makeBounds(origin.x, origin.y, ghostW, ghostH)];
		return computeSnapDelta(dragged, targets, altKey);
	});

	const snappedX = $derived(origin.x + snap.dx);
	const snappedY = $derived(origin.y + snap.dy);

	const inputs = $derived(Object.entries(typeInfo.input_slots));
	const outputs = $derived(Object.entries(typeInfo.output_slots));

	function onMouseMove(e: MouseEvent): void {
		mouseClient = { x: e.clientX, y: e.clientY };
		altKey = e.altKey;
		// A cursor is back on the ghost, so it hangs off its corner again — the same question the
		// touch path asks, answered by the other input.
		anchor = 'top-left';
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

	// --- the same placement, with a finger ---------------------------------------------------
	// The mouse path above is the reference and is untouched: everything below is reached only by a
	// pointer whose `pointerType` is `touch`, asked per EVENT so a hybrid device stays right on both
	// of its inputs. The gesture itself lives in `touchPlacement.ts`, where a unit test can reach it.
	const touch = createTouchPlacement();

	function onPointerDown(e: PointerEvent): void {
		const at = touch.down(e, inCanvas(e.target));
		if (!at) return;
		// `down` answers non-null for a touch and nothing else, so this IS the modality gate — and it
		// is the gesture, not the device, that decides: the flag then rides through `up`, whose commit
		// must read the same anchor the ghost was last drawn with.
		anchor = 'centre';
		mouseClient = at;
	}

	function onPointerMove(e: PointerEvent): void {
		const at = touch.move(e);
		if (at) mouseClient = at;
	}

	function onPointerUp(e: PointerEvent): void {
		const at = touch.up(e);
		if (!at) return;
		mouseClient = at;
		// `snappedX/Y` are `$derived`, i.e. pull-based — reading them here re-evaluates them against
		// the point just assigned, so the commit lands on exactly the snapped position the ghost was
		// drawn at, through the same `computeSnapDelta` the mouse path and the node drag share.
		onCommit([Math.round(snappedX), Math.round(snappedY)]);
	}

	/**
	 * The pan block AND the synthetic-click suppression — one handler, because they are one problem.
	 *
	 * SvelteFlow pans by d3-zoom, which binds `touchstart` on the pane rather than `pointerdown`, so
	 * the pointer handlers above cannot keep a drag off the panner however they are written. This is
	 * the touch analogue of the `mousedown` blocker: `stopPropagation` is what stops the pan.
	 *
	 * `preventDefault` is the other half, and it is HOW THE SYNTHETIC CLICK IS SUPPRESSED. A touch
	 * the page does not cancel replays afterwards as a compat mouse cascade — mousemove, mousedown,
	 * mouseup, click — and by the time it arrives `pointerup` has already committed and torn this
	 * component down, so the trailing click would land on SvelteFlow's pane and clear the selection
	 * the commit just made. Cancelling `touchstart` suppresses that whole cascade at its source,
	 * which is why there is no click guard here racing the unmount to swallow one event.
	 *
	 * Gated on the flag the pointer path already set, so it holds back exactly the gesture this
	 * component is driving: a touch OUTSIDE the canvas is left alone, and its compat click still
	 * reaches `onWindowClick` and cancels the placement precisely as a mouse click does.
	 */
	function onTouchStart(e: TouchEvent): void {
		if (!touch.active) return;
		e.stopPropagation();
		e.preventDefault();
	}

	$effect(() => {
		window.addEventListener('mousemove', onMouseMove);
		window.addEventListener('keydown', onKeyDown, true);
		window.addEventListener('click', onWindowClick, true);
		window.addEventListener('mousedown', onWindowMouseDown, true);
		window.addEventListener('pointerdown', onPointerDown, true);
		window.addEventListener('pointermove', onPointerMove, true);
		window.addEventListener('pointerup', onPointerUp, true);
		window.addEventListener('pointercancel', touch.cancel, true);
		// `passive: false` is load-bearing: Chromium makes window-level touchstart listeners passive
		// by default, which would drop the `preventDefault` above — and the click suppression with it.
		window.addEventListener('touchstart', onTouchStart, { capture: true, passive: false });
		return () => {
			window.removeEventListener('mousemove', onMouseMove);
			window.removeEventListener('keydown', onKeyDown, true);
			window.removeEventListener('click', onWindowClick, true);
			window.removeEventListener('mousedown', onWindowMouseDown, true);
			window.removeEventListener('pointerdown', onPointerDown, true);
			window.removeEventListener('pointermove', onPointerMove, true);
			window.removeEventListener('pointerup', onPointerUp, true);
			window.removeEventListener('pointercancel', touch.cancel, true);
			window.removeEventListener('touchstart', onTouchStart, true);
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

	/* A fine pointer has a CURSOR, and the ghost riding it is most of what says "not placed yet" —
	   so 0.9 and a hairline dash are enough there. A finger has no cursor at all: the drawing is the
	   only thing left to say it, and at 0.9 on a phone this was a node card with a green edge. So a
	   coarse pointer gets the placeholder stated three ways at once — visibly see-through, the same
	   heavy dash `.node-drop-hint` uses for "something goes here", and NO elevation, because a node
	   that is not down yet has not landed on the surface. Desktop is byte-identical. */
	@media (hover: none) and (pointer: coarse) {
		.ghost {
			opacity: 0.6;
		}
		.surface {
			border-width: 2px;
			box-shadow: none;
		}
	}
</style>
