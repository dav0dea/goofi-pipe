<!--
  Draggable divider between two children of a split. Converts pointer movement
  (px along the split axis) into a fractional delta against the parent split's
  measured size and reports it incrementally, so the store applies it to the
  live sizes with min-clamping. Same pointerdown→window-move/up pattern as the
  old side-panel resize handle.
-->
<script lang="ts">
	import type { Direction } from './model';
	import { beginDrag } from '$lib/ui';
	import { onDestroy } from 'svelte';

	let {
		direction,
		onResize,
		onCommit
	}: {
		direction: Direction;
		/** `containerPx` is the split's measured size along the axis — the pixel floor's denominator. */
		onResize: (deltaFraction: number, containerPx: number) => void;
		/** The gesture is over. A resize is continuous, so `onResize` only DRAWS — this is where the
		 * shares it drew become one command, and therefore one ctrl-Z. */
		onCommit: () => void;
	} = $props();

	let el = $state<HTMLDivElement | null>(null);
	let dragging = $state(false);
	/** The in-flight drag's teardown; non-null only between pointerdown and its resolution. */
	let teardown: (() => void) | null = null;

	function onPointerDown(e: PointerEvent): void {
		const container = el?.parentElement;
		if (!container || !el) return;
		e.preventDefault();
		const size = direction === 'row' ? container.clientWidth : container.clientHeight;
		if (size <= 0) return;
		let last = direction === 'row' ? e.clientX : e.clientY;
		dragging = true;
		document.body.style.cursor = direction === 'row' ? 'col-resize' : 'row-resize';

		// A seam draws its resize incrementally, so there is nothing to roll back — a cancelled drag
		// and a released one both just stop, and both COMMIT what they drew (a cancel is the pointer
		// being taken away, not the user changing their mind). What matters is that a cancel stops:
		// without it the listeners outlived the gesture and the next pointer motion kept resizing.
		const finish = (): void => {
			dragging = false;
			document.body.style.cursor = '';
			teardown = null;
			onCommit();
		};
		teardown = beginDrag(el, e.pointerId, {
			move: (m) => {
				const cur = direction === 'row' ? m.clientX : m.clientY;
				onResize((cur - last) / size, size);
				last = cur;
			},
			commit: finish,
			cancel: finish
		});
	}
	onDestroy(() => teardown?.());
</script>

<div
	bind:this={el}
	class="splitter {direction}"
	class:dragging
	onpointerdown={onPointerDown}
	role="separator"
	aria-orientation={direction === 'row' ? 'vertical' : 'horizontal'}
	tabindex="-1"
></div>

<style>
	.splitter {
		flex: 0 0 var(--splitter-size);
		position: relative;
		z-index: var(--z-chrome);
		touch-action: none;
	}
	.splitter.row {
		cursor: col-resize;
	}
	.splitter.column {
		cursor: row-resize;
	}
	/* Thin visible line, centered, brightening on hover/drag. The hit area is
	   the full splitter thickness so it stays easy to grab. */
	.splitter::after {
		content: '';
		position: absolute;
		background: var(--border);
		transition: background var(--dur-fast) var(--ease);
	}
	.splitter.row::after {
		top: 0;
		bottom: 0;
		left: 50%;
		width: 1px;
		transform: translateX(-50%);
	}
	.splitter.column::after {
		left: 0;
		right: 0;
		top: 50%;
		height: 1px;
		transform: translateY(-50%);
	}
	.splitter:hover::after,
	.splitter.dragging::after {
		background: var(--accent);
	}
	/* Touch: --splitter-size grows to 14px under the coarse floor, which is still 30px short of
	   --hit, and the `::after` above is the painted hairline, not a grip. So widen the HIT area
	   alone — an overlay centred on the seam, leaving both the line and the seam's LAYOUT thickness
	   (which the panels either side are measured against) exactly where they are. Axis-specific,
	   because a percentage `inset` resolves against the containing block on each axis: a symmetric
	   one would shrink the long axis to --hit instead of growing the short one.
	   The band does reach ~15px into each neighbouring panel; that is the trade a draggable seam
	   makes on touch, and the splitter already sits above panel content at --z-chrome. */
	@media (hover: none) and (pointer: coarse) {
		.splitter::before {
			content: '';
			position: absolute;
		}
		.splitter.row::before {
			inset: 0 calc((var(--hit) - 100%) / -2);
		}
		.splitter.column::before {
			inset: calc((var(--hit) - 100%) / -2) 0;
		}
	}
</style>
