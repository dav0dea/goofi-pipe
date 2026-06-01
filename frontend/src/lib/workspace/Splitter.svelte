<!--
  Draggable divider between two children of a split. Converts pointer movement
  (px along the split axis) into a fractional delta against the parent split's
  measured size and reports it incrementally, so the store applies it to the
  live sizes with min-clamping. Same pointerdown→window-move/up pattern as the
  old side-panel resize handle.
-->
<script lang="ts">
	import type { Direction } from './model';

	let {
		direction,
		onResize
	}: {
		direction: Direction;
		onResize: (deltaFraction: number) => void;
	} = $props();

	let el = $state<HTMLDivElement | null>(null);
	let dragging = $state(false);

	function onPointerDown(e: PointerEvent): void {
		const container = el?.parentElement;
		if (!container) return;
		e.preventDefault();
		const size = direction === 'row' ? container.clientWidth : container.clientHeight;
		if (size <= 0) return;
		let last = direction === 'row' ? e.clientX : e.clientY;
		dragging = true;
		document.body.style.cursor = direction === 'row' ? 'col-resize' : 'row-resize';

		const onMove = (m: PointerEvent): void => {
			const cur = direction === 'row' ? m.clientX : m.clientY;
			onResize((cur - last) / size);
			last = cur;
		};
		const onUp = (): void => {
			dragging = false;
			document.body.style.cursor = '';
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
		};
		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}
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
		flex: 0 0 var(--splitter-size, 6px);
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
		transition: background 100ms ease;
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
</style>
