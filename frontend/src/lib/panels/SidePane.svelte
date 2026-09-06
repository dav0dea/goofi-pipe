<!-- SidePane — a drag-resizable pane anchored to the host panel's edge: the right edge, or the
     bottom when the host is portrait. It stays MOUNTED and parked when closed, so open and close
     are the same visible slide; `onClosed` fires when the closing slide has finished. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { beginDrag } from 'panelty';
	import { onDestroy } from 'svelte';
	import { PANE_AXES, coordOf, paneSizeAt, type PaneAxis, type PaneDrag } from './paneDrag';

	let {
		open,
		onClosed,
		testid = 'side-pane',
		children
	}: {
		open: boolean;
		onClosed?: () => void;
		testid?: string;
		children?: Snippet;
	} = $props();

	/** A persisted pane size, or `null` — the resting size is then the stylesheet's own `clamp()`. */
	function storedSize(axis: PaneAxis): number | null {
		try {
			const n = parseInt(localStorage.getItem(PANE_AXES[axis].key) ?? '', 10);
			return Number.isFinite(n) ? n : null;
		} catch {
			return null; // private mode; persistence is best-effort
		}
	}

	/** Keyed exactly as `PANE_AXES` is, so a drag's axis selects the state it writes. */
	let paneSize = $state({ x: storedSize('x'), y: storedSize('y') });
	let resizing = $state(false);
	let paneEl = $state<HTMLElement | null>(null);

	function finishTransition(e: TransitionEvent): void {
		if (e.target !== e.currentTarget || e.propertyName !== 'transform' || open) return;
		onClosed?.();
	}

	/** The in-flight resize's teardown; non-null only between pointerdown and its resolution. */
	let teardownResize: (() => void) | null = null;

	function startResize(e: PointerEvent): void {
		if (!paneEl) return;
		e.preventDefault();
		const el = paneEl;
		// Read back off the pane rather than re-derived: the axis is the container query's answer.
		const axis: PaneAxis =
			getComputedStyle(el).getPropertyValue('--pane-axis').trim() === 'y' ? 'y' : 'x';
		const dim = PANE_AXES[axis];
		// The RENDERED size, not the stored one: the bounds live in CSS, so a value restored from a
		// wider screen is not what is on screen.
		const drag: PaneDrag = {
			startSize: dim.sizeOf(el.getBoundingClientRect()),
			startPos: coordOf(axis, e)
		};
		resizing = true;
		// The RENDERED size is what is persisted, so the store can never drift outside CSS's bounds.
		const finish = (): void => {
			resizing = false;
			teardownResize = null;
			const size = dim.sizeOf(el.getBoundingClientRect());
			paneSize[axis] = size;
			try {
				localStorage.setItem(dim.key, String(Math.round(size)));
			} catch {
				/* private mode; persistence is best-effort */
			}
		};
		teardownResize = beginDrag(e.currentTarget as HTMLElement, e.pointerId, {
			move: (m) => {
				paneSize[axis] = paneSizeAt(drag, coordOf(axis, m));
			},
			// One resolution for both: the size is applied live, so commit and cancel agree.
			commit: finish,
			cancel: finish
		});
	}
	onDestroy(() => teardownResize?.());
</script>

<aside
	class="side-panel"
	class:open
	class:resizing
	inert={!open}
	bind:this={paneEl}
	ontransitionend={finishTransition}
	style:--pane-w={paneSize.x === null ? null : `${paneSize.x}px`}
	style:--pane-h={paneSize.y === null ? null : `${paneSize.y}px`}
	data-testid={testid}
>
	<!-- The size arrives as CUSTOM PROPERTIES: which axis it feeds is the container query's
	     decision below, and an inline `width` cannot be beaten by any query. -->
	<div
		class="resize-handle"
		role="separator"
		aria-label="Resize side panel"
		onpointerdown={startResize}
		data-testid="panel-resize-handle"
	></div>
	{@render children?.()}
</aside>

<style>
	.side-panel {
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		/* Which axis the grip drags; the container query below decides it and JS reads it back. */
		--pane-axis: x;
		/* Floor, resting size and ceiling in ONE declaration, so the bounds cannot cross. They are
		   HOST-relative, never `vw`/`vh`: the pane answers to its panel, not to the screen. */
		width: clamp(10%, var(--pane-w, min(40%, 30rem)), 90%);
		/* `inline-size`, never `size`: an (orientation:) query needs the block axis uncontained, or
		   the portrait branch below would answer against the pane instead of `.panel-body`. */
		container-type: inline-size;
		background: color-mix(in srgb, var(--surface-1) 96%, transparent);
		backdrop-filter: blur(8px);
		border-left: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		min-width: 0;
		transform: translateX(100%);
		transition:
			transform var(--dur-slow) var(--ease),
			visibility 0s;
		box-shadow: var(--shadow-side);
		z-index: var(--z-side-panel);
	}
	.side-panel.open {
		transform: translateX(0);
	}
	/* A host rotation swaps the park transform from X to Y, so an unpainted park is what stops a
	   ghost flying diagonally across it. */
	.side-panel:not(.open) {
		visibility: hidden;
		pointer-events: none;
		/* Hold visibility through the outgoing transform, then hide on its final frame. */
		transition:
			transform var(--dur-slow) var(--ease),
			visibility var(--dur-slow) step-end;
	}
	.side-panel.resizing {
		transition: none;
	}
	.side-panel.resizing * {
		user-select: none;
	}
	.resize-handle {
		position: absolute;
		left: -4px;
		top: 0;
		bottom: 0;
		width: 8px;
		cursor: col-resize;
		z-index: 1;
		touch-action: none;
	}
	/* The grabber is stated once here, re-shaped by MODALITY (the coarse block below, which paints
	   it) and by GEOMETRY (the portrait branch, which turns it) — never by both at once. */
	.resize-handle::after {
		content: '';
		position: absolute;
		/* `margin: auto` centres it once modality has made it a pill, on whichever axis it is on. */
		inset: 0 auto 0 3px;
		margin: auto;
		width: 2px;
		height: var(--grab-len, 100%);
		background: transparent;
		transition: background var(--dur-fast) var(--ease);
	}
	.resize-handle:hover::after,
	.side-panel.resizing .resize-handle::after {
		background: var(--accent);
	}
	/* Touch: an 8px seam is under a fifth of --hit, so a `::before` widens the HIT area alone. It
	   leans OUTWARD, over the host: inward is the pane's first column, which carries its leading
	   control, and a band over a control is a control no finger can reach. */
	@media (hover: none) and (pointer: coarse) {
		.resize-handle::before {
			content: '';
			position: absolute;
			/* 36px out + 8px handle = --hit. */
			inset: 0 0 0 -36px;
		}
		/* …and the seam PAINTS at rest, as a grabber: with no hover there is no other affordance. */
		.resize-handle::after {
			--grab-len: 2.5rem;
			background: var(--border-strong);
			border-radius: 999px;
		}
	}

	/* PORTRAIT — the bottom sheet, and the whole of it. The question is asked of the HOST PANEL, not
	   of the viewport. The `@media all` no-op keeps a touch-only flip a one-line edit. */
	@media all {
		@container (orientation: portrait) {
			.side-panel {
				--pane-axis: y;
				top: auto;
				left: 0;
				/* `--kb-inset` is the soft keyboard's overlap, which CSS cannot see. */
				bottom: var(--kb-inset, 0px);
				width: auto;
				height: clamp(10%, var(--pane-h, 60%), 90%);
				padding-bottom: var(--safe-bottom);
				border-left: none;
				border-top: 1px solid var(--border);
				box-shadow: var(--shadow-sheet);
				transform: translateY(100%);
			}
			.side-panel.open {
				transform: translateY(0);
			}
			.resize-handle {
				left: 0;
				right: 0;
				top: -4px;
				bottom: auto;
				width: auto;
				height: 8px;
				cursor: row-resize;
			}
			/* Turning the seam is ALL this branch says about it; how it PAINTS is modality's above. */
			.resize-handle::after {
				inset: 3px 0 auto 0;
				width: var(--grab-len, 100%);
				height: 2px;
			}
			/* The band turns; it leans outward on both geometries, away from the pane's own controls. */
			@media (hover: none) and (pointer: coarse) {
				.resize-handle::before {
					inset: -36px 0 0 0;
				}
			}
		}
	}
</style>
