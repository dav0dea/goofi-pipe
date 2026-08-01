<!--
  Selection inspector for a single editor panel. Slides in from the right edge
  of its host editor (not the whole window) when that editor has exactly one
  node selected, showing its parameters, errors, and metadata. Width is
  drag-resizable.

  Visibility is per-editor: `enabled` is owned by the host NodeEditorPanel, which offers it two
  ways — this pane's own dismiss ✕, and the editor's `inspector-toggle` corner control, which the
  host hides while the pane covers it. There is no global header toggle.

  The ✕ is the whole of D-R9's R1 half. Until R the pane had NO way out of its own: the toggle sat
  under it in z-order, and deselecting meant tapping canvas the pane covered — a dead end on a
  phone, and merely obscure on a desktop, which is the same defect (C9: the comment here promised
  a close button that the markup never had).

  This is additive to the placeable Parameters/Metadata/Errors panels — those
  follow the active editor's selection instead.
-->
<script lang="ts">
	import ParamForm from '$lib/inspector/ParamForm.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { beginDrag, Button, IconButton, ScrollArea } from '$lib/ui';
	import { graph } from '$lib/stores/graph.svelte';
	import { onDestroy } from 'svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { coordOf, paneSizeAt, type PaneAxis, type PaneDrag } from './paneDrag';

	let {
		node,
		enabled,
		onClose
	}: {
		node: NodeInstanceInfo | null;
		enabled: boolean;
		/** Turn this editor's inspector off — the same switch the corner toggle flips, so there is
		 * one piece of state and two doors onto it, never two states. */
		onClose: () => void;
	} = $props();

	// Respawn the selected node with the same params + links. The inspector's
	// error section is the home for this since the dockable Errors panel (which
	// used to carry it) was removed.
	function restart(): void {
		if (!node) return;
		void graph()
			.restartNode(node.uid)
			.catch((e) => console.warn('restart failed', e));
	}

	/** The pane's FLOOR on each axis. The ceilings are CSS's (`max-width` / `max-height` below):
	 *  they are host- and rem-relative, so the stylesheet is the only place that can evaluate them,
	 *  and a number here would be a second answer to one question — which is exactly what the old
	 *  `MAX_PANEL_WIDTH = 720` was, sitting above a host clamp that always bound first. */
	const MIN_PANEL_WIDTH = 260;
	const MIN_PANEL_HEIGHT = 160;

	/** A persisted pane size, or `null` when the user has never dragged one — in which case the
	 *  resting size is the CSS fallback (`var(--pane-w, 420px)` / `var(--pane-h, 60%)`). Null rather
	 *  than a literal because the portrait default is a PERCENTAGE of the host, which no number here
	 *  could express; keeping both defaults in the stylesheet is what lets one idiom serve both. */
	function storedSize(key: string, min: number): number | null {
		try {
			const n = parseInt(localStorage.getItem(key) ?? '', 10);
			return Number.isFinite(n) ? Math.max(min, n) : null;
		} catch {
			return null; // private mode; persistence is best-effort
		}
	}

	let panelWidth = $state(storedSize('goofi.panelWidth', MIN_PANEL_WIDTH));
	let panelHeight = $state(storedSize('goofi.panelHeight', MIN_PANEL_HEIGHT));
	let resizing = $state(false);
	/** The pane itself — the drag measures its RENDERED box, since the ceilings are CSS's. */
	let paneEl = $state<HTMLElement | null>(null);

	/** The in-flight resize's teardown; non-null only between pointerdown and its resolution. */
	let teardownResize: (() => void) | null = null;

	function startPanelResize(e: PointerEvent): void {
		if (!paneEl) return;
		e.preventDefault();
		const el = paneEl;
		// The axis is the container query's answer, read back off the pane rather than re-derived
		// here from the host's box — one question, asked once, of the box CSS asked it of.
		const axis: PaneAxis =
			getComputedStyle(el).getPropertyValue('--pane-axis').trim() === 'y' ? 'y' : 'x';
		const vertical = axis === 'y';
		const box = el.getBoundingClientRect();
		// The RENDERED size, not the stored one: the ceiling lives in CSS, so a value restored from
		// a wider screen (or from before D-I6's cap) is not what is on screen — and a drag that began
		// from it would move the pointer a long way before the pane moved at all.
		const drag: PaneDrag = {
			axis,
			startSize: vertical ? box.height : box.width,
			startPos: coordOf(axis, e),
			min: vertical ? MIN_PANEL_HEIGHT : MIN_PANEL_WIDTH
		};
		const key = vertical ? 'goofi.panelHeight' : 'goofi.panelWidth';
		const apply = (size: number): void => {
			if (vertical) panelHeight = size;
			else panelWidth = size;
		};
		resizing = true;
		// The size is applied live, so there is nothing for a cancel to roll back — and persisting
		// it either way is what keeps a reload agreeing with what is on screen. A cancel that did NOT
		// run this teardown is the actual defect: the window listeners outlived the gesture and the
		// pane kept resizing on the next pointer motion.
		// What is persisted is again the RENDERED size, so the store can never drift above the cap.
		const finish = (): void => {
			resizing = false;
			teardownResize = null;
			const r = el.getBoundingClientRect();
			const size = vertical ? r.height : r.width;
			apply(size);
			try {
				localStorage.setItem(key, String(Math.round(size)));
			} catch {
				/* private mode; persistence is best-effort */
			}
		};
		teardownResize = beginDrag(e.currentTarget as HTMLElement, e.pointerId, {
			move: (m) => apply(paneSizeAt(drag, coordOf(axis, m))),
			commit: finish,
			cancel: finish
		});
	}
	onDestroy(() => teardownResize?.());
</script>

{#if enabled}
	<!-- `inert` while there is nothing to inspect. The pane stays MOUNTED and parked off-screen so
	     the slide plays on the next selection, and `pointer-events: none` already keeps a pointer
	     out — but that says nothing about focus or the accessibility tree, so this ✕ was Tab-reachable
	     in any layout where the editor is not the active panel, and an AT virtual cursor found it in
	     every one. It flips state the user cannot see. -->
	<aside
		class="side-panel"
		class:open={node !== null}
		class:resizing
		inert={node === null}
		bind:this={paneEl}
		style:--pane-w={panelWidth === null ? null : `${panelWidth}px`}
		style:--pane-h={panelHeight === null ? null : `${panelHeight}px`}
		data-testid="auto-side-panel"
	>
		<!-- The size arrives as CUSTOM PROPERTIES rather than as a `width`, because which axis it
		     feeds is the container query's decision below and an inline `width` cannot be beaten by
		     any query. No `aria-orientation`: this separator takes no keyboard, and the axis it would
		     name is owned by CSS — an attribute asserting one of them would be wrong half the time,
		     and re-deriving the orientation in JS to keep it true would be the second source of
		     truth D-I2 exists to avoid. -->
		<div
			class="resize-handle"
			role="separator"
			aria-label="Resize side panel"
			onpointerdown={startPanelResize}
			data-testid="panel-resize-handle"
		></div>
		<div class="ins-head">
			<IconButton
				variant="ghost"
				density="chrome"
				class="ins-close"
				label="Close inspector"
				title="Close the inspector"
				data-testid="inspector-close"
				onclick={onClose}>✕</IconButton
			>
		</div>
		<ScrollArea>
			<ParamForm {node} />
			{#if node && !node.subpatch}
				<MetadataPanel {node} />
				{#if node.error}
					<section class="node-error" data-testid="inspector-error">
						<div class="err-head">
							<header>Error</header>
							<Button
								variant="danger"
								size="sm"
								onclick={restart}
								title="Restart this node (respawn with the same params + links)"
								data-testid="inspector-restart">↻ Restart</Button
							>
						</div>
						<pre>{node.error}</pre>
					</section>
				{/if}
			{/if}
		</ScrollArea>
	</aside>
{/if}

<style>
	.side-panel {
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		/* Which axis the grip drags. The container query at the bottom of this file is what DECIDES
		   the anchor; `startPanelResize` reads this property back instead of re-deriving the
		   orientation in JS, so the question is asked once, of the box CSS asked it of. */
		--pane-axis: x;
		/* The RESTING size is this fallback — the inline `--pane-w` exists only once the user has
		   dragged one. Both defaults live here rather than in the script because portrait's is a
		   percentage of the host (see the container query), which no stored number could express. */
		width: var(--pane-w, 420px);
		/* The comfort cap in rem AND the small-screen guard in percent (D-I6). The root clamp
		   saturates at 14px across the whole range that matters, so 30rem is 420px — exactly the
		   resting width above, which is why the desktop resting size does not move; below a ~1400px
		   host the percent half binds instead. It replaces `100% - --hit - --grip-reach`, which
		   guaranteed only that ONE tap target of canvas survived: on a 412px phone, a 44px strip. */
		max-width: min(30%, 30rem);
		background: color-mix(in srgb, var(--surface-1) 96%, transparent);
		backdrop-filter: blur(8px);
		border-left: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		min-width: 0;
		transform: translateX(100%);
		transition: transform var(--dur-slow) var(--ease);
		box-shadow: var(--shadow-side);
		z-index: var(--z-side-panel);
	}
	.side-panel.open {
		transform: translateX(0);
	}
	/* When closed the panel is parked off-screen; its resize handle would
	   otherwise still poke back into view at the right edge and show a
	   col-resize cursor. Disable all interaction until it's actually open. */
	.side-panel:not(.open) {
		pointer-events: none;
	}
	.side-panel.resizing {
		transition: none;
	}
	.side-panel.resizing * {
		user-select: none;
	}
	/* The pane's own dismiss strip. A row rather than a floating corner button: the pane's top-right
	   is where ParamForm's identity Bar puts the node's state badge, and a control laid over that
	   would read as belonging to it. Right-aligned and background-free, so it reads as the pane's
	   own title bar rather than as a third surface rung. */
	.ins-head {
		display: flex;
		justify-content: flex-end;
		flex: 0 0 auto;
		padding: var(--space-2) var(--space-2) 0;
	}
	.ins-head :global(.ins-close) {
		--icon-btn-size: 22px;
		color: var(--text-dim);
	}
	.ins-head :global(.ins-close:hover) {
		color: var(--text);
	}
	/* Current processing error for the selected node — a simple snapshot, shown
	   only while the node is errored, after the metadata section. */
	.node-error {
		padding: var(--space-6);
		border-top: 1px solid var(--border);
		background: var(--surface-1);
	}
	.err-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-5);
		margin-bottom: var(--space-3);
	}
	.node-error header {
		font-weight: 600;
		font-size: var(--fs-small);
		color: var(--danger);
	}
	.node-error pre {
		font-size: var(--fs-micro);
		color: var(--text-dim);
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
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
	.resize-handle::after {
		content: '';
		position: absolute;
		left: 3px;
		top: 0;
		bottom: 0;
		width: 2px;
		background: transparent;
		transition: background var(--dur-fast) var(--ease);
	}
	.resize-handle:hover::after,
	.side-panel.resizing .resize-handle::after {
		background: var(--accent);
	}
	/* Touch: an 8px seam is under a fifth of --hit. Widen the HIT area only (a `::before` overlay,
	   since `::after` is already the painted line), so the line stays at its 3px offset and the
	   panel's own size is unaffected. One axis only — the handle already spans the other, and a
	   symmetric percentage inset would shrink that to --hit.
	   ASYMMETRIC, though, and this is the trade the way `Splitter.svelte` states its own: a centred
	   band reaches 22px back over the escape strip the clamp reserves, which on the 412px project IS
	   the strip's natural aim point — a reserved target quietly halved. So the band leans inward,
	   over the pane's own rows, and what it does take outward comes out of the pane rather than out
	   of the strip. Full --hit either way; the strip stays live. */
	@media (hover: none) and (pointer: coarse) {
		.resize-handle::before {
			content: '';
			position: absolute;
			/* 12px out + 8px handle + 24px in = --hit. */
			inset: 0 -24px 0 -12px;
		}
		/* …and the seam PAINTS at rest. Transparent-until-hover is the whole of its discoverability
		   on a fine pointer; with no hover the pane simply had an invisible edge. Quieter than the
		   accent the hover/drag state uses, so it reads as an edge rather than as active. */
		.resize-handle::after {
			background: var(--border-strong);
		}
	}

	/* ---------------------------------------------------------------------------------------------
	   PORTRAIT — the bottom sheet, and the WHOLE of it. Everything that differs is under this one
	   prelude, so the anchor has exactly one place to be read and one place to be changed.

	   The question is asked of the HOST PANEL (`workspace/Panel.svelte`'s `.panel-body` is the query
	   container), not of the viewport and not of the device class: "is the surface I live in taller
	   than it is wide?" That is the only question that matters, and it is right in the two cases a
	   viewport query gets wrong in opposite directions — a desktop editor docked as a narrow tall
	   column gets the sheet, and a wide short editor inside a tall window does not.

	   THE FLIP SEAM (spec §6). The desktop consequence is deliberate but visible, and if the user
	   takes it back the edit is the `@media all` line below and nothing else: make it
	   `@media (hover: none) and (pointer: coarse)` and the sheet is touch-only again. The no-op
	   wrapper exists to keep that a one-line change — a container query cannot carry a pointer
	   feature itself, so without it the flip would mean re-nesting the whole block.

	   Nothing here is gated on the pointer, and that is the point: one component, one continuous
	   resize, one persistence idiom. Modality gates only the additive swipe. */
	@media all {
		@container (orientation: portrait) {
			.side-panel {
				--pane-axis: y;
				top: auto;
				left: 0;
				/* The soft keyboard, which CSS cannot see (D-I7): `--kb-inset` is published on <html>
				   by the device store, and a text field inside a bottom sheet is the case it was kept
				   for. This is its fourth consumer. */
				bottom: var(--kb-inset, 0px);
				width: auto;
				max-width: none;
				height: var(--pane-h, 60%);
				max-height: 60%;
				padding-bottom: var(--safe-bottom);
				border-left: none;
				border-top: 1px solid var(--border);
				box-shadow: var(--shadow-sheet);
				/* D-I8: the same motion tokens as the X slide above, a quarter turn round. The
				   `:not(.open)` park rule and its `pointer-events: none` are axis-free and stand. */
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
			/* D-I9: a RESTING grabber, not a seam that appears on hover. The desktop line is
			   transparent until hovered, which CLAUDE.md forbids as the whole of an affordance, and a
			   sheet with no visible grabber does not read as draggable at all. Same element, turned
			   into a pill — the hover/drag accent rules above still out-specify it, so the drag still
			   lights up. */
			.resize-handle::after {
				left: 50%;
				top: 3px;
				bottom: auto;
				width: 2.5rem;
				height: 2px;
				transform: translateX(-50%);
				background: var(--border-strong);
				border-radius: 999px;
			}
			/* The coarse hit band turns with it — but it does NOT keep the landscape band's inward
			   lean. 36px out + 8px handle + 0 in = --hit, entirely in the gutter above the sheet.
			   Inward is where the pane's own top row is, and in this anchor that row is `.ins-head`:
			   the band's 24px reach landed on the ✕ (measured: band bottom 418px, ✕ centre 416px) and
			   swallowed the pane's one pointer door, which D-I4 says must never depend on a gesture.
			   What it takes outward comes out of the 40% of canvas the cap hands back, where 36px is
			   an eighth of what is there — the opposite of the landscape case, where outward is the
			   scarce side. The band always leans away from whatever is scarce. */
			@media (hover: none) and (pointer: coarse) {
				.resize-handle::before {
					inset: -36px 0 0 0;
				}
			}
		}
	}
</style>
