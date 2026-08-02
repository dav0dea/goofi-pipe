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
	import {
		PANE_AXES,
		coordOf,
		endsInDismiss,
		paneMin,
		paneSizeAt,
		type PaneAxis,
		type PaneDrag
	} from './paneDrag';

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

	/** A persisted pane size, or `null` when the user has never dragged one — in which case the
	 *  resting size is the CSS fallback (`var(--pane-w, 420px)` / `var(--pane-h, 60%)`). Null rather
	 *  than a literal because the portrait default is a PERCENTAGE of the host, which no number here
	 *  could express; keeping both defaults in the stylesheet is what lets one idiom serve both.
	 *
	 *  Not re-floored on the way out: what is written below is the RENDERED size, and the pane's
	 *  range cannot be empty (`--pane-min` is written into both ceilings), so a rendered size is
	 *  between the two bounds by construction — on this host and on any other. */
	function storedSize(axis: PaneAxis): number | null {
		try {
			const n = parseInt(localStorage.getItem(PANE_AXES[axis].key) ?? '', 10);
			return Number.isFinite(n) ? n : null;
		} catch {
			return null; // private mode; persistence is best-effort
		}
	}

	/** The pane's size on each axis, keyed exactly as `PANE_AXES` is — so the axis a drag is on
	 *  selects the state it writes, with no branch of its own. */
	let paneSize = $state({ x: storedSize('x'), y: storedSize('y') });
	let resizing = $state(false);
	/** The pane itself — the drag measures its RENDERED box, since the ceilings are CSS's. */
	let paneEl = $state<HTMLElement | null>(null);

	/** The in-flight resize's teardown; non-null only between pointerdown and its resolution. */
	let teardownResize: (() => void) | null = null;

	function startPanelResize(e: PointerEvent): void {
		if (!paneEl) return;
		e.preventDefault();
		const el = paneEl;
		// Both of the pane's own facts, read back off it in ONE question rather than re-derived here:
		// the axis is the container query's answer, and the floor is the stylesheet's — declared
		// beside the ceiling it bounds, because that is the only place either can be evaluated and
		// the only arrangement in which they cannot contradict each other. `dim` is the rest of what
		// the axis selects (which dimension sizes the pane, which key remembers it), so "orientation
		// picks the axis" stays one named fact instead of a condition re-asked at every line below.
		const css = getComputedStyle(el);
		const axis: PaneAxis = css.getPropertyValue('--pane-axis').trim() === 'y' ? 'y' : 'x';
		const dim = PANE_AXES[axis];
		// The RENDERED size, not the stored one: the ceiling lives in CSS, so a value restored from
		// a wider screen (or from before D-I6's cap) is not what is on screen — and a drag that began
		// from it would move the pointer a long way before the pane moved at all.
		const drag: PaneDrag = {
			startSize: dim.sizeOf(el.getBoundingClientRect()),
			startPos: coordOf(axis, e),
			min: paneMin(css.getPropertyValue('--pane-min'))
		};
		// Where the pointer got to. The pane CLAMPS at its floor, so the rendered size cannot say how
		// far a swipe was carried past it — and that overshoot is the whole of the dismiss decision.
		let at = drag.startPos;
		resizing = true;
		// The size is applied live, so there is nothing for a cancel to roll back — and persisting
		// it either way is what keeps a reload agreeing with what is on screen. A cancel that did NOT
		// run this teardown is the actual defect: the window listeners outlived the gesture and the
		// pane kept resizing on the next pointer motion.
		// What is persisted is again the RENDERED size, so the store can never drift above the cap.
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
				at = coordOf(axis, m);
				paneSize[axis] = paneSizeAt(drag, at);
			},
			commit: () => {
				// The swipe (D-I4), layered on the very gesture that resizes — same grip, same module,
				// one extra meaning, and only for a finger. Nothing is persisted: a dismiss is not a
				// resize, so the pane comes back the size it was. `onClose` unmounts it, which is what
				// discards the live shrink the swipe drew.
				if (endsInDismiss(drag, at, e.pointerType)) {
					resizing = false;
					teardownResize = null;
					onClose();
					return;
				}
				finish();
			},
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
		style:--pane-w={paneSize.x === null ? null : `${paneSize.x}px`}
		style:--pane-h={paneSize.y === null ? null : `${paneSize.y}px`}
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
		/* …and the FLOOR of this anchor, which is the OTHER half of one fact. It sits here, beside
		   the ceiling below, because a floor and a ceiling authored in two files are two answers to
		   one question and can cross: `paneDrag.ts` used to state 260px while `min(30%, 30rem)` on
		   the 854px host of a landscape phone resolved to 256px, so the pane's allowed range was
		   EMPTY — it could not be resized at all, and a drag started below its own floor with a
		   dismiss 40px away. Written INTO the ceiling below, that is unspellable. `startPanelResize`
		   reads this back exactly as it reads `--pane-axis`. */
		--pane-min: 260px;
		/* The RESTING size is this fallback — the inline `--pane-w` exists only once the user has
		   dragged one. Both defaults live here rather than in the script because portrait's is a
		   percentage of the host (see the container query), which no stored number could express. */
		width: var(--pane-w, 420px);
		/* The comfort cap in rem AND the small-screen guard in percent (D-I6), over the floor both
		   are measured against. The root clamp saturates at 14px across the whole range that
		   matters, so 30rem is 420px — exactly the resting width above, which is why the desktop
		   resting size does not move; below a ~1400px host the percent half binds instead. It
		   replaces `100% - --hit - --grip-reach`, which guaranteed only that ONE tap target of
		   canvas survived: on a 412px phone, a 44px strip. */
		max-width: clamp(var(--pane-min), 30%, 30rem);
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
	/* ─── THE RULE THIS PANE IS BUILT ON, and the one that must not regress ────────────────────
	   ORIENTATION decides only the AXIS the pane is anchored on — portrait a bottom sheet,
	   landscape/desktop the right-hand edge. INPUT MODALITY decides the GESTURE and its
	   AFFORDANCE — the edge drag is for a mouse AND a finger, the swipe is the finger's extra,
	   and the resting grabber is what a pointer with no hover needs in order to see the seam at
	   all. The two are INDEPENDENT: the modality logic is identical in portrait and in landscape.

	   So the grabber is stated here once and re-shaped exactly twice: once by MODALITY (the coarse
	   block below, which paints it and gives it a length), once by GEOMETRY (the portrait branch,
	   which says which edge the seam hugs and which of its dimensions carries that length — never
	   how it is painted). The portrait branch used to declare the pill itself, which is an
	   affordance chosen by ORIENTATION: one finger got a chunky pill standing up and a thin line
	   lying down, and a narrow docked desktop column got the touch grabber under a mouse. */
	.resize-handle::after {
		content: '';
		position: absolute;
		/* Hugs the pane's leading edge and runs the length of the seam. `margin: auto` is what
		   centres it when it is NOT that full length — i.e. once modality has made it a pill — so
		   neither axis needs a translate of its own. */
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
		/* …and the seam PAINTS at rest, as a grabber. Transparent-until-hover is the whole of its
		   discoverability on a fine pointer; with no hover the pane simply had an invisible edge,
		   and a hairline that IS painted still does not read as draggable (D-I9). MODALITY's call,
		   so it is made once here and holds on BOTH axes — the length lands on whichever dimension
		   the anchor made the long one, and the pill is the same pill either way. Quieter than the
		   accent the hover/drag state uses, so it reads as an edge rather than as active. */
		.resize-handle::after {
			--grab-len: 2.5rem;
			background: var(--border-strong);
			border-radius: 999px;
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
				/* This anchor's floor beside this anchor's ceiling — same pairing, same reason. */
				--pane-min: 160px;
				top: auto;
				left: 0;
				/* The soft keyboard, which CSS cannot see (D-I7): `--kb-inset` is published on <html>
				   by the device store, and a text field inside a bottom sheet is the case it was kept
				   for. This is its fourth consumer. */
				bottom: var(--kb-inset, 0px);
				width: auto;
				max-width: none;
				height: var(--pane-h, 60%);
				/* Over the floor, for the same reason the landscape ceiling is: whatever the host's
				   height, the sheet's range can never come out empty. On every geometry this app has
				   met 60% is the far larger of the two — which is exactly what the landscape anchor
				   assumed about ITS pair, and was wrong about. */
				max-height: max(60%, var(--pane-min));
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
			/* The seam turns with the anchor, and turning it is ALL this branch has to say about it:
			   which edge it hugs, and which of its dimensions carries the length. How it is painted
			   at rest is modality's answer above, which holds here unchanged — as do the hover/drag
			   accent rules, which out-specify both. */
			.resize-handle::after {
				inset: 3px 0 auto 0;
				width: var(--grab-len, 100%);
				height: 2px;
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
