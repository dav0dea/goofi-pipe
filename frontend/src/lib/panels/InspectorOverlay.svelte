<!--
  Selection inspector for a single editor panel. Slides in from the right edge
  of its host editor (not the whole window) when that editor has exactly one
  node selected, showing its parameters, errors, and metadata. Width is
  drag-resizable.

  Visibility is per-editor: `enabled` is owned by the host NodeEditorPanel, which offers it two
  ways — this pane's own dismiss ✕, and the editor's `inspector-toggle` corner control, which the
  host hides while the pane covers it. There is no global header toggle.

  The ✕ is the whole of D-R9's R1 half, and now the ONLY way out of the pane. Until R there was
  none: the toggle sat under it in z-order, and deselecting meant tapping canvas the pane covered —
  a dead end on a phone, and merely obscure on a desktop, which is the same defect (C9: the comment
  here promised a close button that the markup never had). A touch swipe past the floor was briefly
  a second door; it was removed, so nothing about closing depends on a gesture or on a pointer type.

  This is additive to the placeable Parameters/Metadata/Errors panels — those
  follow the active editor's selection instead.
-->
<script lang="ts">
	import ParamForm from '$lib/inspector/ParamForm.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { Button, ScrollArea } from '$lib/ui';
	import { beginDrag } from 'tatami';
	import { graph } from '$lib/stores/graph.svelte';
	import { onDestroy } from 'svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { PANE_AXES, coordOf, paneSizeAt, type PaneAxis, type PaneDrag } from './paneDrag';

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
		if (!renderedNode) return;
		void graph()
			.restartNode(renderedNode.uid)
			.catch((e) => console.warn('restart failed', e));
	}

	/** A persisted pane size, or `null` when the user has never dragged one — in which case the
	 *  resting size is the middle term of the stylesheet's own `clamp()`. Null rather than a literal
	 *  because both defaults are relative to the HOST (a fraction of it, or a rem comfort cap
	 *  measured against it), which no number here could express; keeping them in the stylesheet is
	 *  what lets one idiom serve both anchors.
	 *
	 *  Not re-clamped on the way out: what is written below is the RENDERED size, and both bounds are
	 *  that one `clamp()`, so a rendered size is between them by construction — on this host and on
	 *  any other. */
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
	/** The pane itself — the drag measures its RENDERED box, since both bounds are CSS's. */
	let paneEl = $state<HTMLElement | null>(null);
	/** Closing is a real outro, so keep the last node rendered until the transform finishes. The
	 * pane itself stays mounted and parked; once hidden, dropping this snapshot makes the off-state
	 * as cheap as the old unmounted one without taking the outgoing frame away. */
	let renderedNode = $state<NodeInstanceInfo | null>(null);
	const open = $derived(enabled && node !== null);

	$effect(() => {
		if (open) renderedNode = node;
	});

	function finishPanelTransition(e: TransitionEvent): void {
		if (e.target !== e.currentTarget || e.propertyName !== 'transform' || open) return;
		renderedNode = null;
	}

	/** The in-flight resize's teardown; non-null only between pointerdown and its resolution. */
	let teardownResize: (() => void) | null = null;

	function startPanelResize(e: PointerEvent): void {
		if (!paneEl) return;
		e.preventDefault();
		const el = paneEl;
		// The pane's own fact, read back off it rather than re-derived here: the axis is the container
		// query's answer. `dim` is the rest of what it selects (which dimension sizes the pane, which
		// key remembers it), so "orientation picks the axis" stays one named fact instead of a
		// condition re-asked at every line below. No floor is read: BOTH bounds are one `clamp()` in
		// the stylesheet, so this gesture only ever says what size the pointer asked for.
		const axis: PaneAxis =
			getComputedStyle(el).getPropertyValue('--pane-axis').trim() === 'y' ? 'y' : 'x';
		const dim = PANE_AXES[axis];
		// The RENDERED size, not the stored one: the bounds live in CSS, so a value restored from a
		// wider screen is not what is on screen — and a drag that began from it would move the pointer
		// a long way before the pane moved at all.
		const drag: PaneDrag = {
			startSize: dim.sizeOf(el.getBoundingClientRect()),
			startPos: coordOf(axis, e)
		};
		resizing = true;
		// The size is applied live, so there is nothing for a cancel to roll back — and persisting
		// it either way is what keeps a reload agreeing with what is on screen. A cancel that did NOT
		// run this teardown is the actual defect: the window listeners outlived the gesture and the
		// pane kept resizing on the next pointer motion.
		// What is persisted is again the RENDERED size, so the store can never drift outside the bounds.
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
			// One resolution for both, because there is only one outcome a release can have: the size
			// is already applied and already correct, so committing and abandoning agree. A swipe
			// carried past the floor used to close the pane here instead — the ✕ is the only way out.
			commit: finish,
			cancel: finish
		});
	}
	onDestroy(() => teardownResize?.());
</script>

<!-- `inert` whenever closed. The pane stays MOUNTED and keeps its last content only through the
     outgoing transform, so open and close are the same visible slide. Once parked it becomes
     hidden and empty; an orientation change can update its X/Y park without painting a ghost. -->
<aside
	class="side-panel"
	class:open
	class:resizing
	inert={!open}
	bind:this={paneEl}
	ontransitionend={finishPanelTransition}
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
		<ScrollArea>
			<ParamForm node={renderedNode} {onClose} />
			{#if renderedNode && !renderedNode.subpatch}
				<MetadataPanel node={renderedNode} />
				{#if renderedNode.error}
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
						<pre>{renderedNode.error}</pre>
					</section>
				{/if}
			{/if}
		</ScrollArea>
	</aside>

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
		/* THE PANE'S WHOLE RANGE, IN ONE DECLARATION: floor, resting size, ceiling. A floor and a
		   ceiling authored apart are two answers to one question and can cross — `paneDrag.ts` used
		   to state 260px while a percentage ceiling on the 854px host of a landscape phone resolved
		   to 256px, so the allowed range was EMPTY and the pane could not be resized at all. Spelled
		   as one `clamp()` whose bounds are a tenth and nine tenths of the same host, that is
		   unspellable: 10% is below 90% on every geometry there is.

		   HOST-relative, not `vh`/`vw`, and this is the pane's governing premise (D-I2): it answers
		   to the shape of the panel it lives in, not to the screen. A viewport unit would let it
		   overflow a docked editor smaller than the window — and on a phone the two readings differ
		   by the whole of the app chrome.

		   The middle term is the RESTING size, which the inline `--pane-w` replaces once the user has
		   dragged one. It is stated OUTRIGHT rather than left to emerge from the ceiling: it used to
		   be a flat 420px that the old cap clamped down to `min(40%, 30rem)` on a narrow host, so
		   widening the ceiling to 90% would have moved it. `min(40%, 30rem)` is exactly what that cap
		   resolved to — the comfort cap in rem above a ~1050px host, the small-screen guard in
		   percent below it — so the resting size is unchanged on every geometry: 420px on a desktop,
		   342px on the 854px host of a landscape phone. */
		width: clamp(10%, var(--pane-w, min(40%, 30rem)), 90%);
		/* The pane is a query container for ITS OWN width, so its content can yield as the user
		   drags it toward the floor (ParamForm hides the state badge; a Field stacks a paired
		   control) — the host panel's width says nothing about how much room the PANE has.
		   `inline-size`, never `size`: an (orientation:) query needs the block axis contained, so
		   this file's own portrait branch skips this container and keeps answering against
		   `.panel-body` (D-I2) — `size` here would capture it, and an 86px-wide pane would call
		   itself portrait. The layout containment this switches on also makes the pane the
		   containing block for fixed descendants, which is moot: the one fixed overlay reachable
		   from here (Popover) portals to <body>. */
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
	/* When closed the panel is parked off-screen so the next open can still slide from its current
	   anchor. Keep that parked box unpainted: when a host rotates, the park transform changes from
	   X to Y and the transform transition would otherwise fly its ghost diagonally across the host.
	   Its resize handle would also poke back into view and show a resize cursor, so disable all
	   interaction until the pane is actually open. */
	.side-panel:not(.open) {
		visibility: hidden;
		pointer-events: none;
		/* Hold visibility through the outgoing transform, then hide on its final frame. Entering uses
		   the zero-duration base declaration above, so the pane is visible before it slides in. */
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
	/* A traceback is machine output, so mono (D-T3) — stated, not inherited: a bare <pre> takes only
	   app.css's `font: inherit` reset, which now resolves to the chrome face. The ErrorPanel shows
	   the same text; the two must not disagree about what it is. */
	.node-error pre {
		font-family: var(--font-mono);
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
	   ORIENTATION decides the anchored AXIS — portrait a bottom sheet, landscape/desktop the
	   right-hand edge. INPUT MODALITY decides only the resting AFFORDANCE: the grabber pill and the
	   coarse hit band. THE GESTURE IS UNIFORM — edge-drag, identical on a mouse and a finger — and
	   closing is the explicit ✕ alone. Modality once gated a gesture too (a swipe past the floor
	   threw the pane away); it is gone, so the two axes of this rule no longer meet anywhere.

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
	   resize, one persistence idiom. Modality gates only the resting affordance — never a gesture. */
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
				/* This anchor's whole range, in this anchor's one declaration — same shape, same
				   bounds, the other axis. The resting 60% is D-I6's, and it sits inside them. */
				height: clamp(10%, var(--pane-h, 60%), 90%);
				padding-bottom: var(--safe-bottom);
				border-left: none;
				border-top: 1px solid var(--border);
				box-shadow: var(--shadow-sheet);
				/* D-I8: the same motion tokens as the X slide above, a quarter turn round. The
				   `:not(.open)` park rule remains hidden and inert on either axis. */
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
			   Inward is where the pane's own top row is, and in this anchor that row is ParamForm's
			   identity Bar, which carries the ✕ (it lived in a strip of its own when this band was
			   first measured: band bottom 418px, ✕ centre 416px — the reach swallowed the pane's one
			   pointer door, which D-I4 says must never depend on a gesture; moving the ✕ into the
			   Bar moves it DOWN, so leaning the band outward stays right).
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
