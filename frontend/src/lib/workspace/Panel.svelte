<!--
  A leaf of the layout tree: the panel chrome (header) plus the registered
  content component. Clicking anywhere marks the panel active (capture phase) so
  keyboard shortcuts scope to the focused panel.

  Corner grips drive the Blender-style power-user gesture: drag a corner inward
  to SPLIT (axis follows the dominant drag direction), or drag it onto a sibling
  panel to JOIN (that sibling is absorbed). A live ghost previews the result.
  The explicit header dropdown / right-click menu / draggable borders cover the
  same operations for discoverability.
-->
<script lang="ts">
	import { findParent, type Direction, type PanelNode } from './model';
	import { workspace } from './workspace.svelte';
	import { resolvePanelType } from './registry';
	import { portal } from './portal';
	import PanelHeader from './PanelHeader.svelte';
	import { beginDrag } from '$lib/ui';
	import { onDestroy } from 'svelte';

	let { node }: { node: PanelNode } = $props();
	const ws = workspace();
	const type = $derived(resolvePanelType(node.panelType));
	const active = $derived(ws.activePanelId === node.id);

	// --- drop zones (drag a panel or tab in to split / reposition) ---------
	type DropZone = 'left' | 'right' | 'top' | 'bottom';
	let dropZone = $state<DropZone | null>(null);
	// Zones show while dragging any panel/tab, except onto the source panel
	// itself or the active tab dragged onto its own panels (no-ops).
	const dropActive = $derived.by(() => {
		const d = ws.dragging;
		if (!d) return false;
		if (d.kind === 'tab' && d.workspaceId === ws.state.activeWorkspaceId) return false;
		if (d.kind === 'panel' && d.panelId === node.id) return false;
		return true;
	});
	$effect(() => {
		if (!dropActive) dropZone = null;
	});

	function onNodeDragOver(e: DragEvent): void {
		if (!dropActive) return;
		e.preventDefault();
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const d = {
			left: (e.clientX - r.left) / r.width,
			right: (r.right - e.clientX) / r.width,
			top: (e.clientY - r.top) / r.height,
			bottom: (r.bottom - e.clientY) / r.height
		};
		let z: DropZone = 'left';
		for (const k of ['right', 'top', 'bottom'] as DropZone[]) if (d[k] < d[z]) z = k;
		dropZone = z;
	}
	function onNodeDragLeave(e: DragEvent): void {
		const rt = e.relatedTarget as Node | null;
		if (!(e.currentTarget as HTMLElement).contains(rt)) dropZone = null;
	}
	function onNodeDrop(e: DragEvent): void {
		const z = dropZone;
		dropZone = null;
		if (!ws.dragging || !z) return;
		e.preventDefault();
		const direction = z === 'left' || z === 'right' ? 'row' : 'column';
		ws.dropOnPanel(node.id, direction, z === 'left' || z === 'top');
	}

	const CORNERS = ['tl', 'tr', 'bl', 'br'] as const;
	type Corner = (typeof CORNERS)[number];

	interface Ghost {
		mode: 'split' | 'join';
		x: number;
		y: number;
		w: number;
		h: number;
	}
	type Intent =
		| { mode: 'split'; axis: Direction; placeBefore: boolean; fraction: number }
		| { mode: 'join'; targetId: string }
		| null;

	let ghost = $state<Ghost | null>(null);
	let intent: Intent = null;
	/** The in-flight corner drag's teardown; non-null only between pointerdown and its resolution. */
	let teardownDrag: (() => void) | null = null;
	const THRESHOLD = 24;

	function isSibling(a: string, b: string): boolean {
		const root = ws.active.root;
		const pa = findParent(root, a);
		const pb = findParent(root, b);
		return !!pa && !!pb && pa.parent.id === pb.parent.id;
	}

	/**
	 * Resolve a corner drag into a split: the new panel sits on the side of the
	 * grabbed corner (right/left for a row split, bottom/top for a column), the
	 * cursor is the split line, and the ghost previews the NEW panel's region so
	 * its size matches what gets created. `fraction` is that region's share.
	 */
	function computeSplit(
		rect: DOMRect,
		axis: Direction,
		corner: Corner,
		cx: number,
		cy: number
	): { placeBefore: boolean; fraction: number; ghost: Ghost } {
		if (axis === 'row') {
			const onRight = corner === 'tr' || corner === 'br';
			const b = Math.min(rect.right, Math.max(rect.left, cx));
			return onRight
				? {
						placeBefore: false,
						fraction: (rect.right - b) / rect.width,
						ghost: { mode: 'split', x: b, y: rect.top, w: rect.right - b, h: rect.height }
					}
				: {
						placeBefore: true,
						fraction: (b - rect.left) / rect.width,
						ghost: { mode: 'split', x: rect.left, y: rect.top, w: b - rect.left, h: rect.height }
					};
		}
		const onBottom = corner === 'bl' || corner === 'br';
		const b = Math.min(rect.bottom, Math.max(rect.top, cy));
		return onBottom
			? {
					placeBefore: false,
					fraction: (rect.bottom - b) / rect.height,
					ghost: { mode: 'split', x: rect.left, y: b, w: rect.width, h: rect.bottom - b }
				}
			: {
					placeBefore: true,
					fraction: (b - rect.top) / rect.height,
					ghost: { mode: 'split', x: rect.left, y: rect.top, w: rect.width, h: b - rect.top }
				};
	}

	function startCornerDrag(e: PointerEvent, corner: Corner): void {
		e.preventDefault();
		e.stopPropagation();
		ws.setActive(node.id);
		const grip = e.currentTarget as HTMLElement;
		const section = grip.closest('.panel') as HTMLElement | null;
		if (!section) return;
		const rect = section.getBoundingClientRect();
		const startX = e.clientX;
		const startY = e.clientY;

		const onMove = (m: PointerEvent): void => {
			const dx = m.clientX - startX;
			const dy = m.clientY - startY;
			if (Math.abs(dx) < THRESHOLD && Math.abs(dy) < THRESHOLD) {
				intent = null;
				ghost = null;
				return;
			}
			const el = document.elementFromPoint(m.clientX, m.clientY) as HTMLElement | null;
			const targetEl = el?.closest('[data-panel-id]') as HTMLElement | null;
			const targetId = targetEl?.dataset.panelId ?? null;
			if (targetId && targetId !== node.id && isSibling(node.id, targetId) && targetEl) {
				const r = targetEl.getBoundingClientRect();
				intent = { mode: 'join', targetId };
				ghost = { mode: 'join', x: r.left, y: r.top, w: r.width, h: r.height };
			} else {
				const axis: Direction = Math.abs(dx) >= Math.abs(dy) ? 'row' : 'column';
				const { placeBefore, fraction, ghost: gh } = computeSplit(
					rect,
					axis,
					corner,
					m.clientX,
					m.clientY
				);
				intent = { mode: 'split', axis, placeBefore, fraction };
				ghost = gh;
			}
		};
		/** Drop the preview and the pending intent, committing nothing. */
		const abandon = (): void => {
			intent = null;
			ghost = null;
			teardownDrag = null;
		};
		teardownDrag = beginDrag(grip, e.pointerId, {
			move: onMove,
			commit: () => {
				const committed = intent;
				abandon();
				if (!committed) return;
				if (committed.mode === 'split')
					ws.split(node.id, committed.axis, committed.placeBefore, committed.fraction);
				else ws.close(committed.targetId);
			},
			// The gesture was taken away (a pan reclaimed the pointer, a system gesture started) or
			// this panel unmounted mid-drag. Either way the intent it built is not the user's answer:
			// leaving it armed is what let the NEXT tap anywhere commit a split or a close (H6).
			cancel: abandon
		});
	}
	onDestroy(() => teardownDrag?.());
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<section
	class="panel"
	class:active
	onpointerdowncapture={() => ws.setActive(node.id)}
	ondragover={onNodeDragOver}
	ondragleave={onNodeDragLeave}
	ondrop={onNodeDrop}
	data-panel-id={node.id}
	data-panel-type={node.panelType}
>
	<PanelHeader {node} />
	<div class="panel-body">
		{#if type.component}
			{@const Content = type.component}
			<Content
				panelId={node.id}
				state={node.state}
				setState={(s, intent) => ws.setPanelState(node.id, s, intent)}
			/>
		{:else}
			<div class="missing">Unknown panel type: <code>{node.panelType}</code></div>
		{/if}

		<!-- Corner grips for drag-split / drag-join. Live on the body so they
		     sit clear of the header buttons; the editor nudges its controls
		     inward so the bottom grips stay reachable. -->
		{#each CORNERS as c (c)}
			<div
				class="corner {c}"
				onpointerdown={(e) => startCornerDrag(e, c)}
				role="separator"
				aria-label="Split or join panel"
				tabindex="-1"
			></div>
		{/each}

		{#if dropActive && dropZone}
			<div class="drop-zone {dropZone}" data-testid="tab-drop-zone"></div>
		{/if}
	</div>
</section>

{#if ghost}
	<div
		class="drag-ghost {ghost.mode}"
		use:portal
		style="left:{ghost.x}px; top:{ghost.y}px; width:{ghost.w}px; height:{ghost.h}px"
	></div>
{/if}

<style>
	.panel {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		min-width: 0;
		min-height: 0;
		/* D5, applied where it repeats most: the panel is a SURFACE floating on the workspace ground,
		   not a rectangle drawn on it. The `--bg` gutter between panels is what separates them, so
		   the frame this used to wear is deleted — with it, every split seam stacked three hairlines
		   (this border, the splitter's rule, the neighbour's border) across an 8px span. Panels whose
		   content is a canvas (the node editor, EmptyPanel) still paint `--bg` over this. */
		background: var(--surface-1);
		/* Transparent, not deleted (the EmptyPanel/D5 trade): it paints nothing, but it keeps the
		   1px inset that the active ring below lives in. Chromium paints a non-positioned element's
		   outline before its positioned descendants, so with the body flush to the panel edge an
		   opaque content layer (`.svelte-flow`, a canvas) covers the ring — which is the very
		   failure app.css's ring convention was written about. */
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		overflow: hidden;
		outline: 1px solid transparent;
		outline-offset: -1px;
		transition: outline-color var(--dur-fast) var(--ease);
	}
	/* Active-panel accent, drawn as an OUTLINE, not an inset box-shadow: an inset shadow paints
	   below child content, so the header bar and the editor's opaque `.svelte-flow` background
	   hid every edge of it (app.css's ring convention). An outline is painted last in its stacking
	   context, so it clears both without a pseudo-element — which also matters here, since `.panel`
	   is a flex container (a ::after would become a flex item) whose `position: static` the body's
	   fixed-position overlays rely on. `outline-offset: -1px` lands the line just inside the 1px
	   border, exactly where the shadow used to sit.

	   This is the ONE place the ring is drawn, for every panel type. Four types used to opt out and
	   ring just their own body instead, which drew the same state as two different shapes and left
	   their headers outside the focus indication — the panel is what has focus, so the panel is what
	   the ring frames. */
	.panel.active {
		outline-color: var(--ring-accent);
	}
	.panel-body {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
		/* The query container every panel's content sizes against (`@container` in the UI
		   primitives). `size`, not `inline-size`: the inspector's anchor is decided by whether its
		   host panel is taller than it is wide (D-I2), and `@container (orientation: …)` needs the
		   BLOCK axis contained to answer that. Safe here for the reason the three lines above spell
		   out — `flex: 1; min-height: 0` means this body's height comes from its flex parent and
		   never from its content, so containing the block axis cannot change its used size; every
		   `inline-size` consumer (`ui/Field.svelte`) keeps working, since `size` is a superset.
		   NB: the panel's fixed-position overlays (the add-node menu, ViewerSettingsMenu, the
		   link-ghost) all portal to <body> so the containing block this establishes can never
		   re-anchor them off the viewport. */
		container-type: size;
	}
	.missing {
		display: grid;
		place-items: center;
		height: 100%;
		color: var(--text-dim);
	}
	/* R's floors sweep left the grip at 16px, deliberately. A --hit grip would put a 44px triangle
	   of split-drag over every panel corner — including the lower-left one the node editor's zoom
	   cluster was just moved INTO (`NodeEditorPanel`'s coarse rule tucks it to --space-6, which
	   clears a 16px triangle and would not clear a 44px one). The split/join corner gesture is a
	   fine-pointer power-user affordance; on touch the door is the panel header's own menu, which
	   carries the same operations (D-R5). Growing this would cost a real target to serve a
	   theoretical one.
	   It gets no `touch-action` either, and that is the same decision seen from the other side: with
	   the default, a touch that lands on a corner and then moves is reclaimed by the browser as a
	   pan, which cancels the pointer — and the drag now tears down cleanly on that cancel instead of
	   arming a split for the next tap to commit. Declaring `touch-action: none` here would take that
	   pan away to enable a gesture we deliberately do not offer on touch. */
	.corner {
		position: absolute;
		width: 16px;
		height: 16px;
		z-index: var(--z-chrome);
		opacity: 0;
		transition: opacity var(--dur-slow) var(--ease);
	}
	/* A faint triangular tab in each corner, brightening on hover. */
	.corner::after {
		content: '';
		position: absolute;
		inset: 0;
		background: var(--text-muted);
		opacity: 0.5;
	}
	.panel-body:hover .corner {
		opacity: 0.5;
	}
	.corner:hover {
		opacity: 1;
	}
	.corner.tl {
		top: 0;
		left: 0;
		cursor: nwse-resize;
		clip-path: polygon(0 0, 100% 0, 0 100%);
	}
	.corner.tr {
		top: 0;
		right: 0;
		cursor: nesw-resize;
		clip-path: polygon(0 0, 100% 0, 100% 100%);
	}
	.corner.bl {
		bottom: 0;
		left: 0;
		cursor: nesw-resize;
		clip-path: polygon(0 0, 0 100%, 100% 100%);
	}
	.corner.br {
		bottom: 0;
		right: 0;
		cursor: nwse-resize;
		clip-path: polygon(100% 0, 100% 100%, 0 100%);
	}
	.drag-ghost {
		position: fixed;
		z-index: var(--z-drag-ghost);
		pointer-events: none;
		border-radius: var(--radius-sm);
		box-sizing: border-box;
	}
	.drag-ghost.split {
		background: var(--accent-fill);
		border: 1px solid var(--accent);
	}
	.drag-ghost.join {
		background: var(--danger-fill);
		border: 1px solid var(--danger);
	}
	/* Live preview of where a dragged tab will land when dropped. */
	.drop-zone {
		position: absolute;
		z-index: var(--z-drag-ghost);
		pointer-events: none;
		background: color-mix(in srgb, var(--accent) 20%, transparent);
		border: 2px solid var(--accent);
		border-radius: var(--radius-sm);
	}
	.drop-zone.left {
		left: 0;
		top: 0;
		width: 50%;
		height: 100%;
	}
	.drop-zone.right {
		right: 0;
		top: 0;
		width: 50%;
		height: 100%;
	}
	.drop-zone.top {
		left: 0;
		top: 0;
		width: 100%;
		height: 50%;
	}
	.drop-zone.bottom {
		left: 0;
		bottom: 0;
		width: 100%;
		height: 50%;
	}
</style>
