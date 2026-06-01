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

	let { node }: { node: PanelNode } = $props();
	const ws = workspace();
	const type = $derived(resolvePanelType(node.panelType));
	const active = $derived(ws.activePanelId === node.id);

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
		| { mode: 'split'; axis: Direction; placeBefore: boolean }
		| { mode: 'join'; targetId: string }
		| null;

	let ghost = $state<Ghost | null>(null);
	let intent: Intent = null;
	const THRESHOLD = 24;

	function isSibling(a: string, b: string): boolean {
		const root = ws.active.root;
		const pa = findParent(root, a);
		const pb = findParent(root, b);
		return !!pa && !!pb && pa.parent.id === pb.parent.id;
	}

	function splitGhost(rect: DOMRect, axis: Direction, placeBefore: boolean, cx: number, cy: number): Ghost {
		if (axis === 'row') {
			const b = Math.min(rect.right, Math.max(rect.left, cx));
			return placeBefore
				? { mode: 'split', x: rect.left, y: rect.top, w: b - rect.left, h: rect.height }
				: { mode: 'split', x: b, y: rect.top, w: rect.right - b, h: rect.height };
		}
		const b = Math.min(rect.bottom, Math.max(rect.top, cy));
		return placeBefore
			? { mode: 'split', x: rect.left, y: rect.top, w: rect.width, h: b - rect.top }
			: { mode: 'split', x: rect.left, y: b, w: rect.width, h: rect.bottom - b };
	}

	function startCornerDrag(e: PointerEvent, corner: Corner): void {
		void corner;
		e.preventDefault();
		e.stopPropagation();
		ws.setActive(node.id);
		const section = (e.currentTarget as HTMLElement).closest('.panel') as HTMLElement | null;
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
				const placeBefore = axis === 'row' ? dx < 0 : dy < 0;
				intent = { mode: 'split', axis, placeBefore };
				ghost = splitGhost(rect, axis, placeBefore, m.clientX, m.clientY);
			}
		};
		const onUp = (): void => {
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
			const committed = intent;
			intent = null;
			ghost = null;
			if (!committed) return;
			if (committed.mode === 'split') ws.split(node.id, committed.axis, committed.placeBefore);
			else ws.close(committed.targetId);
		};
		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}
</script>

<section
	class="panel"
	class:active
	onpointerdowncapture={() => ws.setActive(node.id)}
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
				setState={(s) => ws.setPanelState(node.id, s)}
				{active}
			/>
		{:else}
			<div class="missing">Unknown panel type: <code>{node.panelType}</code></div>
		{/if}

		<!-- Corner grips for drag-split / drag-join. Live on the body so they
		     sit clear of the header buttons; the editor nudges its controls /
		     minimap inward so the bottom grips stay reachable. -->
		{#each CORNERS as c (c)}
			<div
				class="corner {c}"
				onpointerdown={(e) => startCornerDrag(e, c)}
				role="separator"
				aria-label="Split or join panel"
				tabindex="-1"
			></div>
		{/each}
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
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		box-shadow: inset 0 0 0 1px transparent;
		transition: box-shadow 100ms ease;
	}
	.panel.active {
		box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 45%, transparent);
	}
	.panel-body {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
	}
	.missing {
		display: grid;
		place-items: center;
		height: 100%;
		color: var(--text-dim);
	}
	.corner {
		position: absolute;
		width: 16px;
		height: 16px;
		z-index: var(--z-chrome);
		opacity: 0;
		transition: opacity 120ms ease;
	}
	/* A faint triangular tab in each corner, brightening on hover. */
	.corner::after {
		content: '';
		position: absolute;
		inset: 0;
		background: var(--text-faint);
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
		background: color-mix(in srgb, var(--accent) 18%, transparent);
		border: 1px solid var(--accent);
	}
	.drag-ghost.join {
		background: color-mix(in srgb, var(--danger) 16%, transparent);
		border: 1px solid var(--danger);
	}
</style>
