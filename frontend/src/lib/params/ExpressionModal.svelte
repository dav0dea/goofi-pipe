<script lang="ts">
	import { onMount } from 'svelte';
	import { portal } from '$lib/workspace/portal';
	import { ui } from '$lib/stores/ui.svelte';

	type Props = {
		title: string;
		initial: string;
		preview: string;
		onApply: (source: string) => void;
		onCancel: () => void;
	};
	const { title, initial, preview, onApply, onCancel }: Props = $props();

	let source = $state(initial);
	let textarea: HTMLTextAreaElement | null = $state(null);
	let modalEl: HTMLDivElement | null = $state(null);

	// Modal geometry. Initial values are reset on mount once we can read
	// the viewport size — we want centered + clamped, not the raw
	// constants below.
	let pos = $state({ x: 0, y: 0 });
	let size = $state({ w: 640, h: 400 });
	const MIN_W = 320;
	const MIN_H = 220;
	const MARGIN = 8;

	let dragStart: { mx: number; my: number; px: number; py: number; pointerId: number } | null = null;
	let resizeStart: { mx: number; my: number; w: number; h: number; pointerId: number } | null = null;

	function clampPos(x: number, y: number, w: number, h: number) {
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		return {
			x: Math.max(MARGIN, Math.min(vw - w - MARGIN, x)),
			y: Math.max(MARGIN, Math.min(vh - h - MARGIN, y))
		};
	}

	function clampSize(w: number, h: number) {
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		return {
			w: Math.max(MIN_W, Math.min(vw - 2 * MARGIN, w)),
			h: Math.max(MIN_H, Math.min(vh - 2 * MARGIN, h))
		};
	}

	function startDrag(e: PointerEvent): void {
		// Don't start a drag from the close button.
		if ((e.target as HTMLElement).closest('button')) return;
		e.preventDefault();
		const target = e.currentTarget as HTMLElement;
		try {
			target.setPointerCapture(e.pointerId);
		} catch {
			/* some touch backends refuse capture; window listener still works */
		}
		dragStart = { mx: e.clientX, my: e.clientY, px: pos.x, py: pos.y, pointerId: e.pointerId };
	}

	function moveDrag(e: PointerEvent): void {
		if (!dragStart) return;
		const dx = e.clientX - dragStart.mx;
		const dy = e.clientY - dragStart.my;
		pos = clampPos(dragStart.px + dx, dragStart.py + dy, size.w, size.h);
	}

	function endDrag(e: PointerEvent): void {
		if (!dragStart) return;
		const target = e.currentTarget as HTMLElement;
		try {
			if (target.hasPointerCapture(dragStart.pointerId)) {
				target.releasePointerCapture(dragStart.pointerId);
			}
		} catch {
			/* ignore */
		}
		dragStart = null;
	}

	function startResize(e: PointerEvent): void {
		e.preventDefault();
		e.stopPropagation();
		const target = e.currentTarget as HTMLElement;
		try {
			target.setPointerCapture(e.pointerId);
		} catch {
			/* ignore */
		}
		resizeStart = {
			mx: e.clientX,
			my: e.clientY,
			w: size.w,
			h: size.h,
			pointerId: e.pointerId
		};
	}

	function moveResize(e: PointerEvent): void {
		if (!resizeStart) return;
		const dx = e.clientX - resizeStart.mx;
		const dy = e.clientY - resizeStart.my;
		const next = clampSize(resizeStart.w + dx, resizeStart.h + dy);
		// Don't allow resize to push the modal off-screen on the right /
		// bottom: clamp width / height against current position too.
		const maxW = Math.max(MIN_W, window.innerWidth - pos.x - MARGIN);
		const maxH = Math.max(MIN_H, window.innerHeight - pos.y - MARGIN);
		size = { w: Math.min(next.w, maxW), h: Math.min(next.h, maxH) };
	}

	function endResize(e: PointerEvent): void {
		if (!resizeStart) return;
		const target = e.currentTarget as HTMLElement;
		try {
			if (target.hasPointerCapture(resizeStart.pointerId)) {
				target.releasePointerCapture(resizeStart.pointerId);
			}
		} catch {
			/* ignore */
		}
		resizeStart = null;
	}

	function apply(): void {
		onApply(source);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			onCancel();
		} else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
			e.preventDefault();
			e.stopPropagation();
			apply();
		}
	}

	function onWindowResize(): void {
		// Re-clamp on viewport change so a previously valid layout doesn't
		// end up off-screen after the user resizes their window.
		const next = clampSize(size.w, size.h);
		size = next;
		pos = clampPos(pos.x, pos.y, next.w, next.h);
	}

	onMount(() => {
		// Size first, position second — clamp size to viewport then center.
		const s = clampSize(size.w, size.h);
		size = s;
		pos = {
			x: Math.max(MARGIN, (window.innerWidth - s.w) / 2),
			y: Math.max(MARGIN, (window.innerHeight - s.h) / 2)
		};
		window.addEventListener('keydown', onKeydown, true);
		window.addEventListener('resize', onWindowResize);
		ui().modalOpen = true; // global undo/redo stands down while editing
		textarea?.focus();
		const len = source.length;
		textarea?.setSelectionRange(len, len);
		return () => {
			window.removeEventListener('keydown', onKeydown, true);
			window.removeEventListener('resize', onWindowResize);
			ui().modalOpen = false;
		};
	});
</script>

<div use:portal class="modal-root">
	<div
		class="modal-overlay"
		role="presentation"
		onclick={onCancel}
		data-testid="expression-modal"
	></div>
	<div
		bind:this={modalEl}
		class="modal"
		role="dialog"
		aria-label="Edit expression: {title}"
		style="left: {pos.x}px; top: {pos.y}px; width: {size.w}px; height: {size.h}px;"
	>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<header
			class="drag-handle"
			onpointerdown={startDrag}
			onpointermove={moveDrag}
			onpointerup={endDrag}
			onpointercancel={endDrag}
		>
			<div class="title">edit expression: <span class="param">{title}</span></div>
			<button
				class="close"
				onclick={onCancel}
				aria-label="Close"
				data-testid="expression-modal-cancel"
			>
				✕
			</button>
		</header>
		<textarea
			bind:this={textarea}
			bind:value={source}
			spellcheck="false"
			autocapitalize="off"
			data-testid="expression-modal-textarea"
		></textarea>
		<footer>
			<div class="preview">
				<span class="hint">preview:</span>
				<span class="value">{preview}</span>
			</div>
			<div class="actions">
				<span class="kbd-hint">⌃⏎ apply · esc cancel</span>
				<button class="btn ghost" onclick={onCancel} data-testid="expression-modal-cancel-btn">
					cancel
				</button>
				<button class="btn primary" onclick={apply} data-testid="expression-modal-apply">
					apply
				</button>
			</div>
		</footer>
		<!-- Resize handle: a small grip in the bottom-right corner. -->
		<div
			class="resize-grip"
			role="presentation"
			onpointerdown={startResize}
			onpointermove={moveResize}
			onpointerup={endResize}
			onpointercancel={endResize}
			data-testid="expression-modal-resize"
		></div>
	</div>
</div>

<style>
	.modal-root {
		/* Portaled to body; lives outside the side-panel stacking context. */
		position: fixed;
		inset: 0;
		z-index: var(--z-modal);
		pointer-events: none;
	}
	.modal-overlay {
		position: absolute;
		inset: 0;
		background: color-mix(in srgb, var(--bg) 65%, transparent);
		backdrop-filter: blur(2px);
		pointer-events: auto;
	}
	.modal {
		position: absolute;
		display: flex;
		flex-direction: column;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
		pointer-events: auto;
		overflow: hidden;
	}
	.drag-handle {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 14px;
		border-bottom: 1px solid var(--border);
		cursor: move;
		user-select: none;
		touch-action: none;
		flex-shrink: 0;
	}
	.title {
		flex: 1;
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--text-dim);
		letter-spacing: 0.02em;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.title .param {
		color: var(--accent);
	}
	.close {
		background: transparent;
		border: none;
		color: var(--text-faint);
		cursor: pointer;
		font-size: 14px;
		padding: 4px 8px;
		border-radius: 3px;
	}
	.close:hover {
		color: var(--text);
		background: var(--bg-elev-3);
	}
	textarea {
		flex: 1;
		min-height: 0;
		resize: none;
		padding: 14px;
		font-family: var(--font-mono);
		font-size: 12px;
		line-height: 1.5;
		background: var(--bg);
		color: var(--accent);
		border: none;
		outline: none;
		tab-size: 4;
	}
	footer {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 8px 14px;
		border-top: 1px solid var(--border);
		flex-shrink: 0;
	}
	.preview {
		flex: 1;
		min-width: 0;
		display: flex;
		gap: 6px;
		align-items: baseline;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text-faint);
		overflow: hidden;
	}
	.preview .hint {
		opacity: 0.6;
	}
	.preview .value {
		color: var(--text-dim);
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.actions {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.kbd-hint {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--text-faint);
		opacity: 0.7;
		margin-right: 4px;
	}
	.btn {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 6px 12px;
		border-radius: 3px;
		cursor: pointer;
		letter-spacing: 0.02em;
		text-transform: lowercase;
		transition:
			background 80ms ease,
			color 80ms ease;
	}
	.btn.ghost {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-dim);
	}
	.btn.ghost:hover {
		color: var(--text);
		border-color: var(--text-dim);
	}
	.btn.primary {
		background: var(--accent);
		border: 1px solid var(--accent);
		color: #0a0c10;
		font-weight: 600;
	}
	.btn.primary:hover {
		background: color-mix(in srgb, var(--accent) 80%, white);
	}
	.resize-grip {
		position: absolute;
		right: 0;
		bottom: 0;
		width: 14px;
		height: 14px;
		cursor: nwse-resize;
		touch-action: none;
		/* A subtle diagonal hash so the affordance is visible without
		   shouting. Two stacked lines drawn via repeating-linear-gradient. */
		background: repeating-linear-gradient(
			135deg,
			transparent 0,
			transparent 3px,
			var(--text-faint) 3px,
			var(--text-faint) 4px
		);
		opacity: 0.5;
		transition: opacity 100ms ease;
	}
	.resize-grip:hover {
		opacity: 1;
	}
</style>
