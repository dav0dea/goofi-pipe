<!--
  Selection inspector for a single editor panel. Slides in from the right edge
  of its host editor (not the whole window) when that editor has exactly one
  node selected, showing its parameters, errors, and metadata. Width is
  drag-resizable.

  Visibility is per-editor: `enabled` is owned by the host NodeEditorPanel and
  toggled by that editor's own `inspector-toggle` corner control — both ways.
  This pane has no close button of its own, and there is no global header toggle.

  This is additive to the placeable Parameters/Metadata/Errors panels — those
  follow the active editor's selection instead.
-->
<script lang="ts">
	import ParamForm from '$lib/inspector/ParamForm.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { beginDrag, Button, ScrollArea } from '$lib/ui';
	import { graph } from '$lib/stores/graph.svelte';
	import { onDestroy } from 'svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let {
		node,
		enabled
	}: {
		node: NodeInstanceInfo | null;
		enabled: boolean;
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

	const MIN_PANEL_WIDTH = 260;
	const MAX_PANEL_WIDTH = 720;
	const STORED_WIDTH = (() => {
		try {
			const raw = localStorage.getItem('goofi.panelWidth');
			if (!raw) return 420;
			const n = parseInt(raw, 10);
			if (!Number.isFinite(n)) return 420;
			return Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, n));
		} catch {
			return 420;
		}
	})();
	let panelWidth = $state(STORED_WIDTH);
	let resizing = $state(false);

	/** The in-flight resize's teardown; non-null only between pointerdown and its resolution. */
	let teardownResize: (() => void) | null = null;

	function startPanelResize(e: PointerEvent): void {
		e.preventDefault();
		const startX = e.clientX;
		const startW = panelWidth;
		resizing = true;
		// The width is applied live, so there is nothing for a cancel to roll back — and persisting
		// it either way is what keeps a reload agreeing with what is on screen. A cancel that did NOT
		// run this teardown is the actual defect: the window listeners outlived the gesture and the
		// pane kept resizing on the next pointer motion.
		const finish = (): void => {
			resizing = false;
			teardownResize = null;
			try {
				localStorage.setItem('goofi.panelWidth', String(Math.round(panelWidth)));
			} catch {
				/* private mode; persistence is best-effort */
			}
		};
		teardownResize = beginDrag(e.currentTarget as HTMLElement, e.pointerId, {
			move: (m) => {
				const next = startW - (m.clientX - startX);
				panelWidth = Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, next));
			},
			commit: finish,
			cancel: finish
		});
	}
	onDestroy(() => teardownResize?.());
</script>

{#if enabled}
	<aside
		class="side-panel"
		class:open={node !== null}
		class:resizing
		style="width: {panelWidth}px"
		data-testid="auto-side-panel"
	>
		<div
			class="resize-handle"
			role="separator"
			aria-orientation="vertical"
			aria-label="Resize side panel"
			onpointerdown={startPanelResize}
			data-testid="panel-resize-handle"
		></div>
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
		/* The width above is JS-inline and clamped only to [260, 720] — never to the host. On an
		   editor narrower than the stored width the pane covered the whole canvas with its own left
		   edge clipped off, which is a DEAD END: deselecting (a tap on the canvas) is the only way
		   to close it. One `--hit` of canvas is exactly the escape hatch it must not eat. */
		max-width: calc(100% - var(--hit));
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
	   panel's JS-owned width is unaffected. Horizontal only — the handle already spans the full
	   height, and a symmetric percentage inset would shrink that to --hit. */
	@media (hover: none) and (pointer: coarse) {
		.resize-handle::before {
			content: '';
			position: absolute;
			inset: 0 calc((var(--hit) - 100%) / -2);
		}
		/* …and the seam PAINTS at rest. Transparent-until-hover is the whole of its discoverability
		   on a fine pointer; with no hover the pane simply had an invisible edge. Quieter than the
		   accent the hover/drag state uses, so it reads as an edge rather than as active. */
		.resize-handle::after {
			background: var(--border-strong);
		}
	}
</style>
