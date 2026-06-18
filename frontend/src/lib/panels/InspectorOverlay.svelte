<!--
  Selection inspector for a single editor panel. Slides in from the right edge
  of its host editor (not the whole window) when that editor has exactly one
  node selected, showing its parameters, errors, and metadata. Width is
  drag-resizable.

  Visibility is per-editor: `enabled` is owned by the host NodeEditorPanel
  (turned on by the editor's own corner control, off by this pane's close
  button) — there is no global header toggle.

  This is additive to the placeable Parameters/Metadata/Errors panels — those
  follow the active editor's selection instead.
-->
<script lang="ts">
	import ParamPanel from '$lib/params/ParamPanel.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let {
		node,
		enabled
	}: {
		node: NodeInstanceInfo | null;
		enabled: boolean;
	} = $props();

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

	function startPanelResize(e: PointerEvent): void {
		e.preventDefault();
		const startX = e.clientX;
		const startW = panelWidth;
		resizing = true;
		const onMove = (m: PointerEvent): void => {
			const next = startW - (m.clientX - startX);
			panelWidth = Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, next));
		};
		const onUp = (): void => {
			resizing = false;
			window.removeEventListener('pointermove', onMove);
			window.removeEventListener('pointerup', onUp);
			try {
				localStorage.setItem('goofi.panelWidth', String(Math.round(panelWidth)));
			} catch {
				/* private mode; persistence is best-effort */
			}
		};
		window.addEventListener('pointermove', onMove);
		window.addEventListener('pointerup', onUp);
	}
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
		<div class="panel-scroll">
			<ParamPanel {node} />
			{#if node && !node.subpatch}
				<MetadataPanel {node} />
				{#if node.error}
					<section class="node-error" data-testid="inspector-error">
						<header>Error</header>
						<pre>{node.error}</pre>
					</section>
				{/if}
			{/if}
		</div>
	</aside>
{/if}

<style>
	.side-panel {
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		background: color-mix(in srgb, var(--bg-elev-1) 96%, transparent);
		backdrop-filter: blur(8px);
		border-left: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		min-width: 0;
		transform: translateX(100%);
		transition: transform 180ms ease;
		box-shadow: -8px 0 24px rgba(0, 0, 0, 0.35);
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
	.panel-scroll {
		flex: 1;
		overflow-y: auto;
		min-height: 0;
	}
	/* Current processing error for the selected node — a simple snapshot, shown
	   only while the node is errored, after the metadata section. */
	.node-error {
		padding: 12px;
		border-top: 1px solid var(--border);
		background: var(--bg-elev-1);
	}
	.node-error header {
		font-weight: 600;
		font-size: 11px;
		color: var(--danger);
		margin-bottom: 6px;
	}
	.node-error pre {
		font-family: var(--font-mono);
		font-size: 10px;
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
		transition: background 100ms ease;
	}
	.resize-handle:hover::after,
	.side-panel.resizing .resize-handle::after {
		background: var(--accent);
	}
</style>
