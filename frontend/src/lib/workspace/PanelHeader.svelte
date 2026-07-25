<!--
  Panel header: the content-type dropdown (left) plus maximize/close buttons,
  and a right-click context menu exposing every structural action (split,
  maximize, change content, close). The dropdown and the menu both build their
  panel-type list from the registry, so new panel types appear automatically.
-->
<script lang="ts">
	import { countPanels, type PanelNode } from './model';
	import { workspace } from './workspace.svelte';
	import { listPanelTypes, resolvePanelType } from './registry';
	import type { MenuItem } from './menu';
	import ContextMenu from './ContextMenu.svelte';
	import { Button, IconButton } from '$lib/ui';

	let { node }: { node: PanelNode } = $props();
	const ws = workspace();
	const type = $derived(resolvePanelType(node.panelType));
	const isMax = $derived(ws.maximizedPanelId === node.id);
	const canClose = $derived(countPanels(ws.active.root) > 1);

	let menu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	function contentItems(): MenuItem[] {
		return listPanelTypes().map((t) => ({
			label: t.title,
			icon: t.icon,
			checked: t.id === node.panelType,
			action: () => ws.setType(node.id, t.id)
		}));
	}

	function openContent(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		menu = { x: r.left, y: r.bottom + 2, items: contentItems() };
	}

	function structuralItems(): MenuItem[] {
		return [
			{ label: 'Split Right', icon: '▕', action: () => ws.split(node.id, 'row') },
			{ label: 'Split Down', icon: '▁', action: () => ws.split(node.id, 'column') },
			{ separator: true },
			{
				label: isMax ? 'Restore' : 'Maximize',
				icon: '⤢',
				action: () => ws.toggleMaximize(node.id)
			},
			{ label: 'Change content', items: contentItems() },
			{ separator: true },
			{ label: 'Close Panel', icon: '✕', disabled: !canClose, action: () => ws.close(node.id) }
		];
	}

	function onHeaderContext(e: MouseEvent): void {
		e.preventDefault();
		menu = { x: e.clientX, y: e.clientY, items: structuralItems() };
	}
</script>

<div
	class="panel-header"
	draggable="true"
	oncontextmenu={onHeaderContext}
	ondragstart={(e) => {
		// Don't start a panel move when the drag begins on a control.
		if ((e.target as HTMLElement).closest('button, select, input')) {
			e.preventDefault();
			return;
		}
		ws.dragging = { kind: 'panel', workspaceId: ws.state.activeWorkspaceId, panelId: node.id };
	}}
	ondragend={() => (ws.dragging = null)}
	role="toolbar"
	tabindex="-1"
	aria-label="Panel header"
	data-testid="panel-header"
>
	<Button variant="ghost" class="content-btn" onclick={openContent} title="Change panel content">
		{#if type.icon}<span class="ic">{type.icon}</span>{/if}
		<span class="title">{type.title}</span>
		<span class="caret">▾</span>
	</Button>
	<div class="spacer"></div>
	<IconButton
		variant="ghost"
		density="chrome"
		class="hdr-btn"
		title={isMax ? 'Restore' : 'Maximize'}
		label={isMax ? 'Restore panel' : 'Maximize panel'}
		onclick={() => ws.toggleMaximize(node.id)}>⤢</IconButton
	>
	<IconButton
		variant="ghost"
		density="chrome"
		class="hdr-btn"
		title="Close panel"
		label="Close panel"
		disabled={!canClose}
		onclick={() => ws.close(node.id)}>✕</IconButton
	>
</div>

{#if menu}
	<ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => (menu = null)} />
{/if}

<style>
	.panel-header {
		display: flex;
		align-items: center;
		height: var(--panel-header-h);
		flex: 0 0 auto;
		padding: 0 var(--space-2);
		background: var(--surface-1);
		border-bottom: 1px solid var(--border);
		gap: var(--space-1);
		user-select: none;
		cursor: grab;
	}
	.panel-header:active {
		cursor: grabbing;
	}
	/* The primitives keep the frozen 20px control geometry of the 26px bar. Under a coarse
	   pointer the bar itself grows to --hit (app.css), so the floors apply unchanged there.
	   The `button` tag qualifier is load-bearing: without it this ties with the primitive's own
	   `.ui-btn.s-md` padding, and the two rules live in separate built CSS chunks — a tie there
	   is settled by the emitted <link> order, not by the source.
	   (Button has no density axis: this pin is padding + gap geometry, not a hit floor.) */
	.panel-header :global(button.content-btn) {
		height: 20px;
		padding: 0 var(--space-3);
		gap: var(--space-2);
	}
	/* The icon buttons state only their box — `density="chrome"` owns the coarse-pointer floor. */
	.panel-header :global(.hdr-btn) {
		--icon-btn-size: 20px;
		color: var(--text-dim);
	}
	.panel-header :global(.hdr-btn:hover:not(:disabled)) {
		color: var(--text);
	}
	/* opacity: intentional — the glyph and the caret are quieted BELOW the title they sit beside
	   (a hierarchy, not a disabled state); --disabled-opacity would read as "this header is inert". */
	.ic {
		opacity: 0.85;
	}
	.title {
		font-weight: 500;
	}
	.caret {
		opacity: 0.5;
		font-size: var(--fs-micro);
	}
	.spacer {
		flex: 1;
	}
</style>
