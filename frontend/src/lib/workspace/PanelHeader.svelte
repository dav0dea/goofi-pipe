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
	<button class="content-btn" onclick={openContent} title="Change panel content">
		{#if type.icon}<span class="ic">{type.icon}</span>{/if}
		<span class="title">{type.title}</span>
		<span class="caret">▾</span>
	</button>
	<div class="spacer"></div>
	<button
		class="hdr-btn"
		title={isMax ? 'Restore' : 'Maximize'}
		onclick={() => ws.toggleMaximize(node.id)}
		aria-label={isMax ? 'Restore panel' : 'Maximize panel'}>⤢</button
	>
	<button
		class="hdr-btn"
		title="Close panel"
		disabled={!canClose}
		onclick={() => ws.close(node.id)}
		aria-label="Close panel">✕</button
	>
</div>

{#if menu}
	<ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => (menu = null)} />
{/if}

<style>
	.panel-header {
		display: flex;
		align-items: center;
		height: var(--panel-header-h, 26px);
		flex: 0 0 auto;
		padding: 0 4px 0 4px;
		background: var(--surface-1);
		border-bottom: 1px solid var(--border);
		gap: 2px;
		user-select: none;
		cursor: grab;
	}
	.panel-header:active {
		cursor: grabbing;
	}
	.content-btn {
		display: flex;
		align-items: center;
		gap: 5px;
		height: 20px;
		padding: 0 6px;
		background: transparent;
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 0.82rem;
		cursor: pointer;
	}
	.content-btn:hover {
		background: var(--surface-3);
		border-color: var(--border);
	}
	.content-btn .ic {
		opacity: 0.85;
	}
	.content-btn .title {
		font-weight: 500;
	}
	.content-btn .caret {
		opacity: 0.5;
		font-size: 0.7em;
	}
	.spacer {
		flex: 1;
	}
	.hdr-btn {
		width: 20px;
		height: 20px;
		display: grid;
		place-items: center;
		padding: 0;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		font-size: 0.85rem;
		cursor: pointer;
	}
	.hdr-btn:hover:not(:disabled) {
		background: var(--surface-3);
		color: var(--text);
	}
	.hdr-btn:disabled {
		opacity: 0.3;
		cursor: default;
	}
</style>
