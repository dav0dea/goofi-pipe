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
	import { createLongPress } from '$lib/editor/longPress';
	import { Button, IconButton } from '$lib/ui';
	import { onDestroy } from 'svelte';

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

	/**
	 * The coarse-pointer door onto the same menu (D-R5). Split Right and Split Down had NO other
	 * door — the header is a `role="toolbar" tabindex="-1"` with no keydown handler, and its menu
	 * was `oncontextmenu`-only — so on a phone a panel could not be split at all.
	 *
	 * Armed for `touch` alone, and the editor's own recognizer is reused rather than a second
	 * gesture concept invented. It never calls `stopPropagation`, so the same pointerdown still
	 * reaches `Panel.svelte`'s capture-phase `setActive` — freezing `activePanelId` is how a
	 * swallowed press would quietly drift every panel's selection scoping.
	 */
	const headerPress = createLongPress((at) => {
		menu = { x: at.clientX, y: at.clientY, items: structuralItems() };
	});

	function onHeaderPointerDown(e: PointerEvent): void {
		if (e.pointerType !== 'touch') return;
		// The header's own controls keep their own actions: a press that landed on ✕ would both
		// open a menu and, on release, close the panel the menu describes.
		if ((e.target as HTMLElement | null)?.closest('button')) return;
		headerPress.start(e);
	}

	// A press in flight must not fire into an unmounted panel (a close, a tab switch, a split).
	onDestroy(headerPress.cancel);
</script>

<div
	class="panel-header"
	draggable="true"
	oncontextmenu={onHeaderContext}
	onpointerdown={onHeaderPointerDown}
	onpointermove={headerPress.move}
	onpointerup={headerPress.cancel}
	onpointercancel={headerPress.cancel}
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
		/* The TOP rung of the panel's own ladder — body `--surface-1`, content toolbar `--surface-2`,
		   this `--surface-3` — so each adjacency is a real step and none needs a hairline (D5), not
		   even the one 26px inside the panel edge. `--surface-2` put this byte-identical to the `Bar`
		   that four of the six panel types render flush beneath it, which is the same "border deleted,
		   nothing behind it" defect one level down; a step is what D5 trades the border FOR. */
		background: var(--surface-3);
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
