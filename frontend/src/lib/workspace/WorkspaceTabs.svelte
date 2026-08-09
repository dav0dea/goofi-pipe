<!--
  Workspace tab strip — the ui `Tabs` primitive with its opt-in affordances switched on (Phil,
  2026-08-08: one component for the layout pages AND the inspector's param groups, so the two can
  never drift apart visually). Click to switch, double-click to rename inline, a hover-revealed ✕
  to close, ＋ to add — all the primitive's; what stays HERE is the drag system, because it is one
  half of the workspace-wide drag (a tab dragged onto another tab reorders; a tab or panel dragged
  onto a panel splits it; a panel dragged onto this bar becomes a new tab), and that lives with the
  workspace store the `$lib/ui` leaf layer must not import. The primitive's half of the seam is
  `tabProps` (per-tab drag attributes) and `previewIndex` (the drop-slot placeholder).
-->
<script lang="ts">
	import { workspace } from './workspace.svelte';
	import { Tabs } from '$lib/ui';

	const ws = workspace();
	const tabs = $derived(ws.state.workspaces);
	const activeId = $derived(ws.state.activeWorkspaceId);

	let dropIndex = $state<number | null>(null);

	// While a panel/tab is dragged over the bar, open a real placeholder slot at the drop index so
	// it's clear where it'll land (and the ＋ shifts to make room) — not a thin insertion sliver.
	const showPreview = $derived(!!ws.dragging && dropIndex !== null);

	function computeDropIndex(container: HTMLElement, clientX: number): number {
		const els = Array.from(container.querySelectorAll('.ui-tab')) as HTMLElement[];
		for (let i = 0; i < els.length; i++) {
			const r = els[i].getBoundingClientRect();
			if (clientX < r.left + r.width / 2) return i;
		}
		return els.length;
	}

	function onBarDragOver(e: DragEvent): void {
		if (!ws.dragging) return;
		e.preventDefault();
		dropIndex = computeDropIndex(e.currentTarget as HTMLElement, e.clientX);
	}
	function onBarDragLeave(e: DragEvent): void {
		if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) dropIndex = null;
	}
	function onBarDrop(e: DragEvent): void {
		const d = ws.dragging;
		const idx = dropIndex ?? tabs.length;
		dropIndex = null;
		if (!d) return;
		e.preventDefault();
		if (d.kind === 'panel') {
			ws.dropPanelOnTabBar(idx); // becomes a new tab (clears dragging itself)
		} else {
			const from = tabs.findIndex((t) => t.id === d.workspaceId);
			if (from >= 0) ws.reorderTab(from, idx > from ? idx - 1 : idx);
			ws.dragging = null;
		}
	}
</script>

<div
	class="strip thin-scrollbar"
	class:dragover={!!ws.dragging}
	ondragover={onBarDragOver}
	ondragleave={onBarDragLeave}
	ondrop={onBarDrop}
	role="presentation"
>
	<Tabs
		class="wtabs"
		data-testid="workspace-tabs"
		items={tabs.map((t) => ({ id: t.id, label: t.name }))}
		active={activeId}
		onSelect={(id) => ws.selectTab(id)}
		onAdd={() => ws.addTab()}
		onRename={(id, name) => ws.renameTab(id, name)}
		onClose={tabs.length > 1 ? (id) => ws.closeTab(id) : undefined}
		previewIndex={showPreview ? dropIndex : null}
		tabProps={(item) => ({
			draggable: true,
			ondragstart: () => (ws.dragging = { kind: 'tab', workspaceId: item.id }),
			ondragend: () => {
				ws.dragging = null;
				dropIndex = null;
			}
		})}
	/>
</div>

<style>
	/* The scroller around the strip: fills the TopBar's central slot and scrolls what the
	   reserve-driven overflow plan could not keep (the plan reserves the strip's CONTENT width,
	   so this scrolls only once even an otherwise-empty bar could not hold every tab). */
	.strip {
		display: flex;
		align-items: stretch;
		height: 100%;
		flex: 1 1 auto;
		min-width: 0;
		overflow-x: auto;
		overflow-y: hidden;
		/* The primitive's two surface hooks, retuned for the HEADER context (the documented
		   per-instance CSS-var escape hatch): the strip blends with the bar rather than painting
		   its own header surface, and the ACTIVE page drops to the workspace ground below the
		   bar — the same connected look the inspector's param tabs make with their panel body. */
		--tabs-surface: transparent;
		--tabs-body: var(--bg);
	}
	.strip.dragover {
		background: color-mix(in srgb, var(--accent) 7%, transparent);
		border-radius: var(--radius-sm);
	}
	/* Sizing the child's box is the parent's layout prerogative — through the class THIS component
	   passes in (never the primitive's own .ui-* class): the strip is exactly the bar's height,
	   and the tablist fills it so the bottom-aligned active tab lands flush on the bar's lower
	   edge — where the workspace it merges with begins. */
	.strip :global(.wtabs) {
		height: 100%;
		flex: 1 0 auto;
	}
</style>
