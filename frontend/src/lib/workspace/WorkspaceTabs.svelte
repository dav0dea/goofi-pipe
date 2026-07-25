<!--
  Workspace tab strip. Click to switch, double-click to rename inline, a small
  × to close, `＋` to add.

  Tabs and panels share one drag system (workspace store `dragging`): a tab
  dragged onto another tab reorders; a tab or panel dragged onto a panel splits
  it; and a panel dragged onto this bar becomes a new tab. The bar shows an
  insertion marker at the drop index while a drag is over it.
-->
<script lang="ts">
	import { workspace } from './workspace.svelte';
	import { IconButton } from '$lib/ui';

	const ws = workspace();
	const tabs = $derived(ws.state.workspaces);
	const activeId = $derived(ws.state.activeWorkspaceId);

	let editing = $state<string | null>(null);
	let editValue = $state('');
	let dropIndex = $state<number | null>(null);

	// While a panel/tab is dragged over the bar, open a real placeholder slot at
	// the drop index so it's clear where it'll land (and the ＋ shifts to make
	// room) — rather than a thin insertion sliver.
	const showPreview = $derived(!!ws.dragging && dropIndex !== null);

	function startRename(id: string, name: string): void {
		editing = id;
		editValue = name;
	}
	function commitRename(): void {
		if (editing) ws.renameTab(editing, editValue);
		editing = null;
	}
	function focusInput(node: HTMLInputElement): void {
		node.focus();
		node.select();
	}

	function computeDropIndex(container: HTMLElement, clientX: number): number {
		const els = Array.from(container.querySelectorAll('.tab')) as HTMLElement[];
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
	class="tabs"
	data-testid="workspace-tabs"
	class:dragover={!!ws.dragging}
	ondragover={onBarDragOver}
	ondragleave={onBarDragLeave}
	ondrop={onBarDrop}
	role="tablist"
	tabindex="-1"
>
	{#each tabs as tab, i (tab.id)}
		{#if showPreview && dropIndex === i}
			<div class="tab-preview" aria-hidden="true"></div>
		{/if}
		<div
			class="tab"
			class:active={tab.id === activeId}
			draggable={editing !== tab.id}
			onclick={() => ws.selectTab(tab.id)}
			ondblclick={() => startRename(tab.id, tab.name)}
			ondragstart={() => (ws.dragging = { kind: 'tab', workspaceId: tab.id })}
			ondragend={() => {
				ws.dragging = null;
				dropIndex = null;
			}}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') ws.selectTab(tab.id);
			}}
			role="tab"
			tabindex="0"
			aria-selected={tab.id === activeId}
		>
			{#if editing === tab.id}
				<!-- svelte-ignore a11y_autofocus -->
				<input
					class="rename"
					value={editValue}
					oninput={(e) => (editValue = e.currentTarget.value)}
					onblur={commitRename}
					onkeydown={(e) => {
						if (e.key === 'Enter') commitRename();
						else if (e.key === 'Escape') editing = null;
					}}
					use:focusInput
				/>
			{:else}
				<span class="name">{tab.name}</span>
				{#if tabs.length > 1}
					<IconButton
						variant="ghost"
						size="sm"
						class="close"
						label="Close tab"
						onclick={(e) => {
							e.stopPropagation();
							ws.closeTab(tab.id);
						}}>✕</IconButton
					>
				{/if}
			{/if}
		</div>
	{/each}
	{#if showPreview && dropIndex === tabs.length}
		<div class="tab-preview" aria-hidden="true"></div>
	{/if}
	<IconButton
		variant="ghost"
		size="sm"
		density="chrome"
		class="add"
		label="New tab"
		onclick={() => ws.addTab()}>＋</IconButton
	>
</div>

<style>
	.tabs {
		/* Lives inside the TopBar's central slot: fills the slack, blends with
		   the header (no own background / bottom border), matches its height. */
		display: flex;
		align-items: center;
		gap: var(--space-1);
		height: 100%;
		flex: 1 1 auto;
		min-width: 0;
		padding: 0 var(--space-2);
		background: transparent;
		overflow-x: auto;
		overflow-y: hidden;
		scrollbar-width: thin;
	}
	.tabs.dragover {
		background: color-mix(in srgb, var(--accent) 7%, transparent);
		border-radius: var(--radius-sm);
	}
	.tab {
		display: flex;
		align-items: center;
		gap: 0;
		/* Even padding on all sides; vertically centred in the header (no margin)
		   so the pill sits neatly inside the bar rather than filling its height. */
		padding: var(--space-3) var(--space-6);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		font-size: var(--fs-small);
		line-height: 1;
		white-space: nowrap;
		cursor: pointer;
		user-select: none;
		border: 1px solid transparent;
	}
	.tab:hover {
		background: var(--surface-2);
		color: var(--text);
	}
	.tab.active {
		background: var(--surface-3);
		color: var(--text);
		border-color: var(--border);
	}
	/* Drop placeholder: a tab-sized slot that opens up at the drop index so the
	   landing spot is obvious and the ＋ shifts over to make room. */
	.tab-preview {
		flex: 0 0 auto;
		align-self: center;
		width: 96px;
		height: 26px;
		border-radius: var(--radius-sm);
		border: 1px dashed var(--accent);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.rename {
		width: 9ch;
		padding: 1px var(--space-2);
		font: inherit;
		font-size: var(--fs-small);
	}
	/* Collapsed when hidden so an inactive tab stays evenly padded; expands (with its left
	   gap) only on hover / when active. The collapse forces the box under the primitive's
	   --hit floor, and `overflow: hidden` also clips IconButton's coarse hit-rect ::after.
	   `border: 0` because under border-box a 1px border clamps `width: 0` to 2px — the ghost
	   border is invisible anyway, and without this every inactive tab gains 2px.
	   Deliberately NOT `density="chrome"` (which the ＋ above uses): that density means "dense on
	   fine, floored back to --hit on coarse", and this control is sub-floor on BOTH pointers — the
	   collapse needs `min-width: 0` to reach zero width, which the density's floor would undo.
	   TODO(R): this reveal is hover-only AND sub-floor on touch — it needs a coarse-pointer
	   door of its own, not a patch here. */
	.tab :global(.close) {
		width: 0;
		min-width: 0;
		height: 16px;
		min-height: 16px;
		margin-left: 0;
		border: 0;
		overflow: hidden;
		line-height: 1;
		color: var(--text-muted);
		opacity: 0;
	}
	.tab:hover :global(.close),
	.tab.active :global(.close) {
		width: 16px;
		margin-left: var(--space-2);
		opacity: 1;
	}
	/* Only the colour; the hover surface is the primitive's. */
	.tab :global(.close:hover) {
		color: var(--danger);
	}
	/* The frozen 22px box: the ~23px tab pills sit beside it, so the primitive's --hit floor
	   (28px fine) would leave the ＋ standing taller than the strip it belongs to. Stating the
	   box is all this strip does — `density="chrome"` restores the floor under a coarse pointer. */
	.tabs :global(.add) {
		--icon-btn-size: 22px;
	}
</style>
