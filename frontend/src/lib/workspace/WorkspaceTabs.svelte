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

	const ws = workspace();
	const tabs = $derived(ws.state.workspaces);
	const activeId = $derived(ws.state.activeWorkspaceId);

	let editing = $state<string | null>(null);
	let editValue = $state('');
	let dropIndex = $state<number | null>(null);

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
		<div
			class="tab"
			class:active={tab.id === activeId}
			class:insert={dropIndex === i}
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
					<button
						class="close"
						title="Close tab"
						aria-label="Close tab"
						onclick={(e) => {
							e.stopPropagation();
							ws.closeTab(tab.id);
						}}>✕</button
					>
				{/if}
			{/if}
		</div>
	{/each}
	<button
		class="add"
		class:insert={dropIndex === tabs.length}
		onclick={() => ws.addTab()}
		title="New tab"
		aria-label="New tab">＋</button
	>
</div>

<style>
	.tabs {
		display: flex;
		align-items: stretch;
		gap: 2px;
		height: var(--tabs-h, 30px);
		flex: 0 0 auto;
		padding: 0 6px;
		background: var(--bg-elev-1);
		border-bottom: 1px solid var(--border);
		overflow-x: auto;
		overflow-y: hidden;
	}
	.tabs.dragover {
		background: color-mix(in srgb, var(--accent) 8%, var(--bg-elev-1));
	}
	.tab {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 6px 0 12px;
		margin: 4px 0;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		font-size: 0.82rem;
		white-space: nowrap;
		cursor: pointer;
		user-select: none;
		border: 1px solid transparent;
	}
	.tab:hover {
		background: var(--bg-elev-2);
		color: var(--text);
	}
	.tab.active {
		background: var(--bg-elev-3);
		color: var(--text);
		border-color: var(--border);
	}
	/* Insertion marker: an accent bar on the left edge of the drop slot. */
	.tab.insert,
	.add.insert {
		box-shadow: inset 2px 0 0 0 var(--accent);
	}
	.rename {
		width: 9ch;
		padding: 1px 4px;
		font: inherit;
		font-size: 0.82rem;
	}
	.close {
		display: grid;
		place-items: center;
		width: 16px;
		height: 16px;
		padding: 0;
		font-size: 0.7rem;
		line-height: 1;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-faint);
		opacity: 0;
		cursor: pointer;
	}
	.tab:hover .close,
	.tab.active .close {
		opacity: 1;
	}
	.close:hover {
		background: var(--bg-elev-1);
		color: var(--danger);
	}
	.add {
		align-self: center;
		width: 22px;
		height: 22px;
		display: grid;
		place-items: center;
		padding: 0;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
	}
	.add:hover {
		background: var(--bg-elev-2);
		color: var(--text);
	}
</style>
