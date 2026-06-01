<!--
  Workspace tab strip. Click to switch, double-click to rename inline, a small
  × to close, and `＋` to add a tab. Tabs are draggable: dropped on another tab
  they reorder; dragged onto a panel's drop zone they split it (see Panel.svelte
  / the workspace store's `draggingTabId`).
-->
<script lang="ts">
	import { workspace } from './workspace.svelte';

	const ws = workspace();
	const tabs = $derived(ws.state.workspaces);
	const activeId = $derived(ws.state.activeWorkspaceId);

	let editing = $state<string | null>(null);
	let editValue = $state('');
	let dragIndex = $state<number | null>(null);
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
</script>

<div class="tabs" data-testid="workspace-tabs">
	{#each tabs as tab, i (tab.id)}
		<div
			class="tab"
			class:active={tab.id === activeId}
			class:droptarget={dropIndex === i && dragIndex !== i}
			draggable={editing !== tab.id}
			onclick={() => ws.selectTab(tab.id)}
			ondblclick={() => startRename(tab.id, tab.name)}
			ondragstart={() => {
				dragIndex = i;
				ws.draggingTabId = tab.id;
			}}
			ondragover={(e) => {
				e.preventDefault();
				dropIndex = i;
			}}
			ondragend={() => {
				dragIndex = null;
				dropIndex = null;
				ws.draggingTabId = null;
			}}
			ondrop={(e) => {
				e.preventDefault();
				if (dragIndex !== null && dragIndex !== i) ws.reorderTab(dragIndex, i);
				dragIndex = null;
				dropIndex = null;
				ws.draggingTabId = null;
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
	<button class="add" onclick={() => ws.addTab()} title="New tab" aria-label="New tab">＋</button>
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
	.tab.droptarget {
		border-color: var(--accent);
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
