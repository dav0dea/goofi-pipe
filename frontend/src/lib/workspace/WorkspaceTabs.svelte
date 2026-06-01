<!--
  Workspace tab strip. Click to switch, double-click to rename inline,
  right-click for rename/duplicate/close, and `＋` to add a tab. Each tab is an
  independent layout tree.
-->
<script lang="ts">
	import { workspace } from './workspace.svelte';
	import type { MenuItem } from './menu';
	import ContextMenu from './ContextMenu.svelte';

	const ws = workspace();
	const tabs = $derived(ws.state.workspaces);
	const activeId = $derived(ws.state.activeWorkspaceId);

	let editing = $state<string | null>(null);
	let editValue = $state('');
	let menu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);
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
	function onTabContext(e: MouseEvent, id: string, name: string): void {
		e.preventDefault();
		menu = {
			x: e.clientX,
			y: e.clientY,
			items: [
				{ label: 'Rename', action: () => startRename(id, name) },
				{ label: 'Duplicate', action: () => ws.duplicateTab(id) },
				{ separator: true },
				{ label: 'Close Tab', disabled: tabs.length <= 1, action: () => ws.closeTab(id) }
			]
		};
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
			oncontextmenu={(e) => onTabContext(e, tab.id, tab.name)}
			ondragstart={() => (dragIndex = i)}
			ondragover={(e) => {
				e.preventDefault();
				dropIndex = i;
			}}
			ondragend={() => {
				dragIndex = null;
				dropIndex = null;
			}}
			ondrop={(e) => {
				e.preventDefault();
				if (dragIndex !== null && dragIndex !== i) ws.reorderTab(dragIndex, i);
				dragIndex = null;
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
			{/if}
		</div>
	{/each}
	<button class="add" onclick={() => ws.addTab()} title="New tab" aria-label="New tab">＋</button>
</div>

{#if menu}
	<ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => (menu = null)} />
{/if}

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
		padding: 0 12px;
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
