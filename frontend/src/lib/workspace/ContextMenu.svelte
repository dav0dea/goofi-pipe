<!--
  Reusable popup menu — portaled to <body> so it stacks above panels and
  SvelteFlow viewports. Supports one-deep (recursive) submenus on hover, a
  checked marker, separators, and disabled items. Closes on Escape or any
  pointerdown outside any `.context-menu` element (so submenu clicks count as
  inside). Shared by panel headers, the content dropdown, and tab menus.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import type { MenuItem } from './menu';
	import { portal } from './portal';
	import Self from './ContextMenu.svelte';

	let {
		x,
		y,
		items,
		onClose,
		root = true
	}: {
		x: number;
		y: number;
		items: MenuItem[];
		onClose: () => void;
		root?: boolean;
	} = $props();

	let menuEl = $state<HTMLDivElement | null>(null);
	// Initial spawn point only; the $effect below re-clamps it to the viewport.
	let pos = $state(untrack(() => ({ x, y })));
	let openSub = $state<{ index: number; x: number; y: number } | null>(null);

	$effect(() => {
		if (!menuEl) return;
		const r = menuEl.getBoundingClientRect();
		let nx = x;
		let ny = y;
		if (nx + r.width > window.innerWidth - 6) nx = Math.max(6, window.innerWidth - r.width - 6);
		if (ny + r.height > window.innerHeight - 6) ny = Math.max(6, window.innerHeight - r.height - 6);
		pos = { x: nx, y: ny };
	});

	function pick(item: MenuItem): void {
		if (item.disabled || item.separator || item.items) return;
		item.action?.();
		onClose();
	}

	function hover(item: MenuItem, index: number, e: MouseEvent): void {
		if (item.items && !item.disabled) {
			const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
			openSub = { index, x: r.right - 3, y: r.top - 5 };
		} else {
			openSub = null;
		}
	}

	function onWindowPointerDown(e: PointerEvent): void {
		const t = e.target as HTMLElement | null;
		if (!t || !t.closest('.context-menu')) onClose();
	}
	function onWindowKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') onClose();
	}
</script>

<!-- Only the root menu listens for outside-click / Escape; submenus share the
     same `onClose`, and the `.context-menu` closest() check treats them as
     inside. svelte:window must be top-level, so the handlers are nulled out
     for submenus rather than the tag being conditionally rendered. -->
<svelte:window
	onpointerdown={root ? onWindowPointerDown : undefined}
	onkeydown={root ? onWindowKeydown : undefined}
/>

<div
	bind:this={menuEl}
	class="context-menu"
	style="left:{pos.x}px; top:{pos.y}px"
	use:portal
	role="menu"
	tabindex="-1"
>
	{#each items as item, i (i)}
		{#if item.separator}
			<div class="sep"></div>
		{:else}
			<button
				class="item"
				class:checkable={items.some((it) => it.checked !== undefined)}
				disabled={item.disabled}
				onclick={() => pick(item)}
				onmouseenter={(e) => hover(item, i, e)}
				role="menuitem"
			>
				<span class="check">{item.checked ? '✓' : ''}</span>
				{#if item.icon}<span class="ic">{item.icon}</span>{/if}
				<span class="label">{item.label}</span>
				{#if item.items}<span class="arrow">▸</span>{/if}
			</button>
		{/if}
	{/each}

	{#if openSub}
		{@const sub = items[openSub.index]}
		{#if sub?.items}
			<Self x={openSub.x} y={openSub.y} items={sub.items} {onClose} root={false} />
		{/if}
	{/if}
</div>

<style>
	.context-menu {
		position: fixed;
		z-index: var(--z-menu);
		min-width: 180px;
		padding: 4px;
		background: var(--bg-elev-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-2);
		display: flex;
		flex-direction: column;
		gap: 1px;
		font-size: 0.85rem;
		user-select: none;
	}
	.item {
		display: flex;
		align-items: center;
		gap: 6px;
		width: 100%;
		padding: 5px 8px;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text);
		text-align: left;
		cursor: pointer;
	}
	.item:hover:not(:disabled) {
		background: var(--accent);
		color: #0a0c10;
	}
	.item:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.check {
		width: 12px;
		flex: 0 0 12px;
		font-size: 0.8em;
	}
	.ic {
		width: 14px;
		text-align: center;
	}
	.label {
		flex: 1;
		white-space: nowrap;
	}
	.arrow {
		opacity: 0.6;
		font-size: 0.8em;
	}
	.sep {
		height: 1px;
		margin: 3px 4px;
		background: var(--border);
	}
</style>
