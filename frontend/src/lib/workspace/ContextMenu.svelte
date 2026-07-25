<!--
  Reusable popup menu — portaled to <body> so it stacks above panels and
  SvelteFlow viewports. Supports one-deep (recursive) submenus on hover, a
  checked marker, separators, and disabled items. Closes on Escape or any
  pointerdown outside any `.context-menu` element (so submenu clicks count as
  inside). Shared by panel headers, the content dropdown, and tab menus.

  Deliberately NOT built on the `Popover` primitive: this menu is anchored to a
  *point* (a right-click's clientX/clientY) where Popover clamps against an anchor
  *element*, and each submenu portals to <body> as its own root — so Popover's
  `menuEl.contains(target)` outside-test would read a submenu click as outside and
  dismiss the menu. The `.context-menu` closest() test above is what makes submenus
  work. Popover would buy the clamp and cost the submenus.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import type { MenuItem } from './menu';
	import { portal } from './portal';
	import { clampToViewport } from '$lib/ui';
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
		// The spawn point is a degenerate anchor rect — `left`/`bottom` are the point itself, so
		// the shared clamp's "flush under the anchor's bottom-left" origin IS the click position.
		const p = clampToViewport(
			{ left: x, top: y, right: x, bottom: y, width: 0, height: 0 },
			{ width: r.width, height: r.height },
			{ width: window.innerWidth, height: window.innerHeight }
		);
		pos = { x: p.left, y: p.top };
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
		/* px, not rem: TopBar's save-menu spawn point clamps against this same 180. */
		min-width: 180px;
		padding: var(--space-2);
		background: var(--surface-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-2);
		display: flex;
		flex-direction: column;
		gap: 1px;
		font-size: var(--fs-small);
		user-select: none;
	}
	/* A menu row, not a Button: it keeps its own complete style (full-bleed, left-aligned,
	   accent-filled on hover) — the same carve-out the other list/menu rows take. */
	.item {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		width: 100%;
		padding: var(--space-3) var(--space-4);
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text);
		text-align: left;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.item:hover:not(:disabled) {
		background: var(--accent);
		color: var(--on-accent);
	}
	.item:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
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
		margin: var(--space-1) var(--space-2);
		background: var(--border);
	}
</style>
