<!--
  Reusable popup menu — portaled to <body> so it stacks above panels and
  SvelteFlow viewports. Supports one-deep (recursive) submenus — a mouse opens
  one by hovering the parent row, any pointer by TAPPING it (R-Task 7: they
  were `mouseenter`-only and a click on a parent row was an explicit no-op, so
  "Change content ▸" could not be expanded by touch at all) — plus a
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
	import { clampToViewport, overlayViewport } from '$lib/ui';
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

	// The two gutters are decided ONCE PER MENU, not per row: a row with no icon still reserves the
	// column its neighbour's icon needs, so every label in a menu starts in the same place — and a
	// menu with no icons (or nothing checkable) pays for neither. The check gutter used to be
	// unconditional and the icon gutter per-row, which is exactly backwards: it left two label
	// columns in the one menu that mixes them (the header's overflow) and a permanent 12px indent
	// in the ones that check nothing.
	const checkable = $derived(items.some((it) => it.checked !== undefined));
	const iconic = $derived(items.some((it) => it.icon));

	let menuEl = $state<HTMLDivElement | null>(null);
	// Initial spawn point only; the $effect below re-clamps it to the viewport.
	let pos = $state(untrack(() => ({ x, y })));
	let openSub = $state<{ index: number; x: number; y: number } | null>(null);

	$effect(() => {
		if (!menuEl) return;
		const r = menuEl.getBoundingClientRect();
		// The spawn point is a degenerate anchor rect — `left`/`bottom` are the point itself, so
		// the shared clamp's "flush under the anchor's bottom-left" origin IS the click position.
		// `overlayViewport()` is the same measurement Popover clamps against, so a menu opened with
		// the soft keyboard up does not land underneath it. No visualViewport listener here (Popover
		// has one): a menu row cannot take focus into a text field, so the keyboard state cannot
		// change while THIS surface is open — it is only ever wrong at open time.
		const p = clampToViewport(
			{ left: x, top: y, right: x, bottom: y, width: 0, height: 0 },
			{ width: r.width, height: r.height },
			overlayViewport()
		);
		pos = { x: p.left, y: p.top };
	});

	function openSubAt(index: number, row: HTMLElement): void {
		const r = row.getBoundingClientRect();
		openSub = { index, x: r.right - 3, y: r.top - 5 };
	}

	function pick(item: MenuItem, index: number, e: MouseEvent): void {
		if (item.disabled || item.separator) return;
		// A parent row OPENS its submenu rather than toggling it: on a fine pointer the hover has
		// already opened it, and a toggle would make the click that follows the hover close it.
		if (item.items) {
			openSubAt(index, e.currentTarget as HTMLElement);
			return;
		}
		item.action?.();
		onClose();
	}

	/** Hover-to-expand, for a MOUSE only. `pointerenter` rather than `mouseenter` so the pointer
	 * type is knowable: a tap fires the compatibility mouse events too, and it fires them BEFORE
	 * the click — so an unfiltered hover would open the submenu and the tap that caused it would
	 * then be the second event on the same row. */
	function hover(item: MenuItem, index: number, e: PointerEvent): void {
		if (e.pointerType !== 'mouse') return;
		if (item.items && !item.disabled) openSubAt(index, e.currentTarget as HTMLElement);
		else openSub = null;
	}

	function onWindowPointerDown(e: PointerEvent): void {
		const t = e.target as HTMLElement | null;
		if (!t || !t.closest('.context-menu')) onClose();
	}
	function onWindowKeydown(e: KeyboardEvent): void {
		if (e.key !== 'Escape') return;
		// CONSUMED at capture phase — the dismissal belongs to the topmost surface, and a
		// bubble-phase listener loses the race to `NodeEditorPanel`'s. `ui/Popover` states the full
		// reasoning where the same rule now lives for every other anchored surface in the app.
		e.stopPropagation();
		onClose();
	}
</script>

<!-- Only the root menu listens for outside-click / Escape; submenus share the
     same `onClose`, and the `.context-menu` closest() check treats them as
     inside. svelte:window must be top-level, so the handlers are nulled out
     for submenus rather than the tag being conditionally rendered. -->
<svelte:window
	onpointerdown={root ? onWindowPointerDown : undefined}
	onkeydowncapture={root ? onWindowKeydown : undefined}
/>

<div
	bind:this={menuEl}
	class="context-menu thin-scrollbar"
	style="left:{pos.x}px; top:{pos.y}px"
	use:portal
	role="menu"
	tabindex="-1"
>
	{#each items as item, i (i)}
		{#if item.separator}
			<div class="sep"></div>
		{:else}
			<!-- The three glyph spans are DECORATION: `aria-hidden` keeps them out of the row's
			     accessible name, which is the label alone. Checked-ness is then said in the tree
			     rather than drawn with a ✓ only a sighted user can read. -->
			<button
				class="item"
				disabled={item.disabled}
				onclick={(e) => pick(item, i, e)}
				onpointerenter={(e) => hover(item, i, e)}
				role={item.checked === undefined ? 'menuitem' : 'menuitemcheckbox'}
				aria-checked={item.checked}
			>
				{#if checkable}<span class="check" aria-hidden="true">{item.checked ? '✓' : ''}</span>{/if}
				{#if iconic}<span class="ic" aria-hidden="true">{item.icon ?? ''}</span>{/if}
				<span class="label">{item.label}</span>
				{#if item.items}<span class="arrow" aria-hidden="true">▸</span>{/if}
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
		/* The clamp can only SHIFT a surface that fits — `Math.max(MARGIN, …)` floors an oversized
		   one at 6px and lets the rest run off the bottom, which is how the TopBar's overflow menu
		   (13 rows at the 44px coarse floor) put its canvas commands under a 360px landscape phone
		   with no other pointer door. `12px` is the clamp's own MARGIN on both edges. Submenus
		   portal to <body> as their own roots, so this scroller cannot clip them. */
		max-height: calc(100dvh - 12px);
		overflow-y: auto;
		overscroll-behavior: contain;
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
	   accent-filled on hover) — the same carve-out the other list/menu rows take. "Complete"
	   includes the font: a <button> does NOT inherit one, so without this the row falls back to
	   the UA face the moment M-Task 7 strips app.css's base `button { font: inherit }`. */
	.item {
		font: inherit;
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
		font-size: var(--fs-micro);
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
		/* opacity: intentional — the submarker is quieter than the label it follows; the row it
		   sits in is fully enabled (a disabled row dims wholesale via --disabled-opacity). */
		opacity: 0.6;
		font-size: var(--fs-micro);
	}
	.sep {
		height: 1px;
		margin: var(--space-1) var(--space-2);
		background: var(--border);
	}
</style>
