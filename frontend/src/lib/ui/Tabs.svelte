<!--
  Tabs — the integrated connected tab bar (spec §2.4). NOT floating pills and NOT underlined tabs:
  the ACTIVE tab drops to the BODY surface (`--tabs-body`) while the inactive tabs and the strip sit
  at the HEADER surface (`--tabs-surface`), so the active tab visually merges downward into the panel
  body rendered flush beneath it — one connected piece, no divider lines. A consumer paints its body
  region with the same `--tabs-body` token to complete the seam.

  A horizontal WAI-ARIA tablist: `role=tablist` of `role=tab` buttons, roving tabindex (the active
  tab is the one tab-stop), Left/Right + Home/End move the selection AND focus (automatic activation),
  `aria-selected` marks the active tab. The active-resolution + arrow-navigation logic is the pure,
  unit-tested `tabsState` (so an unset/stale `active` still resolves to a shown tab). `items` in,
  `onSelect(id)` out — dumb. `class` merged, `data-testid` (and any other attribute) forwarded.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { resolveActive, nextIndex, type TabItem, type ArrowKey } from './tabsState';

	let {
		items,
		active,
		onSelect,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		items: TabItem[];
		/** The selected tab id; unset or stale resolves to the first tab (see `resolveActive`). */
		active?: string;
		onSelect: (id: string) => void;
	} = $props();

	const resolved = $derived(resolveActive(items, active));
	// Roving-tabindex targets, bound per tab so arrow keys can move DOM focus with the selection.
	let tabEls = $state<HTMLButtonElement[]>([]);

	const ARROW_KEYS: ArrowKey[] = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];

	function onKeydown(e: KeyboardEvent): void {
		if (!ARROW_KEYS.includes(e.key as ArrowKey)) return;
		e.preventDefault();
		const current = items.findIndex((it) => it.id === resolved);
		const ni = nextIndex(current, items.length, e.key as ArrowKey);
		if (ni < 0) return;
		onSelect(items[ni].id);
		tabEls[ni]?.focus();
	}
</script>

<div {...rest} class={`ui-tabs ${klass}`.trim()} role="tablist">
	{#each items as item, i (item.id)}
		<button
			bind:this={tabEls[i]}
			type="button"
			role="tab"
			class="ui-tab"
			class:active={item.id === resolved}
			aria-selected={item.id === resolved}
			tabindex={item.id === resolved ? 0 : -1}
			onclick={() => onSelect(item.id)}
			onkeydown={onKeydown}
		>
			{item.label}
		</button>
	{/each}
</div>

<style>
	/* The strip sits at the header surface; the active tab drops out of it onto the body surface. */
	.ui-tabs {
		display: flex;
		align-items: flex-end;
		gap: var(--space-1);
		min-width: 0;
		padding: var(--space-2) var(--space-2) 0;
		background: var(--tabs-surface, var(--surface-2));
		font-family: var(--font-mono);
	}
	.ui-tab {
		flex: 0 1 auto;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-height: var(--hit);
		padding: var(--space-3) var(--space-6);
		font-size: var(--fs-small);
		/* Inactive tabs read as part of the header strip. */
		background: var(--tabs-surface, var(--surface-2));
		color: var(--text-dim);
		border: none;
		border-radius: var(--radius-sm) var(--radius-sm) 0 0;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-tab:hover:not(.active) {
		background: var(--surface-3);
		color: var(--text);
	}
	/* The connected look: the active tab drops to the body surface so it merges with the panel body
	   painted flush beneath it (same `--tabs-body` token) — one piece, no line. */
	.ui-tab.active {
		background: var(--tabs-body, var(--surface-1));
		color: var(--text);
		font-weight: 600;
		cursor: default;
	}
	/* Keyboard focus ring — the app :focus-visible convention (never suppressed). */
	.ui-tab:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -2px;
	}
</style>
