<!--
  Tabs — the integrated connected tab bar. NOT floating pills and NOT underlined tabs: the ACTIVE
  tab drops to the BODY surface (`--tab-body`) while the inactive tabs and the strip sit at the
  HEADER surface (`--tab-surface`), so the active tab visually merges downward into the body
  rendered flush beneath it — one connected piece, no divider lines. A consumer paints its body
  region with the same `--tab-body` to complete the seam.

  A horizontal WAI-ARIA tablist: `role=tablist` of `role=tab` buttons, roving tabindex (the active
  tab is the one tab-stop), Left/Right + Home/End move the selection AND focus (automatic
  activation), `aria-selected` marks the active tab. The active-resolution + arrow-navigation logic
  is the pure, unit-tested `tabsState`. `items` in, `onSelect(id)` out — dumb.

  This is a SEGMENTED CONTROL, and the workspace's page strip is not one of its consumers: that is a
  tab group's header, which panelty draws, and which carries a drag bus, a drop preview and a per-tab
  ✕ that nothing here has any business knowing about.
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
	let tabEls = $state<HTMLElement[]>([]);

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
			<span class="ui-tab-label">{item.label}</span>
		</button>
	{/each}
</div>

<style>
	.ui-tabs {
		display: flex;
		align-items: flex-end;
		gap: var(--space-1);
		min-width: 0;
		padding: var(--space-2) var(--space-2) 0;
		background: var(--tab-surface, var(--surface-2));
		overflow-x: auto;
		overflow-y: hidden;
		scrollbar-width: none;
	}
	.ui-tab {
		display: inline-flex;
		align-items: center;
		flex: 0 0 auto;
		min-height: var(--hit);
		padding: 0 var(--space-4);
		border: 0;
		border-radius: var(--radius-sm) var(--radius-sm) 0 0;
		background: transparent;
		color: var(--text-dim);
		font-family: var(--font-sans);
		font-size: var(--tab-fs, var(--fs-small));
		line-height: 1;
		white-space: nowrap;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-tab:hover {
		color: var(--text);
	}
	/* The connected look: the active tab drops to the surface the body beneath it paints. */
	.ui-tab.active {
		background: var(--tab-body, var(--surface-1));
		color: var(--text);
	}
	.ui-tab-label {
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>
