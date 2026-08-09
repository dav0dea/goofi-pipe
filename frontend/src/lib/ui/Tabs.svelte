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

  ONE component, two consumers (Phil, 2026-08-08): the inspector's param groups AND the header's
  layout pages are the same control, so the extra affordances the layout bar needs are OPT-IN props
  that default off — absent, this renders exactly the bare tablist the inspector always had:
   · `onAdd`      — a trailing ＋ that mints a new tab.
   · `onRename`   — double-click (double-tap) opens an inline rename; Enter commits, Escape cancels,
                    blur commits.
   · `onClose`    — a hover-revealed ✕ per tab (rested open under a coarse pointer — C17's door).
   · `onReorder`  is deliberately NOT here: the layout bar's drag is one half of the workspace-wide
                    drag system (a PANEL dropped on the bar becomes a tab), which lives with the
                    workspace store — and `$lib/ui` is a leaf layer that must not import stores. The
                    seam is `tabProps` (per-tab attributes: draggable, ondragstart, …) plus
                    `previewIndex` (the drop-slot placeholder), so the consumer owns the drag and
                    this component only draws it.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import Icon from './Icon.svelte';
	import { resolveActive, nextIndex, type TabItem, type ArrowKey } from './tabsState';
	import { MODE_ATTRS } from './inputMode';

	let {
		items,
		active,
		onSelect,
		onAdd,
		onRename,
		onClose,
		previewIndex = null,
		tabProps,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		items: TabItem[];
		/** The selected tab id; unset or stale resolves to the first tab (see `resolveActive`). */
		active?: string;
		onSelect: (id: string) => void;
		/** Renders the trailing ＋ ("New tab"). Absent → no add affordance. */
		onAdd?: () => void;
		/** Enables double-click inline rename. Absent → labels are static. */
		onRename?: (id: string, label: string) => void;
		/** Renders a hover-revealed ✕ per tab. Absent → tabs cannot be closed. The "keep the last
		 *  tab" policy belongs to the consumer: pass `undefined` when only one tab remains. */
		onClose?: (id: string) => void;
		/** Draw the drop-slot placeholder before this item index (items.length = at the end).
		 *  Null → none. The drag itself is the consumer's — see the header comment. */
		previewIndex?: number | null;
		/** Extra attributes for each tab (draggable, ondragstart, …) — the consumer's half of the
		 *  drag seam. */
		tabProps?: (item: TabItem) => HTMLAttributes<HTMLDivElement>;
	} = $props();

	const resolved = $derived(resolveActive(items, active));
	// Roving-tabindex targets, bound per tab so arrow keys can move DOM focus with the selection.
	let tabEls = $state<HTMLElement[]>([]);

	let editing = $state<string | null>(null);
	let editValue = $state('');

	function startRename(item: TabItem): void {
		if (!onRename) return;
		editing = item.id;
		editValue = item.label;
	}
	function commitRename(): void {
		if (editing && onRename) onRename(editing, editValue);
		editing = null;
	}
	function focusInput(node: HTMLInputElement): void {
		node.focus();
		node.select();
	}

	const ARROW_KEYS: ArrowKey[] = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];

	function onKeydown(e: KeyboardEvent, item: TabItem): void {
		// A tab is a `div[role=tab]`, not a <button> — the optional per-tab ✕ nests inside it,
		// and an interactive descendant inside a real <button> is invalid ARIA — so Enter/Space
		// activation is authored here rather than inherited.
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			onSelect(item.id);
			return;
		}
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
		{#if previewIndex === i}
			<div class="ui-tab-preview" aria-hidden="true"></div>
		{/if}
		{#if editing === item.id}
			<!-- The rename editor replaces the tab for its duration: an input cannot live inside a
			     <button>, and the tab's whole face IS the field while renaming. -->
			<div class="ui-tab active editing">
				<!-- svelte-ignore a11y_autofocus -->
				<input
					{...MODE_ATTRS.search}
					class="ui-tab-rename"
					aria-label="Tab name"
					value={editValue}
					oninput={(e) => (editValue = e.currentTarget.value)}
					onblur={commitRename}
					onkeydown={(e) => {
						if (e.key === 'Enter') commitRename();
						else if (e.key === 'Escape') editing = null;
					}}
					use:focusInput
				/>
			</div>
		{:else}
			<div
				bind:this={tabEls[i]}
				role="tab"
				class="ui-tab"
				class:active={item.id === resolved}
				aria-selected={item.id === resolved}
				tabindex={item.id === resolved ? 0 : -1}
				onclick={() => onSelect(item.id)}
				ondblclick={() => startRename(item)}
				onkeydown={(e) => onKeydown(e, item)}
				{...tabProps?.(item)}
			>
				<span class="ui-tab-label">{item.label}</span>
				{#if onClose}
					<button
						type="button"
						class="ui-tab-close"
						tabindex="-1"
						aria-label="Close tab"
						title="Close tab"
						onclick={(e) => {
							e.stopPropagation();
							onClose(item.id);
						}}><Icon name="x" /></button
					>
				{/if}
			</div>
		{/if}
	{/each}
	{#if previewIndex === items.length}
		<div class="ui-tab-preview" aria-hidden="true"></div>
	{/if}
	{#if onAdd}
		<button type="button" class="ui-tab-add" aria-label="New tab" title="New tab" onclick={onAdd}
			><Icon name="plus" /></button
		>
	{/if}
</div>

<style>
	/* The strip sits at the header surface; the active tab drops out of it onto the body surface.
	   Two more per-instance hooks beside the surface pair: `--tabs-align` and `--tabs-pad`. The
	   default (bottom-hugged pills under a breathing-room inset) is the inspector's strip look;
	   the header's layout bar sets `stretch`/`0` so each pill spans the full strip and its LABEL
	   centres on the bar's midline — level with the ＋ and the rest of the header row — while the
	   pill still reaches the bottom edge it merges over (stretch touches both edges). */
	.ui-tabs {
		display: flex;
		align-items: var(--tabs-align, flex-end);
		gap: var(--space-1);
		min-width: 0;
		padding: var(--tabs-pad, var(--space-2) var(--space-2) 0);
		background: var(--tabs-surface, var(--surface-2));
		font-family: var(--font-mono);
	}
	.ui-tab {
		flex: 0 1 auto;
		min-width: 0;
		display: inline-flex;
		align-items: center;
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
	.ui-tab-label {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
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

	/* --- the opt-in affordances ------------------------------------------------------------- */

	/* The per-tab ✕: collapsed to zero width when hidden so an inactive tab stays evenly padded;
	   expands (with its left gap) on hover / when active. `overflow: hidden` is what clips it
	   away; the coarse door below rests it open instead (C17 — a hover reveal is unreachable on a
	   device with no hover). */
	.ui-tab-close {
		position: relative; /* anchor for the coarse-pointer hit-rect ::after */
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 0;
		min-width: 0;
		height: 16px;
		margin-left: 0;
		padding: 0;
		background: transparent;
		border: none;
		overflow: hidden;
		color: var(--text-muted);
		opacity: 0;
		cursor: pointer;
		transition:
			opacity var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-tab:hover .ui-tab-close,
	.ui-tab.active .ui-tab-close {
		width: 16px;
		margin-left: var(--space-2);
		opacity: 1;
	}
	.ui-tab-close :global(svg) {
		width: 12px;
		height: 12px;
		flex: 0 0 auto;
	}
	.ui-tab-close:hover {
		color: var(--danger);
	}

	/* The trailing ＋. Self-styled like the tabs beside it (this is a leaf primitive — it does not
	   compose IconButton, whose --hit floor would leave the ＋ towering over the strip). */
	.ui-tab-add {
		flex: 0 0 auto;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		align-self: center;
		width: 22px;
		height: 22px;
		padding: 0;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-tab-add :global(svg) {
		width: 14px;
		height: 14px;
	}
	.ui-tab-add:hover {
		background: var(--surface-3);
		color: var(--text);
	}
	.ui-tab-add:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -2px;
	}

	/* The drop-slot placeholder: a tab-sized slot at the drop index so the landing spot is obvious
	   and the ＋ shifts over to make room. */
	.ui-tab-preview {
		flex: 0 0 auto;
		align-self: center;
		width: 96px;
		height: 26px;
		border-radius: var(--radius-sm);
		border: 1px dashed var(--accent);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
	}

	/* The inline rename editor — the tab's face becomes the field. */
	.ui-tab.editing {
		cursor: text;
	}
	.ui-tab-rename {
		width: 9ch;
		padding: 1px var(--space-2);
		font: inherit;
		font-size: var(--fs-small);
	}

	/* Touch (C17). The ✕ is hover-revealed, sub-floor AND clipped, so on a device with no hover it
	   is not merely small — it is unreachable. Resting it open at its 16px paint and releasing the
	   clip is the whole fix; the tab pill itself already carries the --hit floor. The rename input
	   is raised to 16px so iOS does not force-zoom the page on focus. */
	@media (hover: none) and (pointer: coarse) {
		.ui-tab-close {
			width: 16px;
			margin-left: var(--space-2);
			overflow: visible;
			opacity: 1;
		}
		/* The ✕ cannot grow — it sits inside the pill — so IconButton's hit-rect idiom carries its
		   tap target out to --hit while the paint stays 16px. */
		.ui-tab-close::after {
			content: '';
			position: absolute;
			inset: calc((var(--hit) - 100%) / -2);
		}
		/* The ＋ CAN grow: the pills beside it already stand at --hit under coarse, so the dense
		   22px box is a fine-pointer affordance only and the floor comes back here — the same
		   restore `density="chrome"` gave its IconButton predecessor (touch-floor.spec.ts pins the
		   BOX, not the hit rect). */
		.ui-tab-add {
			width: var(--hit);
			height: var(--hit);
		}
		.ui-tab-rename {
			font-size: 16px;
		}
	}
</style>
