<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { editorAt } from '$lib/panels/editorCommands';
	import { untrack, type Snippet } from 'svelte';
	import type { MenuItem } from '$lib/workspace/menu';
	import ContextMenu from '$lib/workspace/ContextMenu.svelte';
	import PerfHud from './PerfHud.svelte';
	import { createWidthCache, planOverflow, type OverflowItem } from './overflowFit';
	import { Button, IconButton, Badge } from '$lib/ui';

	// The header holds APP-GLOBAL actions only: session undo/redo and the patch's save/load.
	// Anything that acts on one panel belongs to that panel — a node editor already carries its
	// own fit control and its own add-node doors, and resolving "the active editor" up here only
	// hid which of several open editors an action would land in.
	type Props = {
		onSave: () => void;
		onSaveAs: () => void;
		onSaveInBrowser: () => void;
		onLoad: () => void;
		/** Workspace tab strip, rendered in the header's central gap between the
		 * filename and the action buttons. */
		tabs?: Snippet;
	};

	const { onSave, onSaveAs, onSaveInBrowser, onLoad, tabs }: Props = $props();

	const g = graph();
	const h = history();
	const sel = selection();

	// Save split-button dropdown — opened via the shared ContextMenu, which
	// portals to <body> at --z-menu so it stacks above side panels.
	let saveMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	function openSaveMenu(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		saveMenu = {
			x: Math.max(6, r.right - 180),
			y: r.bottom + 4,
			items: saveOptions()
		};
	}

	function saveOptions(): MenuItem[] {
		return [
			{ label: 'Save As…', action: onSaveAs },
			{ label: 'Save in browser', action: onSaveInBrowser }
		];
	}

	// --- progressive overflow (D-R6) -----------------------------------------
	//
	// The actions live in the bar and give themselves up to the overflow menu ONE AT A TIME,
	// lowest priority first, as the width runs out. `overflowFit.ts` owns the arithmetic (and its
	// three traps); this file owns only the measuring — which boxes to read, and against what.
	//
	// The budget is deliberately derived from boxes the plan cannot move: the header's own inner
	// width, the brand cluster (`flex: 0 0 auto`, so spilling an action never resizes it), and a
	// reservation for the tab strip. Reading `.actions`' own width instead would be the
	// oscillation bug: it shrinks the moment an item leaves.

	/** Lowest priority first — the order the bar gives its actions up. D-R6: Undo · Redo · Save ·
	 * Load… · Save▾ is the keep order, so the caret goes first and the split degrades into a plain
	 * Save button. */
	const SPILL_ORDER = ['topbar-save-caret', 'topbar-load', 'topbar-save', 'topbar-redo', 'topbar-undo'];
	/** The tab strip is not an action and does not participate — but it is what the actions used
	 * to squeeze to zero width (o1), taking every layout tab and the ＋ with it. Two tap targets
	 * is the floor they spill against, in the same token that decides what "tappable" means, so it
	 * scales with the pointer rather than naming a phone. */
	const TABSLOT_HITS = 2;

	let barEl = $state<HTMLDivElement | null>(null);
	let brandEl = $state<HTMLDivElement | null>(null);
	let zoneEl = $state<HTMLDivElement | null>(null);
	let actionsEl = $state<HTMLDivElement | null>(null);
	let spilled = $state<Set<string>>(new Set());
	let overflowMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	const isSpilled = (id: string): boolean => spilled.has(id);

	function px(el: Element, prop: string): number {
		return parseFloat(getComputedStyle(el).getPropertyValue(prop)) || 0;
	}

	/** Every action's intrinsic width, read with none of them hidden.
	 *
	 * The hide class is stripped and restored inside one synchronous block, so nothing is ever
	 * painted mid-measurement and Svelte's own class bookkeeping stays correct (it re-applies from
	 * `spilled` on the next update either way). Only runs when the root font size moved. */
	function measureWidths(): number[] {
		const host = actionsEl;
		if (!host) return [];
		const hidden = [...host.querySelectorAll<HTMLElement>('.spilled')];
		for (const el of hidden) el.classList.remove('spilled');
		const gap = px(host, 'gap');
		const widths = SPILL_ORDER.map((id) => {
			const el = host.querySelector<HTMLElement>(`[data-testid="${id}"]`);
			if (!el) return 0;
			const w = el.getBoundingClientRect().width;
			// The caret shares the split control's flex slot with Save and introduces no gap of its
			// own, so its cost nets out the one gap the plan charges every item.
			return id === 'topbar-save-caret' ? w - gap : w;
		});
		for (const el of hidden) el.classList.add('spilled');
		return widths;
	}

	const widthCache = createWidthCache(measureWidths);

	function replan(): void {
		const bar = barEl;
		const brand = brandEl;
		const host = actionsEl;
		const trigger = zoneEl?.querySelector<HTMLElement>('[data-testid="topbar-overflow"]');
		if (!bar || !brand || !host || !trigger) return;
		const rem = px(document.documentElement, 'font-size');
		const widths = widthCache.widths(rem);
		if (widths.length === 0) return;
		const items: OverflowItem[] = SPILL_ORDER.map((id, i) => ({ id, width: widths[i] }));

		const barGap = px(bar, 'gap');
		const hit = px(document.documentElement, '--hit');
		// The header's three sections are brand · tabs · action zone, so two gaps; the tab strip is
		// only rendered when a `tabs` snippet was given.
		const sections = tabs ? 2 : 1;
		const reserve = tabs ? TABSLOT_HITS * hit : 0;
		const budget =
			bar.clientWidth -
			px(bar, 'padding-left') -
			px(bar, 'padding-right') -
			brand.getBoundingClientRect().width -
			barGap * sections -
			reserve;

		const next = planOverflow(items, SPILL_ORDER, {
			gap: px(host, 'gap'),
			budget,
			trigger: trigger.getBoundingClientRect().width
		});
		// Write only on a real change: the observer re-fires on the layout this write causes, and
		// an unconditional assignment would keep the effect alive forever even though the plan has
		// converged.
		if (next.size !== spilled.size || [...next].some((id) => !spilled.has(id))) spilled = next;
	}

	$effect(() => {
		const bar = barEl;
		const brand = brandEl;
		if (!bar || !brand || !zoneEl || !actionsEl) return;
		const ro = new ResizeObserver(replan);
		ro.observe(bar);
		// …and the brand cluster, whose width moves on its own (the filename, the dirty dot, the
		// connection badge) and is a term in the budget.
		ro.observe(brand);
		// `untrack`: replan READS `spilled` to decide whether the plan changed, and writing it from
		// inside a tracked call would make this effect its own dependency — tearing down and
		// rebuilding the observer on every spill.
		untrack(replan);
		// The first measurement can land before the webfont does, and a text button is a different
		// number of pixels in the fallback face — a change no resize and no root-size step reports.
		let live = true;
		void document.fonts?.ready.then(() => {
			if (!live) return;
			widthCache.invalidate();
			replan();
		});
		return () => {
			live = false;
			ro.disconnect();
		};
	});

	// --- the overflow menu ---------------------------------------------------
	//
	// A ContextMenu, not a Popover: it is a MENU — checked rows, disabled rows, separators — and
	// this file already opens one for the Save caret. Popover is the bare anchored surface for
	// things that are not menus (D-M2's split), and building rows on top of it here would be a
	// second menu vocabulary in the same component.

	/** The canvas commands (D-R4). They are overflow-resident at EVERY width — they have no bar
	 * slot to lose, which is why one menu serves both jobs.
	 *
	 * Addressed through `editorAt` and the selection store's active editor: the editor the user
	 * last worked in, which is the same one the standalone Parameters/Metadata/Errors panels
	 * already follow. Strictly — with no editor open there is no unambiguous target, and a row
	 * that deletes must disable rather than guess which of several editors it meant. */
	function canvasItems(): MenuItem[] {
		const ed = editorAt(sel.activeEditorId);
		const has = ed?.hasSelection() ?? false;
		return [
			{ label: 'Select all', icon: '▦', disabled: !ed, action: () => ed?.selectAll() },
			{
				label: 'Multi-select mode',
				checked: sel.multiSelect,
				action: () => sel.toggleMultiSelect()
			},
			{ separator: true },
			{ label: 'Delete selection', icon: '✕', disabled: !has, action: () => ed?.deleteSelection() },
			{
				label: 'Group into sub-patch',
				icon: '▣',
				disabled: !has,
				action: () => ed?.groupSelection()
			},
			{ label: 'Copy', disabled: !has, action: () => ed?.copySelection() },
			{ label: 'Paste', disabled: !ed, action: () => ed?.pasteClipboard() },
			{ label: 'Duplicate', disabled: !has, action: () => ed?.duplicateSelection() }
		];
	}

	/** The bar's own actions, but only the ones that no longer fit. Same commands, second
	 * representation — never a parallel implementation (D-R2). */
	function spilledItems(): MenuItem[] {
		const items: MenuItem[] = [];
		if (isSpilled('topbar-undo'))
			items.push({
				label: 'Undo',
				icon: '↶',
				disabled: !h.canUndo,
				action: () => void h.undo()
			});
		if (isSpilled('topbar-redo'))
			items.push({ label: 'Redo', icon: '↷', disabled: !h.canRedo, action: () => void h.redo() });
		if (isSpilled('topbar-save')) items.push({ label: 'Save', action: onSave });
		if (isSpilled('topbar-save-caret')) items.push(...saveOptions());
		if (isSpilled('topbar-load')) items.push({ label: 'Load…', action: onLoad });
		return items;
	}

	function openOverflow(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const above = spilledItems();
		overflowMenu = {
			x: Math.max(6, r.right - 180),
			y: r.bottom + 4,
			items: above.length ? [...above, { separator: true }, ...canvasItems()] : canvasItems()
		};
	}
</script>

<div class="topbar" bind:this={barEl}>
	<div class="brand" bind:this={brandEl}>
		<span class="logo">⟁</span>
		<span class="name">goofi-pipe</span>
		<Badge tone={g.connected ? 'success' : 'warning'}>
			{g.connected ? 'connected' : 'connecting…'}
		</Badge>
		<PerfHud />
		{#if g.savePath}
			<span class="path" title={g.savePath}
				>{g.unsavedChanges ? '● ' : ''}{g.savePath.split('/').pop()}</span
			>
		{:else if g.unsavedChanges}
			<span class="path">● untitled</span>
		{/if}
	</div>

	{#if tabs}
		<div class="tabslot">{@render tabs()}</div>
	{/if}

	<div class="action-zone" bind:this={zoneEl}>
		<div class="actions" bind:this={actionsEl}>
			<IconButton
				variant="ghost"
				data-testid="topbar-undo"
				class={isSpilled('topbar-undo') ? 'spilled' : ''}
				disabled={!h.canUndo}
				title={h.undoLabel ? `Undo ${h.undoLabel}` : 'Nothing to undo'}
				label="Undo"
				onclick={() => void h.undo()}>↶</IconButton
			>
			<IconButton
				variant="ghost"
				data-testid="topbar-redo"
				class={isSpilled('topbar-redo') ? 'spilled' : ''}
				disabled={!h.canRedo}
				title={h.redoLabel ? `Redo ${h.redoLabel}` : 'Nothing to redo'}
				label="Redo"
				onclick={() => void h.redo()}>↷</IconButton
			>
			<!-- The split control degrades rather than disappearing: the caret is the first thing to
			     spill, and `.no-caret` restores Save's own right-hand corners so the seam does not
			     hang off a button with nothing beside it. -->
			<div class="split" class:spilled={isSpilled('topbar-save')} class:no-caret={isSpilled('topbar-save-caret')}>
				<Button variant="ghost" class="seg-main" data-testid="topbar-save" onclick={onSave}
					>Save</Button
				>
				<IconButton
					variant="ghost"
					class={`seg-caret ${isSpilled('topbar-save-caret') ? 'spilled' : ''}`}
					data-testid="topbar-save-caret"
					label="Save options"
					onclick={openSaveMenu}>▾</IconButton
				>
			</div>
			<Button
				variant="ghost"
				data-testid="topbar-load"
				class={isSpilled('topbar-load') ? 'spilled' : ''}
				onclick={onLoad}>Load…</Button
			>
		</div>
		<!-- Resident at every width: it carries the canvas commands, which have no bar slot to lose.
		     `aria-pressed` is multi-select mode's always-visible tell — the user asked for a mode,
		     and a mode you cannot see is a gesture with extra steps. -->
		<IconButton
			variant="ghost"
			data-testid="topbar-overflow"
			class={sel.multiSelect ? 'multi-on' : ''}
			aria-pressed={sel.multiSelect}
			title={sel.multiSelect ? 'More actions — multi-select mode is on' : 'More actions'}
			label="More actions"
			onclick={openOverflow}>⋯</IconButton
		>
	</div>
</div>

{#if saveMenu}
	<ContextMenu
		x={saveMenu.x}
		y={saveMenu.y}
		items={saveMenu.items}
		onClose={() => (saveMenu = null)}
	/>
{/if}

{#if overflowMenu}
	<ContextMenu
		x={overflowMenu.x}
		y={overflowMenu.y}
		items={overflowMenu.items}
		onClose={() => (overflowMenu = null)}
	/>
{/if}

<style>
	.topbar {
		display: flex;
		align-items: center;
		gap: var(--space-7);
		padding: 0 var(--space-6);
		/* A surface step above the `--bg` workspace ground below it — that separates, so the hairline
		   this used to draw is deleted (D5). */
		background: var(--surface-1);
		/* Frozen: 44px is the coarse --hit floor, so the bar already fits a touch-sized
		   control without growing — a rem height would break that flush relationship. */
		height: 44px;
		font-size: var(--fs-body);
		z-index: 10;
		/* The bar is its own query container, so the wordmark below can stand down on WIDTH rather
		   than on device class — the same rule the progressive overflow follows (D-R6). Safe as
		   containment: nothing inside this bar is positioned out of it (both menus portal to
		   <body>), so the stacking context it establishes traps nothing. */
		container-type: inline-size;
		container-name: topbar;
	}
	/* Shrinkable, and it was not. The status cluster does not participate in the progressive
	   overflow (D-R6) — but `flex: 0 0 auto` on a cluster whose widest member is a filename meant
	   that on a live patch at 412px the brand alone claimed ~375 of 391px and pushed the overflow
	   trigger, the one control that must always be reachable, clean off the right edge. */
	.brand {
		display: flex;
		align-items: center;
		gap: var(--space-6);
		flex: 0 1 auto;
		min-width: 0;
	}
	/* The tab strip fills the slack between the filename and the actions. */
	.tabslot {
		flex: 1 1 auto;
		min-width: 0;
		align-self: stretch;
		display: flex;
		align-items: stretch;
	}
	.logo {
		font-size: var(--fs-title);
		color: var(--text);
		font-weight: 600;
	}
	/* `nowrap` is not cosmetic: the brand shrinks now, and a wrapped wordmark is two lines inside a
	   44px bar. */
	.name {
		font-weight: 600;
		white-space: nowrap;
		flex: 0 0 auto;
	}
	/* Below this the wordmark stands down and the ⟁ carries the identity alone — it is the one
	   thing in the cluster that says nothing the rest does not, and the ~95px it costs is what the
	   layout tab strip needs to keep more than its ＋. A width threshold, not a rung. */
	@container topbar (max-width: 520px) {
		.name {
			display: none;
		}
	}
	/* …and the filename is where the brand's shrink is absorbed: it is the longest and by far the
	   most variable of the cluster, and the only one an ellipsis still leaves readable. */
	.path {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		flex: 0 1 auto;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The actions and the overflow trigger they spill into. Two boxes, because `.actions` is pinned
	   by `topbar.spec.ts` as EXACTLY the five app-global actions — the trigger is chrome for the
	   menu, not a sixth action. */
	.action-zone {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		flex: 0 0 auto;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
	/* A spilled action is in the menu instead; it stays in the DOM so its intrinsic width can be
	   re-read when the responsive root size moves it. `:global`, because the class rides a
	   primitive's `class` prop onto its inner <button> — and that same global part is what also
	   catches `.split.spilled`, the wrapper that goes when both its segments have. */
	.actions :global(.spilled) {
		display: none;
	}
	.action-zone :global(.multi-on) {
		color: var(--accent);
		border-color: var(--accent);
	}
	/* Save split — two adjacent Buttons sharing a seam: the touching corners are
	   squared so the main action + its caret read as one segmented control. */
	.split {
		position: relative;
		display: inline-flex;
		align-items: stretch;
	}
	.split :global(.seg-main) {
		border-top-right-radius: 0;
		border-bottom-right-radius: 0;
	}
	.split :global(.seg-caret) {
		border-top-left-radius: 0;
		border-bottom-left-radius: 0;
	}
	.split.no-caret :global(.seg-main) {
		border-top-right-radius: var(--radius-sm);
		border-bottom-right-radius: var(--radius-sm);
	}
</style>
