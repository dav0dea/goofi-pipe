<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { activeOrOnlyEditor } from '$lib/panels/editorCommands';
	import { untrack, type Snippet } from 'svelte';
	import type { MenuItem } from '$lib/workspace/menu';
	import ContextMenu from '$lib/workspace/ContextMenu.svelte';
	import PerfHud from './PerfHud.svelte';
	import { createWidthCache, planOverflow, type OverflowItem } from './overflowFit';
	import { Button, IconButton, Badge, Icon } from '$lib/ui';

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
	// The budget is MEASURED each replan: the header's own inner width, minus the status cluster's
	// laid-out rect, minus a reservation for the tab strip. The one width it must never read is
	// `.actions`' own — that shrinks the moment an item leaves, which is the oscillation bug.
	//
	// The cluster is not a fixed term (it was `flex: 0 0 auto` when this comment was first written;
	// `8596297` made it `flex: 0 1 auto` because a live patch's filename otherwise pushed the
	// overflow trigger off the right edge at 412px). That is exactly why `replan` re-reads its rect
	// and the observer below watches it. Deleting the brand made the cluster narrower, not fixed:
	// the filename inside it is still unbounded, and the connection Badge — absent while the socket
	// is healthy — comes back at its full width the moment it is not.
	// The plan still converges: `.tabslot` is the only growable box and `.action-zone` is rigid, so
	// a spilled action's width goes to the strip 1:1 — the fit condition reduces to a monotone width
	// threshold. Where the line is ALREADY overflowing, cluster and strip shrink together, so two
	// adjacent spill sets can both be self-consistent: a narrow band with hysteresis, settling on
	// whichever direction the width was approached from.
	// The cluster's yielding is measured, not claimed here: `touch-reflow.spec.ts`'s crowding-name
	// test fails at 412px with `flex: 0 0 auto` put back.

	/** Lowest priority first — the order the bar gives its actions up. D-R6: Undo · Redo · Save ·
	 * Load… · Save▾ is the keep order, so the caret goes first and the split degrades into a plain
	 * Save button. */
	const SPILL_ORDER = ['topbar-save-caret', 'topbar-load', 'topbar-save', 'topbar-redo', 'topbar-undo'];
	/** The tab strip is not an action and does not participate — but it is what the actions used
	 * to squeeze to zero width (o1), taking every layout tab and the ＋ with it. Two tap targets
	 * is the floor they spill against, in the same token that decides what "tappable" means, so it
	 * scales with the pointer rather than naming a phone.
	 *
	 * It is a term in this BUDGET, and deliberately not a `min-width` on `.tabslot`. The two are
	 * different questions: this one decides when an action gives up its slot, and the layout that
	 * follows then shares what is left between a shrinkable status cluster and a shrinkable strip,
	 * which on a 60-character filename lands the strip BELOW this reservation (`touch-reflow.spec.ts`
	 * asserts the property that actually matters instead: never below ONE tap target, with `.tabs`
	 * its own `overflow-x` scroller for the rest). Pinning the layout to this same number would take
	 * the difference straight back out of the filename, so the two mechanisms would be fighting over
	 * one budget. R's audit raised it; this is the verdict. */
	const TABSLOT_HITS = 2;

	let barEl = $state<HTMLDivElement | null>(null);
	let statusEl = $state<HTMLDivElement | null>(null);
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
		const status = statusEl;
		const host = actionsEl;
		const trigger = zoneEl?.querySelector<HTMLElement>('[data-testid="topbar-overflow"]');
		if (!bar || !status || !host || !trigger) return;
		const rem = px(document.documentElement, 'font-size');
		const widths = widthCache.widths(rem);
		if (widths.length === 0) return;
		const items: OverflowItem[] = SPILL_ORDER.map((id, i) => ({ id, width: widths[i] }));

		const barGap = px(bar, 'gap');
		const hit = px(document.documentElement, '--hit');
		// The header's three sections are status · tabs · action zone, so two gaps; the tab strip is
		// only rendered when a `tabs` snippet was given.
		const sections = tabs ? 2 : 1;
		const reserve = tabs ? TABSLOT_HITS * hit : 0;
		const budget =
			bar.clientWidth -
			px(bar, 'padding-left') -
			px(bar, 'padding-right') -
			status.getBoundingClientRect().width -
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
		const status = statusEl;
		if (!bar || !status || !zoneEl || !actionsEl) return;
		const ro = new ResizeObserver(replan);
		ro.observe(bar);
		// …and the status cluster, whose width moves on its own (the filename, the dirty dot, the
		// connection badge) and is a term in the budget.
		ro.observe(status);
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
	 * Addressed through the selection store's active editor: the editor the user last worked in,
	 * which is the same one the standalone Parameters/Metadata/Errors panels already follow — or,
	 * when that id has gone stale and only one editor is open, that one. Never a guess between
	 * several: a row that deletes disables rather than pick whichever is first in the map. */
	function canvasItems(): MenuItem[] {
		const ed = activeOrOnlyEditor(sel.activeEditorId);
		const has = ed?.hasSelection() ?? false;
		return [
			{ label: 'Select all', icon: 'square-dashed', disabled: !ed, action: () => ed?.selectAll() },
			// Multi-select's way out. With the mode on, a tap on empty canvas no longer clears (it
			// would wipe the selection the mode is for), and Escape is a keyboard's door only — so
			// this row is what makes the fold in `clickPane` safe to ship.
			{ label: 'Clear selection', disabled: !has, action: () => ed?.clearSelection() },
			{
				label: 'Multi-select mode',
				checked: sel.multiSelect,
				action: () => sel.toggleMultiSelect()
			},
			{ separator: true },
			{ label: 'Delete selection', icon: 'x', disabled: !has, action: () => ed?.deleteSelection() },
			{
				label: 'Group into sub-patch',
				icon: 'group',
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
				icon: 'undo-2',
				disabled: !h.canUndo,
				action: () => void h.undo()
			});
		if (isSpilled('topbar-redo'))
			items.push({ label: 'Redo', icon: 'redo-2', disabled: !h.canRedo, action: () => void h.redo() });
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

<div class="topbar" class:disconnected={g.disconnected} bind:this={barEl}>
	<div class="status" bind:this={statusEl}>
		<!-- The connection speaks ONLY when it needs attention. "Connected" was true in every
		     screenshot of a working app and spent 72px of a 412px bar saying so; the alarm state
		     (established, then lost — `graph.disconnected`, which is what keeps a boot quiet) takes
		     that width back at the moment it is worth something. It sits in `.status`, which D-R6
		     keeps OUT of the progressive overflow, so it can never spill into a menu — a warning
		     the user has to open a menu to find is not a warning. -->
		{#if g.disconnected}
			<Badge tone="warning" data-testid="topbar-connection">disconnected</Badge>
		{/if}
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
				onclick={() => void h.undo()}><Icon name="undo-2" /></IconButton
			>
			<IconButton
				variant="ghost"
				data-testid="topbar-redo"
				class={isSpilled('topbar-redo') ? 'spilled' : ''}
				disabled={!h.canRedo}
				title={h.redoLabel ? `Redo ${h.redoLabel}` : 'Nothing to redo'}
				label="Redo"
				onclick={() => void h.redo()}><Icon name="redo-2" /></IconButton
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
					onclick={openSaveMenu}><Icon name="chevron-down" /></IconButton
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
		     The accent and the title are multi-select mode's always-visible tell — the user asked for
		     a mode, and a mode you cannot see is a gesture with extra steps. The tell is all this
		     button carries of it: the STATE belongs to the row that toggles it (`aria-checked`),
		     while `aria-expanded` is the one thing this button does own and does change. -->
		<IconButton
			variant="ghost"
			data-testid="topbar-overflow"
			class={sel.multiSelect ? 'multi-on' : ''}
			aria-expanded={overflowMenu !== null}
			title={sel.multiSelect ? 'More actions — multi-select mode is on' : 'More actions'}
			label="More actions"
			onclick={openOverflow}><Icon name="ellipsis" /></IconButton
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
		/* The bar is its own query container, so the perf HUD below can stand down on WIDTH rather
		   than on device class — the same rule the progressive overflow follows (D-R6). Safe as
		   containment: nothing inside this bar is positioned out of it (both menus portal to
		   <body>), so the stacking context it establishes traps nothing. */
		container-type: inline-size;
		container-name: topbar;
	}
	/* The other half of the alarm, and the loud half: a chip in a 412px bar is easy to miss, so the
	   whole strip of constant chrome wears the fault ink.
	   An OUTLINE with a negative offset — app.css's sanctioned "frame the whole box" ring form. It
	   is painted last in its stacking context (above the bar's own children, no pseudo-element and
	   no z-index) and it is painted OUTSIDE the box model, so it costs no layout: the 44px height,
	   the workspace origin below it, and every rect `replan` measures are exactly what they were a
	   frame earlier. A border-width would move all four. */
	.topbar.disconnected {
		outline: 3px solid var(--warning);
		outline-offset: -3px;
	}
	/* Shrinkable, and it was not. The status cluster does not participate in the progressive
	   overflow (D-R6) — but `flex: 0 0 auto` on a cluster whose widest member is a filename meant
	   that on a live patch at 412px it alone claimed ~375 of 391px and pushed the overflow trigger,
	   the one control that must always be reachable, clean off the right edge. Deleting the brand
	   took the ⟁ and the wordmark off that figure and none of its reason: the filename is unbounded.
	   What is left here is the patch's own state — live rate, name, dirty dot, and the connection
	   only while it is lost. */
	.status {
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
	/* Below this the perf HUD stands down. On a LIVE patch it is the widest box in this cluster
	   (82px measured at 412px, against the connection Badge's 72 — which now only appears when that
	   connection is lost) and the only one of them that is a dev diagnostic rather than the patch's
	   own state — `styleDrift`'s own exemption calls it "sized to be unobtrusive". With it up, the
	   filename — the cluster's sole shrink absorber, because the HUD is `nowrap` with
	   `min-width: auto` and cannot give an inch — was ellipsised to five or six characters of a
	   sixty-character name. A width threshold, not a device class, and not a component swap, since
	   D-R6 keeps the status cluster out of the progressive overflow and this is the only lever the
	   narrow end has. */
	@container topbar (max-width: 520px) {
		.status :global(.hud) {
			display: none;
		}
	}
	/* …and the filename is where the cluster's shrink is absorbed: it is the longest and by far the
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
