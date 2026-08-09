<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { perfStats } from '$lib/api/perfStats.svelte';
	import { activeOrOnlyEditor } from '$lib/panels/editorCommands';
	import { tick, untrack, type Snippet } from 'svelte';
	import type { MenuItem } from '$lib/workspace/menu';
	import ContextMenu from '$lib/workspace/ContextMenu.svelte';
	import PerfHud from './PerfHud.svelte';
	import { createWidthCache, planOverflow, type OverflowItem } from './overflowFit';
	import { IconButton, Badge, Icon } from '$lib/ui';

	// The header holds APP-GLOBAL actions only: session undo/redo and the patch's save/load.
	// Anything that acts on one panel belongs to that panel — a node editor already carries its
	// own fit control and its own add-node doors, and resolving "the active editor" up here only
	// hid which of several open editors an action would land in.
	type Props = {
		onSave: () => void;
		onSaveAs: () => void;
		onLoad: () => void;
		/** Workspace tab strip, rendered in the header's central gap between the
		 * filename and the action buttons. */
		tabs?: Snippet;
	};

	const { onSave, onSaveAs, onLoad, tabs }: Props = $props();

	const g = graph();
	const h = history();
	const sel = selection();
	const ws = workspace();
	const p = perfStats();

	// Mirror of PerfHud's own `{#if active}` gate, so the plan and the menu agree with the HUD
	// about whether there is anything to show. A boolean derived, NOT `p.fps` read raw in an
	// effect — fps ticks at 4Hz and would re-fire anything tracking it on every tick.
	const hudActive = $derived(p.fps > 0.05 || p.dps > 0.05);

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

	// One row. It stays a menu (not a bare second button) so the split control keeps its shape and
	// its spill behaviour — and so a future save option has a home that is not a fourth button.
	function saveOptions(): MenuItem[] {
		return [{ label: 'Save As…', testid: 'topbar-save-as', action: onSaveAs }];
	}

	// --- progressive overflow (D-R6) -----------------------------------------
	//
	// The bar's residents give themselves up to the overflow menu ONE AT A TIME, lowest priority
	// first, as the width runs out. `overflowFit.ts` owns the arithmetic (and its three traps);
	// this file owns only the measuring — which boxes to read, and against what.
	//
	// The LAYOUT TAB STRIP is the header's first-class citizen (Phil, 2026-08-08): it owns the
	// left edge, and the budget reserves its CONTENT width — so the identity chips and then the
	// actions yield in the strip's favour, and the strip's own `overflow-x` scroller is the
	// fallback only once even an otherwise-empty bar could not hold every tab.
	//
	// Everything else sits right, in ONE spillable group: the patch identity (perf HUD, filename +
	// dirty dot) yields FIRST — informational, so it outranks nothing — then the actions in the
	// order below. The connection chip is deliberately NOT in the plan: a warning that overflows
	// into a hidden menu is not a warning, so its width is subtracted from the budget instead.
	//
	// The budget is MEASURED each replan: the header's own inner width, minus the alarm chip's
	// rect, minus the tab reservation. The one width it must never read is the zone's own — that
	// shrinks the moment an item leaves, which is the oscillation bug. The plan converges because
	// `.tabslot` is the only growable box: a spilled item's width goes to the strip 1:1, so the
	// fit condition reduces to a monotone width threshold.

	/** Lowest priority first — the order the bar gives its residents up. The identity chips go
	 * before any action; then D-R6's keep order (Undo · Redo · Save · Load… · Save▾) unchanged,
	 * so the caret goes first among actions and the split degrades into a plain Save button. */
	const SPILL_ORDER = [
		'topbar-hud',
		'topbar-path',
		'topbar-save-caret',
		'topbar-load',
		'topbar-save',
		'topbar-redo',
		'topbar-undo'
	];
	/** The FLOOR under the tab reservation, in tap targets. The reserve normally follows the
	 * strip's measured content (tabs get their room before anything else keeps a slot), but a
	 * strip with one tab must still never be squeezed below what a finger can hit (o1). */
	const TABSLOT_HITS = 2;

	let barEl = $state<HTMLDivElement | null>(null);
	let tabslotEl = $state<HTMLDivElement | null>(null);
	let zoneEl = $state<HTMLDivElement | null>(null);
	let actionsEl = $state<HTMLDivElement | null>(null);
	let spilled = $state<Set<string>>(new Set());
	let overflowMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	const isSpilled = (id: string): boolean => spilled.has(id);

	function px(el: Element, prop: string): number {
		return parseFloat(getComputedStyle(el).getPropertyValue(prop)) || 0;
	}

	/** Every resident's intrinsic width, read with none of them hidden.
	 *
	 * The hide class is stripped and restored inside one synchronous block, so nothing is ever
	 * painted mid-measurement and Svelte's own class bookkeeping stays correct (it re-applies from
	 * `spilled` on the next update either way). Scoped to the ZONE: the identity chips are
	 * residents of the same plan as the actions, just lower priority. Re-runs when the root font
	 * size moved — and when the chips' CONTENT moved (the filename, the HUD's presence), which the
	 * invalidation effect below owns. */
	function measureWidths(): number[] {
		const host = zoneEl;
		if (!host) return [];
		const hidden = [...host.querySelectorAll<HTMLElement>('.spilled')];
		for (const el of hidden) el.classList.remove('spilled');
		const gap = px(host, 'gap');
		const widths = SPILL_ORDER.map((id) => {
			const el = host.querySelector<HTMLElement>(`[data-testid="${id}"]`);
			if (!el) return 0;
			const w = el.getBoundingClientRect().width;
			// The caret shares the split control's flex slot with Save and introduces no gap of its
			// own, so its cost nets out the one gap the plan charges every item. (`.actions` and
			// the zone share one gap token, so a single number serves both nesting levels.)
			return id === 'topbar-save-caret' ? w - gap : w;
		});
		for (const el of hidden) el.classList.add('spilled');
		return widths;
	}

	const widthCache = createWidthCache(measureWidths);

	function replan(): void {
		const bar = barEl;
		const zone = zoneEl;
		const trigger = zone?.querySelector<HTMLElement>('[data-testid="topbar-overflow"]');
		if (!bar || !zone || !actionsEl || !trigger) return;
		const rem = px(document.documentElement, 'font-size');
		const widths = widthCache.widths(rem);
		if (widths.length === 0) return;
		const items: OverflowItem[] = SPILL_ORDER.map((id, i) => ({ id, width: widths[i] }));

		const barGap = px(bar, 'gap');
		const zoneGap = px(zone, 'gap');
		const hit = px(document.documentElement, '--hit');
		// The header's two sections are tabs · zone, so one gap when the strip is rendered.
		const sections = tabs ? 1 : 0;
		// Tabs get their room FIRST: the reserve follows the strip's measured CONTENT, floored at
		// two tap targets (o1). Content is summed from the scroller's children — NOT read off
		// `scrollWidth`, because the strip is `flex: 1 1 auto`: a non-overflowing scroller's
		// scrollWidth is its laid-out width, i.e. whatever slack the LAST plan left it, which is
		// the self-read that put hysteresis into the plan (caught by the one-width-one-answer
		// walk). The pills are `nowrap` with min-content floors, so their rects are the same
		// number from either approach direction.
		const strip = tabslotEl?.querySelector<HTMLElement>('[data-testid="workspace-tabs"]');
		let tabsContent = 0;
		if (strip) {
			const kids = [...strip.children] as HTMLElement[];
			tabsContent =
				kids.reduce((a, el) => a + el.getBoundingClientRect().width, 0) +
				px(strip, 'gap') * Math.max(0, kids.length - 1) +
				px(strip, 'padding-left') +
				px(strip, 'padding-right');
		}
		const reserve = tabs ? Math.max(TABSLOT_HITS * hit, tabsContent) : 0;
		// The alarm chip is not in the plan (a warning must never land in a menu), so its width —
		// plus the zone gap it introduces — comes off the budget whenever it is up.
		const alarm = zone.querySelector<HTMLElement>('[data-testid="topbar-connection"]');
		const alarmW = alarm ? alarm.getBoundingClientRect().width + zoneGap : 0;
		const budget =
			bar.clientWidth -
			px(bar, 'padding-left') -
			px(bar, 'padding-right') -
			alarmW -
			barGap * sections -
			reserve;

		const next = planOverflow(items, SPILL_ORDER, {
			gap: zoneGap,
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
		if (!bar || !zoneEl || !actionsEl) return;
		const ro = new ResizeObserver(replan);
		ro.observe(bar);
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

	// The identity chips change width from CONTENT, which no bar resize reports: a save renames
	// the patch, the dirty dot comes and goes, the HUD mounts with the first flowing frame. Their
	// cached intrinsic widths go stale at exactly those moments — and the tab strip's content
	// (the reserve) moves when a layout tab is added, closed or renamed. One effect owns all of
	// it: track the inputs, drop the cache, replan AFTER the DOM settles. The `tick()` is
	// load-bearing for the strip: WorkspaceTabs is a sibling tree (a snippet from AppShell), and
	// this effect can run before its new pills exist — a synchronous replan then measures the OLD
	// content and nothing re-fires it, leaving a stale plan (caught by the tabs-get-the-room e2e).
	$effect(() => {
		void g.savePath;
		void g.unsavedChanges;
		void hudActive;
		void ws.state.workspaces;
		widthCache.invalidate();
		void tick().then(replan);
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

	/** The bar's own residents, but only the ones that no longer fit. Same content, second
	 * representation — never a parallel implementation (D-R2). The identity chips become
	 * DISABLED rows: information relocates, it does not become clickable. */
	function spilledItems(): MenuItem[] {
		const items: MenuItem[] = [];
		if (isSpilled('topbar-path') && (g.savePath || g.unsavedChanges)) {
			// The same 32ch the chip caps at, applied to the DATA: a menu row does not ellipsize,
			// and an uncapped 60-character name pushes the whole menu off a 412px screen.
			const name = g.savePath?.split('/').pop() ?? 'untitled';
			items.push({
				label: `${g.unsavedChanges ? '● ' : ''}${name.length > 32 ? `${name.slice(0, 31)}…` : name}`,
				disabled: true,
				action: () => {}
			});
		}
		if (isSpilled('topbar-hud') && hudActive)
			items.push({ label: `${p.fps.toFixed(0)} fps`, disabled: true, action: () => {} });
		if (isSpilled('topbar-undo'))
			items.push({
				label: 'Undo',
				icon: 'undo-2',
				disabled: !h.canUndo,
				action: () => void h.undo()
			});
		if (isSpilled('topbar-redo'))
			items.push({ label: 'Redo', icon: 'redo-2', disabled: !h.canRedo, action: () => void h.redo() });
		if (isSpilled('topbar-save')) items.push({ label: 'Save', icon: 'save', action: onSave });
		if (isSpilled('topbar-save-caret')) items.push(...saveOptions());
		if (isSpilled('topbar-load'))
			items.push({ label: 'Load…', icon: 'folder-open', action: onLoad });
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
	{#if tabs}
		<!-- First-class and first: the layout tab strip owns the left edge and the slack, and the
		     budget reserves its content width — everything to the right yields in its favour. -->
		<div class="tabslot" bind:this={tabslotEl}>{@render tabs()}</div>
	{/if}

	<div class="action-zone" bind:this={zoneEl}>
		<!-- The connection speaks ONLY when it needs attention. "Connected" was true in every
		     screenshot of a working app and spent 72px of a 412px bar saying so; the alarm state
		     (established, then lost — `graph.disconnected`, which is what keeps a boot quiet) takes
		     that width back at the moment it is worth something. Deliberately OUTSIDE the
		     progressive overflow — its width is subtracted from the budget instead — so it can
		     never spill into a menu: a warning the user has to open a menu to find is not a
		     warning. -->
		{#if g.disconnected}
			<Badge tone="warning" data-testid="topbar-connection">disconnected</Badge>
		{/if}
		<!-- The patch identity: informational, so it is the FIRST thing the bar gives up. Each chip
		     is a plan resident with a testid the measurement reads; spilled, it becomes a disabled
		     row in the ⋯ menu. -->
		<span
			class="info"
			class:spilled={isSpilled('topbar-hud')}
			data-testid="topbar-hud"><PerfHud /></span
		>
		{#if g.savePath}
			<span
				class="info path"
				class:spilled={isSpilled('topbar-path')}
				data-testid="topbar-path"
				title={g.savePath}>{g.unsavedChanges ? '● ' : ''}{g.savePath.split('/').pop()}</span
			>
		{:else if g.unsavedChanges}
			<span class="info path" class:spilled={isSpilled('topbar-path')} data-testid="topbar-path"
				>● untitled</span
			>
		{/if}
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
				<IconButton
					variant="ghost"
					class="seg-main"
					data-testid="topbar-save"
					label="Save"
					onclick={onSave}><Icon name="save" /></IconButton
				>
				<IconButton
					variant="ghost"
					class={`seg-caret ${isSpilled('topbar-save-caret') ? 'spilled' : ''}`}
					data-testid="topbar-save-caret"
					label="Save options"
					onclick={openSaveMenu}><Icon name="chevron-down" /></IconButton
				>
			</div>
			<IconButton
				variant="ghost"
				data-testid="topbar-load"
				class={isSpilled('topbar-load') ? 'spilled' : ''}
				label="Load…"
				onclick={onLoad}><Icon name="folder-open" /></IconButton
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
	}
	/* The tab strip fills the slack between the bar's edges and the right-hand group — and the
	   overflow budget reserves its CONTENT width, so the slack is genuinely its before anything
	   else keeps a slot. */
	.tabslot {
		flex: 1 1 auto;
		min-width: 0;
		align-self: stretch;
		display: flex;
		align-items: stretch;
	}
	/* An identity chip. In the plan, so its hidden state is the same `.spilled` the actions use;
	   never a shrink absorber — a chip either fits whole or moves to the menu whole, which is what
	   keeps the plan's measured widths honest. */
	.info {
		display: inline-flex;
		align-items: center;
		min-width: 0;
	}
	.info.spilled {
		display: none;
	}
	/* The filename. Ellipsis against its own CAP, not against the bar's pressure (the plan spills
	   it long before the bar squeezes): 32ch keeps a long name readable while bounding what a
	   60-character patch can claim of the zone. */
	.path {
		color: var(--text-dim);
		/* The bar's shared chrome size (see app.css) — every word in the header on one baseline. */
		font-size: var(--fs-chrome);
		max-width: 32ch;
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
