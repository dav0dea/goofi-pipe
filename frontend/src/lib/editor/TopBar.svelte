<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from 'tatami';
	import { harnesses, harnessLabel } from '$lib/stores/harness.svelte';
	import { perfStats } from '$lib/api/perfStats.svelte';
	import { activeOrOnlyEditor } from '$lib/panels/editorCommands';
	import { tick, untrack, type Snippet } from 'svelte';
	import type { MenuItem } from 'tatami';
	import { ContextMenu } from 'tatami';
	import PerfHud from './PerfHud.svelte';
	import { createWidthCache, planOverflow, type OverflowItem } from 'tatami';
	import { IconButton, Badge, Button, Icon } from '$lib/ui';

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
	const hs = harnesses();

	// Mirror of PerfHud's own `{#if active}` gate, so the plan and the menu agree with the HUD
	// about whether there is anything to show. A boolean derived, NOT `p.fps` read raw in an
	// effect — fps ticks at 4Hz and would re-fire anything tracking it on every tick.
	const hudActive = $derived(p.fps > 0.05);

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
		return [{ label: 'Save As…', action: onSaveAs }];
	}

	// --- the agent chip ------------------------------------------------------
	//
	// The one door onto a running harness from outside its panel. It counts, and it ACTS: each row
	// is an instance, and choosing one asks whether to detach or kill (the shell's dialog, which
	// needs no panel of its own). There is deliberately no LAUNCH here: launching is the agent
	// panel's empty state, so the header never mints something it cannot then show.
	let agentMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	/** Raise the detach-or-kill question, and point the user at the terminal it is about when there
	 * is one on screen: the panel showing that instance, else any agent panel. The question itself
	 * needs no panel (the dialog is the shell's), which is what keeps this path free of a layout
	 * write — it used to open a TAB to hold the dialog, dirtying the patch and pushing two undo
	 * steps for what is only a question. Showing and focusing are viewpoint; asking is not even
	 * that. */
	function askClose(id: string): void {
		hs.requestClose(id);
		const panel = hs.panelShowing(id) ?? hs.firstPanel;
		if (!panel) return;
		hs.show(panel, id);
		ws.exitMaximize();
		ws.setActive(panel);
	}

	function openAgents(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		agentMenu = {
			x: Math.max(6, r.right - 180),
			y: r.bottom + 4,
			items: hs.instances.map((i) => ({
				label: `${harnessLabel(i)}${i.state === 'running' ? '' : ` (${i.state})`}`,
				icon: 'x',
				action: () => askClose(i.id)
			}))
		};
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
	 * `spilled` on the next update either way). Scoped to the one ITEMS group shared by identity and
	 * actions. Re-runs when the root font size moved — and when the chips' CONTENT moved (the
	 * filename, the HUD's presence), which the invalidation effect below owns. */
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
		const zone = zoneEl;
		const trigger = zone?.querySelector<HTMLElement>('[data-testid="topbar-overflow"]');
		if (!bar || !zone || !actionsEl || !trigger) return;
		const rem = px(document.documentElement, 'font-size');
		const widths = widthCache.widths(rem);
		if (widths.length === 0) return;
		const items: OverflowItem[] = SPILL_ORDER.map((id, i) => ({ id, width: widths[i] }));

		const barGap = px(bar, 'gap');
		const zoneGap = px(zone, 'gap');
		const itemGap = px(actionsEl, 'gap');
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
		// Neither live chip is in the plan, and for the same reason: a warning that spills into a
		// hidden menu is not a warning, and the agent chip is the only door onto detach/kill from
		// outside the panel. Their widths — plus the zone gap each introduces — come off the budget
		// instead. Both are conditional, so each is MEASURED when it is up and costs nothing when
		// it is not.
		const unplanned = (id: string): number => {
			const el = zone.querySelector<HTMLElement>(`[data-testid="${id}"]`);
			return el ? el.getBoundingClientRect().width + zoneGap : 0;
		};
		const alarmW = unplanned('topbar-connection') + unplanned('topbar-agents');
		const budget =
			bar.clientWidth -
			px(bar, 'padding-left') -
			px(bar, 'padding-right') -
			alarmW -
			barGap * sections -
			reserve;

		const next = planOverflow(items, SPILL_ORDER, {
			gap: itemGap,
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
		void hs.running;
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
		if (isSpilled('topbar-hud') && hudActive)
			items.push({ label: `${p.fps.toFixed(0)} fps`, disabled: true, action: () => {} });
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
		<!-- Present only while an agent is. Pressable, not a Badge: it is the door onto detach and
		     kill, and on a coarse pointer it is the only one that is not inside the panel it talks
		     about. Out of the spill plan for the same reason the alarm is — its width comes off the
		     budget above instead. -->
		{#if hs.running > 0}
			<!-- Ghost, not a Chip: every other thing in this bar is ink without a fill, and a pill
			     among them read as an alert rather than a count. The accent survives as the ink —
			     that is what carries "an agent is running" — while the box stays the --hit target a
			     finger needs, since dropping the fill must not shrink what it aims at. -->
			<Button
				variant="ghost"
				size="sm"
				style="--tatami-btn-ink: var(--accent)"
				data-testid="topbar-agents"
				aria-expanded={agentMenu !== null}
				title="Running agents"
				onclick={openAgents}><Icon name="bot" />{hs.running}</Button
			>
		{/if}
		<div class="actions" bind:this={actionsEl}>
			<!-- Identity and actions are ONE overflow group with ONE gap. The text items add the same
			     clear space around their ink that IconButton's square adds around each glyph; they still
			     spill first because information yields before actions. -->
			<span
				class="info hud-info"
				class:active={hudActive}
				class:spilled={isSpilled('topbar-hud')}
				data-testid="topbar-hud"><PerfHud /></span
			>
			{#if g.savePath}
				<span
					class="info path"
					class:spilled={isSpilled('topbar-path')}
					data-testid="topbar-path"
					title={g.savePath}
				>
					<span class="path-value"
						>{g.unsavedChanges ? '● ' : ''}{g.savePath.split('/').pop()}</span
					>
				</span>
			{:else if g.unsavedChanges}
				<span
					class="info path"
					class:spilled={isSpilled('topbar-path')}
					data-testid="topbar-path"
				>
					<span class="path-value">● untitled</span>
				</span>
			{/if}
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

{#if agentMenu}
	<ContextMenu
		x={agentMenu.x}
		y={agentMenu.y}
		items={agentMenu.items}
		onClose={() => (agentMenu = null)}
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
		/* No fixed height: the resident --hit-sized controls make this 28px on a fine pointer and
		   grow it to the 44px touch floor under the coarse-pointer density rule. */
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
	/* An identity item in the same group and on the same rhythm as every action. IconButton centres
	   a --fs-body glyph in a --hit square; the matching inline inset gives bare text the same clear
	   space around its ink. Never a shrink absorber — an item either fits whole or moves to the menu
	   whole, which is what keeps the plan's measured widths honest. */
	.info {
		display: inline-flex;
		align-items: center;
		min-height: var(--hit);
		min-width: 0;
		padding-inline: var(--topbar-info-inset);
		font-size: var(--fs-chrome);
		flex: 0 0 auto;
	}
	.info.spilled {
		display: none;
	}
	/* PerfHud owns the timer that makes `hudActive` live, so its host stays mounted while idle.
	   Remove that empty host from layout until the HUD has ink; otherwise a zero-width flex item
	   still introduces a gap before the patch name. */
	.hud-info:not(.active) {
		display: none;
	}
	/* The filename. Ellipsis against its own CAP, not against the bar's pressure (the plan spills
	   it long before the bar squeezes): 32ch keeps a long name readable while bounding what a
	   60-character patch can claim of the zone. */
	.path {
		color: var(--text-dim);
	}
	.path-value {
		display: block;
		max-width: 32ch;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The spillable items and their resident overflow trigger are two boxes: `.actions` owns every
	   item that can relocate (identity + the five app-global actions), while the trigger is chrome
	   for the menu, not a sixth spillable action. One token keeps the inter-item and trigger gaps on
	   the same rhythm. */
	.action-zone {
		--topbar-item-gap: var(--space-2);
		--topbar-info-inset: calc((var(--hit) - var(--fs-body)) / 2);
		display: flex;
		align-items: center;
		gap: var(--topbar-item-gap);
		flex: 0 0 auto;
	}
	.actions {
		display: flex;
		align-items: center;
		gap: var(--topbar-item-gap);
	}
	/* A spilled item is in the menu instead; it stays in the DOM so its intrinsic width can be
	   re-read when the responsive root size moves it. `:global`, because action classes ride a
	   primitive's `class` prop onto its inner <button> — and that same global part also catches
	   `.split.spilled`, the wrapper that goes when both its segments have. */
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
