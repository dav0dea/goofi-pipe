<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from 'panelty';
	import { harnesses, harnessLabel } from '$lib/stores/harness.svelte';
	import { perfStats } from '$lib/api/perfStats.svelte';
	import { activeOrOnlyEditor } from '$lib/panels/editorCommands';
	import { tick, untrack, type Snippet } from 'svelte';
	import type { MenuItem } from 'panelty';
	import { ContextMenu } from 'panelty';
	import PerfHud from './PerfHud.svelte';
	import { createWidthCache, planOverflow, type OverflowItem } from 'panelty';
	import { IconButton, Badge, Button, Icon } from '$lib/ui';

	// The header holds APP-GLOBAL actions only; anything acting on one panel belongs to that panel.
	type Props = {
		onSave: () => void;
		onSaveAs: () => void;
		onLoad: () => void;
		/** Workspace tab strip, rendered in the header's central gap. */
		tabs?: Snippet;
	};

	const { onSave, onSaveAs, onLoad, tabs }: Props = $props();

	const g = graph();
	const h = history();
	const sel = selection();
	const ws = workspace();
	const p = perfStats();
	const hs = harnesses();

	// A boolean, never raw `p.fps`: fps ticks at 4Hz and would re-fire everything tracking it.
	const hudActive = $derived(p.fps > 0.05);

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
		return [{ label: 'Save As…', action: onSaveAs }];
	}

	let agentMenu = $state<{ x: number; y: number; items: MenuItem[] } | null>(null);

	/** Raise the detach-or-kill question, and show the terminal it is about. Writes no layout: a
	 * question must not dirty the patch. */
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

	// Progressive overflow: `planOverflow` owns the arithmetic, this file owns only the measuring.
	// The budget must never read the action zone's own width — it shrinks as items leave, which
	// oscillates; `.tabslot` is the only growable box, which is what makes the plan converge.

	/** Lowest priority first: the order the bar gives its residents up. */
	const SPILL_ORDER = [
		'topbar-hud',
		'topbar-path',
		'topbar-save-caret',
		'topbar-load',
		'topbar-save',
		'topbar-redo',
		'topbar-undo'
	];
	/** The floor under the tab reservation, in tap targets, so a one-tab strip stays hittable. */
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

	/** Every resident's intrinsic width, read with none of them hidden. The hide class is stripped
	 * and restored in ONE synchronous block, so nothing is painted mid-measurement. */
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
			// The caret introduces no gap of its own, so it nets out the gap the plan charges.
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
		// Summed from the children, NEVER off `scrollWidth`: the strip is `flex: 1 1 auto`, so its
		// scrollWidth is the slack the LAST plan left it — a self-read, and hysteresis.
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
		// Neither live chip may spill into a hidden menu, so its width comes off the budget instead.
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
		// Write only on a real change: the observer re-fires on the layout this write causes.
		if (next.size !== spilled.size || [...next].some((id) => !spilled.has(id))) spilled = next;
	}

	$effect(() => {
		const bar = barEl;
		if (!bar || !zoneEl || !actionsEl) return;
		const ro = new ResizeObserver(replan);
		ro.observe(bar);
		// `untrack`: replan reads `spilled`, so a tracked call makes this effect its own dependency.
		untrack(replan);
		// The first measurement can land before the webfont, which no resize afterwards reports.
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

	// The residents change width from CONTENT, which no resize reports. The `tick()` is
	// load-bearing: the tab strip is a sibling tree, so a synchronous replan measures the old one.
	$effect(() => {
		void g.savePath;
		void g.unsavedChanges;
		void hudActive;
		void hs.running;
		void ws.state.workspaces;
		widthCache.invalidate();
		void tick().then(replan);
	});

	/** The canvas commands, overflow-resident at every width. They address the active editor — or,
	 * when that id is stale and one editor is open, that one; never a guess between several. */
	function canvasItems(): MenuItem[] {
		const ed = activeOrOnlyEditor(sel.activeEditorId);
		const has = ed?.hasSelection() ?? false;
		return [
			{ label: 'Select all', icon: 'square-dashed', disabled: !ed, action: () => ed?.selectAll() },
			// Multi-select's only pointer way out: with the mode on, a tap on empty canvas keeps.
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

	/** The bar's own residents, but only the ones that no longer fit. */
	function spilledItems(): MenuItem[] {
		const items: MenuItem[] = [];
		if (isSpilled('topbar-hud') && hudActive)
			items.push({ label: `${p.fps.toFixed(0)} fps`, disabled: true, action: () => {} });
		if (isSpilled('topbar-path') && (g.savePath || g.unsavedChanges)) {
			// The chip's own 32ch cap, applied to the DATA: a menu row does not ellipsize.
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
		<div class="tabslot" bind:this={tabslotEl}>{@render tabs()}</div>
	{/if}

	<div class="action-zone" bind:this={zoneEl}>
		<!-- The connection speaks only when it needs attention, and never spills into a menu. -->
		{#if g.disconnected}
			<Badge tone="warning" data-testid="topbar-connection">disconnected</Badge>
		{/if}
		{#if hs.running > 0}
			<Button
				variant="ghost"
				size="sm"
				style="--panelty-btn-ink: var(--accent)"
				data-testid="topbar-agents"
				aria-expanded={agentMenu !== null}
				title="Running agents"
				onclick={openAgents}><Icon name="bot" />{hs.running}</Button
			>
		{/if}
		<div class="actions" bind:this={actionsEl}>
			<!-- Identity and actions are ONE overflow group with ONE gap. -->
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
			<!-- `.no-caret` restores Save's right-hand corners once the caret has spilled. -->
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
		<!-- Resident at every width; its accent is multi-select mode's always-visible tell. -->
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
		background: var(--surface-1);
		/* No fixed height: the resident --hit-sized controls set it, so it grows on a coarse pointer. */
		font-size: var(--fs-body);
		z-index: 10;
	}
	.tabslot {
		flex: 1 1 auto;
		min-width: 0;
		align-self: stretch;
		display: flex;
		align-items: stretch;
	}
	/* Never a shrink absorber: an item fits whole or moves whole, which keeps the plan honest. */
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
	/* PerfHud owns the timer behind `hudActive`, so its host stays mounted; hide it while empty. */
	.hud-info:not(.active) {
		display: none;
	}
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
	/* Two boxes: `.actions` holds every item that can relocate, the trigger is chrome beside it. */
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
	/* A spilled item stays in the DOM, so its intrinsic width stays re-readable. `:global`, because
	   an action's class rides a primitive's `class` prop onto its inner <button>. */
	.actions :global(.spilled) {
		display: none;
	}
	.action-zone :global(.multi-on) {
		color: var(--accent);
		border-color: var(--accent);
	}
	/* Save split: the touching corners are squared, so the two read as one segmented control. */
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
