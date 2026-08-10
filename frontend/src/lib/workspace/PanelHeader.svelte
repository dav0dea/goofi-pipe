<!--
  Panel header: the content-type dropdown (left), the structural actions (right), and a right-click
  / long-press context menu exposing every structural action (split, maximize, change content,
  close). The dropdown and the menu both build their panel-type list from the registry, so new
  panel types appear automatically.

  The right end is a progressive overflow (D-R6) sharing `editor/overflowFit.ts` with the app
  header rather than restating its arithmetic: Split Right, Split Down and Maximize sit in the
  header while they fit and give themselves up one at a time, lowest priority first, into a ⋯ menu.
  A panel header is the harder case of the two — its width is the PANEL's, not the window's, so two
  panels side by side on a laptop are already narrower than the app header ever gets on a phone.

  Two things differ from the app header, and both are stated at the call site below:
    · the ✕ is NOT in the plan. It is the one control that must be reachable at every width, so it
      is charged off the budget rather than offered a slot in it.
    · the ⋯ is NOT resident (`residentTrigger: false`). Its menu holds the spilled actions and
      nothing else, so at a width where all three fit there is no menu to open and no trigger drawn.

  Naming: the user's "split vertical / split horizontal" is deliberately not the wording used here.
  Those two names mean opposite things in different apps — a "vertical split" is a vertical DIVIDER
  in one editor and a vertical STACK in the next — while this header already had a vocabulary that
  cannot be read two ways, in the context menu these very commands live in. So there is one wording
  and it is the existing one: Split RIGHT is `'row'` (the new panel beside this one) and Split DOWN
  is `'column'` (the new panel under it). One command, one name, two representations (D-R2) — the
  bar button and the menu row are built from the same record below, so they cannot drift.

  The two split ICONS are Lucide's `square-split-horizontal` / `square-split-vertical` as drawn,
  never one of them rotated: Lucide names them for the axis the split runs ALONG, so the
  `-horizontal` one carries the vertical divider (Split Right) and the `-vertical` one the
  horizontal divider (Split Down).
-->
<script lang="ts">
	import { countPanels, type PanelNode } from './model';
	import { workspace } from './workspace.svelte';
	import { listPanelTypes, resolvePanelType } from './registry';
	import type { MenuItem } from './menu';
	import ContextMenu from './ContextMenu.svelte';
	import { createLongPress } from '$lib/editor/longPress';
	import { createWidthCache, planOverflow, type OverflowItem } from '$lib/editor/overflowFit';
	import { Button, IconButton, Icon, type IconName } from '$lib/ui';
	import { onDestroy, untrack } from 'svelte';

	let { node }: { node: PanelNode } = $props();
	const ws = workspace();
	const type = $derived(resolvePanelType(node.panelType));
	const isMax = $derived(ws.maximizedPanelId === node.id);
	const canClose = $derived(countPanels(ws.active.root) > 1);

	// `from` discriminates which surface opened this menu: three of them share the one ContextMenu,
	// and only the ⋯ owns an `aria-expanded` that must not light up for the other two.
	let menu = $state<{ x: number; y: number; items: MenuItem[]; from?: 'overflow' } | null>(null);

	/** The ✕, and the menu row beside it. A panel type may TAKE OVER its own close (an agent panel
	 * asks whether to detach or kill first, in the panel holding the terminal); everything else
	 * closes here, as it always did. */
	function requestClose(): void {
		if (type.confirmClose?.(node.id)) return;
		ws.close(node.id);
	}

	function contentItems(): MenuItem[] {
		return listPanelTypes().map((t) => ({
			label: t.title,
			icon: t.icon,
			checked: t.id === node.panelType,
			action: () => ws.setType(node.id, t.id)
		}));
	}

	function openContent(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		menu = { x: r.left, y: r.bottom + 2, items: contentItems() };
	}

	// --- the structural actions, in one place --------------------------------

	interface HdrAction {
		/** Stable id — the button's `data-testid`, and its key in the spill order. */
		id: string;
		icon: IconName;
		/** The menu row's text. */
		label: string;
		/** The button's accessible name: an icon says nothing on its own. */
		name: string;
		run: () => void;
	}

	/**
	 * LOWEST PRIORITY FIRST — which is also the DOM order, so the header empties from the left and
	 * whatever survives stays adjacent to the ⋯ it is spilling into.
	 *
	 * Maximize is kept longest on purpose: it is the small-screen mechanism (CLAUDE.md), so it is
	 * the last action a narrowing panel should make a user open a menu for. Between the two splits,
	 * a narrow panel is a tall one, so stacking (Split Down) outlives splitting beside it.
	 */
	function actions(): HdrAction[] {
		return [
			{
				id: 'panel-split-row',
				icon: 'square-split-horizontal',
				label: 'Split Right',
				name: 'Split panel right',
				run: () => ws.split(node.id, 'row')
			},
			{
				id: 'panel-split-column',
				icon: 'square-split-vertical',
				label: 'Split Down',
				name: 'Split panel down',
				run: () => ws.split(node.id, 'column')
			},
			{
				id: 'panel-maximize',
				icon: 'maximize-2',
				label: isMax ? 'Restore' : 'Maximize',
				name: isMax ? 'Restore panel' : 'Maximize panel',
				run: () => ws.toggleMaximize(node.id)
			}
		];
	}

	const asRow = (a: HdrAction): MenuItem => ({ label: a.label, icon: a.icon, action: a.run });

	function structuralItems(): MenuItem[] {
		const [right, down, max] = actions();
		return [
			asRow(right),
			asRow(down),
			{ separator: true },
			asRow(max),
			{ label: 'Change content', items: contentItems() },
			{ separator: true },
			{ label: 'Close Panel', icon: 'x', disabled: !canClose, action: requestClose }
		];
	}

	function onHeaderContext(e: MouseEvent): void {
		e.preventDefault();
		menu = { x: e.clientX, y: e.clientY, items: structuralItems() };
	}

	/**
	 * The coarse-pointer door onto the same menu (D-R5). Split Right and Split Down had NO other
	 * door — the header is a `role="toolbar" tabindex="-1"` with no keydown handler, and its menu
	 * was `oncontextmenu`-only — so on a phone a panel could not be split at all.
	 *
	 * Armed for `touch` alone, and the editor's own recognizer is reused rather than a second
	 * gesture concept invented. It never calls `stopPropagation`, so the same pointerdown still
	 * reaches `Panel.svelte`'s capture-phase `setActive` — freezing `activePanelId` is how a
	 * swallowed press would quietly drift every panel's selection scoping.
	 */
	const headerPress = createLongPress((at) => {
		menu = { x: at.clientX, y: at.clientY, items: structuralItems() };
	});

	function onHeaderPointerDown(e: PointerEvent): void {
		if (e.pointerType !== 'touch') return;
		// The header's own controls keep their own actions: a press that landed on ✕ would both
		// open a menu and, on release, close the panel the menu describes.
		if ((e.target as HTMLElement | null)?.closest('button')) return;
		headerPress.start(e);
	}

	// A press in flight must not fire into an unmounted panel (a close, a tab switch, a split).
	onDestroy(headerPress.cancel);

	// --- progressive overflow (D-R6) -----------------------------------------

	const TRIGGER = 'panel-overflow';
	const CLOSE = 'panel-close';

	let headerEl = $state<HTMLDivElement | null>(null);
	let zoneEl = $state<HTMLDivElement | null>(null);
	let spilled = $state<Set<string>>(new Set());

	const isSpilled = (id: string): boolean => spilled.has(id);

	function px(el: Element, prop: string): number {
		return parseFloat(getComputedStyle(el).getPropertyValue(prop)) || 0;
	}

	/** The three actions' intrinsic widths and the trigger's, read with none of them hidden.
	 *
	 * The hide class is stripped and restored inside one synchronous block, so nothing is painted
	 * mid-measurement and Svelte's own class bookkeeping stays correct (it re-applies from
	 * `spilled` on the next update either way). Only runs when the root font size moved — which is
	 * what the coarse `--hit` floor does to every one of these boxes at once. */
	function measureWidths(): number[] {
		const host = zoneEl;
		if (!host) return [];
		const hidden = [...host.querySelectorAll<HTMLElement>('.spilled')];
		for (const el of hidden) el.classList.remove('spilled');
		const widths = [...actions().map((a) => a.id), TRIGGER].map(
			(id) =>
				host.querySelector<HTMLElement>(`[data-testid="${id}"]`)?.getBoundingClientRect().width ?? 0
		);
		for (const el of hidden) el.classList.add('spilled');
		return widths;
	}

	const widthCache = createWidthCache(measureWidths);

	function replan(): void {
		const bar = headerEl;
		const host = zoneEl;
		if (!bar || !host) return;
		const content = bar.querySelector<HTMLElement>('.content-btn');
		const close = host.querySelector<HTMLElement>(`[data-testid="${CLOSE}"]`);
		if (!content || !close) return;
		const widths = widthCache.widths(px(document.documentElement, 'font-size'));
		if (widths.length === 0) return;
		const ids = actions().map((a) => a.id);
		const items: OverflowItem[] = ids.map((id, i) => ({ id, width: widths[i] }));

		const gap = px(host, 'gap');
		// MEASURED each replan, off the boxes that do not move when an action spills: the header's
		// own inner width, less the content dropdown and the two gaps around the flexible spacer.
		// Never the action zone's OWN width — that is exactly what shrinks the moment an item
		// leaves, and reading it is the oscillation bug `overflowFit.ts` opens with.
		const zone =
			bar.clientWidth -
			px(bar, 'padding-left') -
			px(bar, 'padding-right') -
			content.getBoundingClientRect().width -
			px(bar, 'gap') * 2;
		// …and the ✕ is charged off the top with the gap beside it, rather than given a slot in the
		// plan. That IS the always-visible guarantee: an action can only ever spill into space the
		// close button has already been paid out of.
		const budget = zone - close.getBoundingClientRect().width - gap;

		const next = planOverflow(items, ids, {
			gap,
			budget,
			trigger: widths[widths.length - 1],
			residentTrigger: false
		});
		// Write only on a real change: the observer re-fires on the layout this write causes, and an
		// unconditional assignment would keep the effect alive forever though the plan has converged.
		if (next.size !== spilled.size || [...next].some((id) => !spilled.has(id))) spilled = next;
	}

	$effect(() => {
		const bar = headerEl;
		if (!bar || !zoneEl) return;
		const ro = new ResizeObserver(replan);
		ro.observe(bar);
		// …and the content dropdown, whose width is a term in the budget and moves on its own when
		// the panel changes type ("Node Editor" is not "Console").
		const content = bar.querySelector<HTMLElement>('.content-btn');
		if (content) ro.observe(content);
		// `untrack`: replan READS `spilled` to decide whether the plan changed, and writing it from
		// inside a tracked call would make this effect its own dependency.
		untrack(replan);
		// The first measurement can land before the webfont does, and the dropdown is a text button
		// — a different number of pixels in the fallback face, which no resize reports.
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

	/** The bar's own actions, but only the ones that no longer fit — same commands, second
	 *  representation, one record (D-R2). */
	const spilledItems = (): MenuItem[] =>
		actions()
			.filter((a) => isSpilled(a.id))
			.map(asRow);

	function openOverflow(e: MouseEvent): void {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		menu = {
			x: Math.max(6, r.right - 180),
			y: r.bottom + 4,
			items: spilledItems(),
			from: 'overflow'
		};
	}
</script>

<div
	class="panel-header"
	bind:this={headerEl}
	draggable="true"
	oncontextmenu={onHeaderContext}
	onpointerdown={onHeaderPointerDown}
	onpointermove={headerPress.move}
	onpointerup={headerPress.cancel}
	onpointercancel={headerPress.cancel}
	ondragstart={(e) => {
		// Don't start a panel move when the drag begins on a control.
		if ((e.target as HTMLElement).closest('button, select, input')) {
			e.preventDefault();
			return;
		}
		ws.dragging = { kind: 'panel', workspaceId: ws.state.activeWorkspaceId, panelId: node.id };
	}}
	ondragend={() => (ws.dragging = null)}
	role="toolbar"
	tabindex="-1"
	aria-label="Panel header"
	data-testid="panel-header"
>
	<Button variant="ghost" class="content-btn" onclick={openContent} title="Change panel content">
		{#if type.icon}<span class="ic"><Icon name={type.icon} /></span>{/if}
		<span class="title">{type.title}</span>
		<span class="caret"><Icon name="chevron-down" /></span>
	</Button>
	<div class="spacer"></div>
	<div class="hdr-actions" bind:this={zoneEl}>
		{#each actions() as a (a.id)}
			<IconButton
				variant="ghost"
				density="chrome"
				class={`hdr-btn${isSpilled(a.id) ? ' spilled' : ''}`}
				data-testid={a.id}
				title={a.label}
				label={a.name}
				onclick={a.run}><Icon name={a.icon} /></IconButton
			>
		{/each}
		<!-- Not resident: it goes when there is nothing behind it. It stays in the DOM either way so
		     its width can be measured, which is the one thing the plan needs from it. -->
		<IconButton
			variant="ghost"
			density="chrome"
			class={`hdr-btn${spilled.size ? '' : ' spilled'}`}
			data-testid={TRIGGER}
			aria-expanded={menu?.from === 'overflow'}
			title="More panel actions"
			label="More panel actions"
			onclick={openOverflow}><Icon name="ellipsis" /></IconButton
		>
		<IconButton
			variant="ghost"
			density="chrome"
			class="hdr-btn"
			data-testid={CLOSE}
			title="Close panel"
			label="Close panel"
			disabled={!canClose}
			onclick={requestClose}><Icon name="x" /></IconButton
		>
	</div>
</div>

{#if menu}
	<ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => (menu = null)} />
{/if}

<style>
	.panel-header {
		display: flex;
		align-items: center;
		height: var(--panel-header-h);
		flex: 0 0 auto;
		padding: 0 var(--space-2);
		/* The TOP rung of the panel's own ladder — body `--surface-1`, content toolbar `--surface-2`,
		   this `--surface-3` — so each adjacency is a real step and none needs a hairline (D5), not
		   even the one 26px inside the panel edge. `--surface-2` put this byte-identical to the `Bar`
		   that four of the six panel types render flush beneath it, which is the same "border deleted,
		   nothing behind it" defect one level down; a step is what D5 trades the border FOR. */
		background: var(--surface-3);
		gap: var(--space-1);
		user-select: none;
		cursor: grab;
	}
	.panel-header:active {
		cursor: grabbing;
	}
	/* The primitives keep the frozen 20px control geometry of the 26px bar. Under a coarse
	   pointer the bar itself grows to --hit (app.css), so the floors apply unchanged there.
	   The `button` tag qualifier is load-bearing: without it this ties with the primitive's own
	   `.ui-btn.s-md` padding, and the two rules live in separate built CSS chunks — a tie there
	   is settled by the emitted <link> order, not by the source.
	   (Button has no density axis: this pin is padding + gap geometry, not a hit floor.)
	   `min-width: 0` is the overflow's other half: the dropdown is the header's SHRINK ABSORBER, so
	   a long panel-type name gives way to an ellipsis instead of pushing the ✕ out of the panel —
	   the same trade the app header's status cluster makes, and the reason the ✕ can be promised at
	   every width rather than at every width anyone happened to test. */
	.panel-header :global(button.content-btn) {
		height: 20px;
		padding: 0 var(--space-3);
		gap: var(--space-2);
		min-width: 0;
	}
	/* The icon buttons state only their box — `density="chrome"` owns the coarse-pointer floor. */
	.panel-header :global(.hdr-btn) {
		--icon-btn-size: 20px;
		color: var(--text-dim);
	}
	.panel-header :global(.hdr-btn:hover:not(:disabled)) {
		color: var(--text);
	}
	/* opacity: intentional — the type icon and the caret are quieted BELOW the title they sit beside
	   (a hierarchy, not a disabled state); --disabled-opacity would read as "this header is inert".
	   `display: flex` collapses each span onto its icon's own box, so neither adds the line box a
	   text glyph used to need inside a 20px header button. */
	.ic,
	.caret {
		display: flex;
		align-items: center;
	}
	.ic {
		opacity: 0.85;
		flex: 0 0 auto;
	}
	.title {
		font-weight: 500;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.caret {
		opacity: 0.5;
		font-size: var(--fs-micro);
		flex: 0 0 auto;
	}
	.spacer {
		flex: 1;
	}
	/* The overflow-able actions and the trigger they spill into, in one rigid box the budget above
	   is measured AGAINST rather than FROM. */
	.hdr-actions {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		flex: 0 0 auto;
	}
	/* A spilled action is a row in the ⋯ menu instead; it stays in the DOM so its intrinsic width
	   can be re-read when the responsive root size moves it. `:global`, because the class rides a
	   primitive's `class` prop onto its inner <button>. */
	.hdr-actions :global(.spilled) {
		display: none;
	}
</style>
