<!-- Console panel — a Chrome-devtools-style log of every node's stdout/stderr. The lines arrive on
     the control WS like every other event and are buffered by `stores/console.svelte.ts`, so this
     panel subscribes to a ring, never to a transport. Shows all nodes
     by default; drag a node onto it to filter to just that node. stdout/stderr
     chips filter by stream.

     Every entry renders the same way: wrapped monospace text, clamped to 3 lines
     by default (CSS line-clamp). An entry that overflows 3 lines shows a caret
     and clicks to expand to its full height. The list is virtualized over the
     console store's ring buffer with a cumulative-height model — each rendered
     row's real height is measured (ResizeObserver) and fed into a prefix-sum so
     the scrollbar stays accurate at any line count. Auto-scrolls to the newest
     line unless the user has scrolled up. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import { consoleStore, type ConsoleEntry, type ConsoleView } from '$lib/stores/console.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { linkedNodeName, withLinkedNode } from '$lib/workspace/panelState';
	import { copyText } from '$lib/clipboard';
	import { COLLAPSE_LINES, estimateRowHeight } from './consoleRowHeight';
	import { Bar, Chip, Badge, IconButton, EmptyState } from '$lib/ui';
	import { onDestroy, tick } from 'svelte';

	let { panelId, state: linkState, setState }: PanelProps = $props();
	const sel = selection();
	const uiStore = ui();
	const cs = consoleStore();

	const filterName = $derived(linkedNodeName(linkState)); // the bound node's uid (identity)
	// The chip shows the readable display name, not the raw uid.
	const filterLabel = $derived(filterName ? (graph().nodeById(filterName)?.name ?? filterName) : null);
	const dragActive = $derived(uiStore.nodeDrag !== null);
	const over = $derived(uiStore.nodeDragTarget === panelId);

	let showStdout = $state(true);
	let showStderr = $state(true);

	const OVERSCAN = 8;

	// Per-row expansion + measured geometry, keyed by uid. Panel-local: wrapped
	// heights depend on *this* panel's width, so they can't live in the store.
	let expanded = $state(new Set<number>());
	let measured = $state(new Map<number, { h: number; trunc: boolean }>());

	/**
	 * The row's content floor, in the px `estimateRowHeight` computes in (C16). A row is one 16px
	 * text line on a fine pointer, but every control it hosts — the node chip and the copy button —
	 * is floored to `--hit` under a coarse one, so the row is too while its TEXT still says 16.
	 *
	 * Read from the same token, behind the same query, the CSS floors with, so the two cannot drift.
	 * Not a device branch: a measurement of the floor that is actually in force.
	 */
	function contentFloor(): number {
		if (typeof window === 'undefined') return 0;
		if (!window.matchMedia('(hover: none) and (pointer: coarse)').matches) return 0;
		return parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit')) || 0;
	}
	function heightOf(e: ConsoleEntry, floor: number): number {
		return measured.get(e.uid)?.h ?? estimateRowHeight(e.lines, expanded.has(e.uid), floor);
	}
	function expandable(e: ConsoleEntry): boolean {
		// Truncated once measured; before that, fall back to logical line count.
		return measured.get(e.uid)?.trunc ?? e.lines > COLLAPSE_LINES;
	}

	// Observe a rendered row: record its height (for the offset model) and, while
	// collapsed, whether its text is clipped (→ expandable). ResizeObserver fires
	// after layout, so writing `measured` here can't recurse into the row's size.
	function measure(node: HTMLElement, params: { uid: number; exp: boolean }) {
		let cur = params;
		const report = (): void => {
			const h = node.offsetHeight;
			const txt = node.querySelector('.txt');
			const trunc = !cur.exp && txt ? txt.scrollHeight - txt.clientHeight > 1 : undefined;
			const prev = measured.get(cur.uid);
			const next = { h, trunc: trunc ?? prev?.trunc ?? false };
			if (!prev || prev.h !== next.h || prev.trunc !== next.trunc) {
				const m = new Map(measured);
				m.set(cur.uid, next);
				measured = m;
			}
		};
		const ro = new ResizeObserver(report);
		ro.observe(node);
		report();
		return {
			update(next: { uid: number; exp: boolean }) {
				cur = next;
				report();
			},
			destroy: () => ro.disconnect()
		};
	}

	function toggle(uid: number): void {
		const next = new Set(expanded);
		if (next.has(uid)) next.delete(uid);
		else next.add(uid);
		expanded = next;
	}

	// A text-selection drag ends with a mouseup→click on the row, which would
	// otherwise toggle it. Record where the press started; only a stationary click
	// with nothing selected counts as a pure click that toggles.
	let downX = 0;
	let downY = 0;
	function onRowDown(ev: MouseEvent): void {
		downX = ev.clientX;
		downY = ev.clientY;
	}
	function onRowClick(ev: MouseEvent, uid: number, canToggle: boolean): void {
		if (!canToggle) return;
		// A text-selection drag travels before the mouseup; a pure click barely
		// moves. Anything past a few px is a selection, not a toggle.
		if (Math.hypot(ev.clientX - downX, ev.clientY - downY) > 4) return;
		toggle(uid);
	}

	// Copy one entry's full text; flash a check on the button briefly.
	let copiedUid = $state(-1);
	let copiedTimer: ReturnType<typeof setTimeout> | undefined;
	async function copy(text: string, uid: number): Promise<void> {
		if (!(await copyText(text))) {
			console.warn('clipboard copy failed');
			return;
		}
		copiedUid = uid;
		clearTimeout(copiedTimer);
		copiedTimer = setTimeout(() => (copiedUid = -1), 1000);
	}

	onDestroy(() => {
		clearTimeout(copiedTimer);
	});

	// Acquire a console view (ring) for the active filter; release the prior one.
	// Each view has its own uid space (restarting at 0), so reset the uid-keyed
	// geometry/expansion when switching views — otherwise a new filter would
	// inherit the prior view's heights for colliding uids.
	let view = $state<ConsoleView | null>(null);
	$effect(() => {
		const v = cs.acquireView(filterName, showStdout, showStderr);
		view = v;
		expanded = new Set();
		measured = new Map();
		return () => cs.releaseView(v.sig);
	});

	let scrollEl = $state<HTMLDivElement | null>(null);
	let scrollTop = $state(0);
	let viewportH = $state(0);
	let stuck = $state(true); // pinned to the bottom until the user scrolls up

	// Cumulative row offsets: cum[i] = total height of rows [0, i). Rebuilt only
	// when the entry set, expansion, or measured heights change (NOT on a
	// coalesce count-bump — that's what layoutVersion gates).
	const layout = $derived.by<{ n: number; cum: Float64Array; height: number }>(() => {
		cs.layoutVersion;
		measured;
		expanded;
		const v = view;
		const n = v ? v.total() : 0;
		const cum = new Float64Array(n + 1);
		const floor = contentFloor(); // once per rebuild, not once per row
		for (let i = 0; i < n; i++) cum[i + 1] = cum[i] + heightOf(v!.get(i), floor);
		return { n, cum, height: cum[n] };
	});

	// Largest index i with cum[i] <= y.
	function indexAt(cum: Float64Array, y: number): number {
		let lo = 0;
		let hi = cum.length - 1;
		while (lo < hi) {
			const mid = (lo + hi + 1) >> 1;
			if (cum[mid] <= y) lo = mid;
			else hi = mid - 1;
		}
		return lo;
	}

	const start = $derived(Math.max(0, indexAt(layout.cum, scrollTop) - OVERSCAN));
	const end = $derived(Math.min(layout.n, indexAt(layout.cum, scrollTop + viewportH) + OVERSCAN + 1));
	// Shallow-copy each visible entry so the keyed {#each} re-renders the row when
	// `count` is bumped in place on coalesce (a same-reference item would not).
	const windowRows = $derived.by<{ e: ConsoleEntry; exp: boolean; canToggle: boolean }[]>(() => {
		cs.version;
		const v = view;
		if (!v) return [];
		const out: { e: ConsoleEntry; exp: boolean; canToggle: boolean }[] = [];
		const e = Math.min(end, v.total());
		for (let i = start; i < e; i++) {
			const copy = { ...v.get(i) };
			const exp = expanded.has(copy.uid);
			out.push({ e: copy, exp, canToggle: exp || expandable(copy) });
		}
		return out;
	});
	const topPad = $derived(layout.cum[Math.min(start, layout.n)]);
	const bottomPad = $derived(Math.max(0, layout.height - layout.cum[Math.min(end, layout.n)]));

	function onScroll(): void {
		if (!scrollEl) return;
		scrollTop = scrollEl.scrollTop;
		const dist = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight;
		stuck = dist < 24;
	}

	function scrollToBottom(): void {
		if (!scrollEl) return;
		scrollEl.scrollTop = scrollEl.scrollHeight;
		stuck = true;
	}

	// Keep pinned to the bottom as the content height grows (new lines, expansion).
	$effect(() => {
		layout.height;
		if (stuck && scrollEl) {
			void tick().then(() => {
				if (scrollEl && stuck) scrollEl.scrollTop = scrollEl.scrollHeight;
			});
		}
	});

	function focus(name: string): void {
		if (sel.activeEditorId) sel.selectNodes(sel.activeEditorId, [name]);
	}
	function clearFilter(): void {
		setState(withLinkedNode(linkState, null));
	}
</script>

<div class="wrap" data-testid="console-panel">
	<Bar>
		{#snippet start()}
			<Chip
				tone={showStdout ? 'accent' : 'neutral'}
				aria-pressed={showStdout}
				onclick={() => (showStdout = !showStdout)}
				title="Show stdout">out</Chip
			>
			<Chip
				tone={showStderr ? 'danger' : 'neutral'}
				aria-pressed={showStderr}
				onclick={() => (showStderr = !showStderr)}
				title="Show stderr">err</Chip
			>
		{/snippet}
		{#snippet end()}
			{#if filterName}
				<span class="fl">filtering</span>
				<span class="fn" title={filterLabel}>{filterLabel}</span>
				<IconButton
					variant="ghost"
					size="sm"
					title="Show all nodes"
					label="Clear filter"
					onclick={clearFilter}>✕</IconButton
				>
			{/if}
		{/snippet}
	</Bar>

	<div
		class="scroll thin-scrollbar"
		bind:this={scrollEl}
		bind:clientHeight={viewportH}
		onscroll={onScroll}
	>
		{#if layout.n === 0}
			<EmptyState>
				{#snippet hint()}No output{filterName ? ' for this node' : ''} yet.{/snippet}
			</EmptyState>
		{:else}
			<div style="height:{topPad}px"></div>
			{#each windowRows as row (row.e.uid)}
				<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
				<!-- A toggleable row carries role="button" + tabindex in the same branch;
				     the static check can't pair the conditional attributes. -->
				<div
					class="row"
					class:err={row.e.stream === 'stderr'}
					class:toggleable={row.canToggle}
					data-testid="console-entry"
					data-node={row.e.node}
					data-stream={row.e.stream}
					role={row.canToggle ? 'button' : undefined}
					tabindex={row.canToggle ? 0 : undefined}
					onmousedown={onRowDown}
					onclick={(ev) => onRowClick(ev, row.e.uid, row.canToggle)}
					onkeydown={(ev) => {
						if (row.canToggle && (ev.key === 'Enter' || ev.key === ' ')) {
							ev.preventDefault();
							toggle(row.e.uid);
						}
					}}
					use:measure={{ uid: row.e.uid, exp: row.exp }}
				>
					<span class="caret">{row.exp ? '▾' : row.canToggle ? '▸' : ''}</span>
					{#if !filterName}
						<button
							class="node"
							onclick={(ev) => {
								ev.stopPropagation();
								focus(row.e.node);
							}}>{row.e.node}</button
						>
					{/if}
					<pre class="txt" class:clamp={!row.exp}>{row.e.text}</pre>
					<div class="actions">
						{#if row.e.count > 1}
							<Badge data-testid="console-count" title="{row.e.count} occurrences"
								>×{row.e.count}</Badge
							>
						{/if}
						<!-- `ghost`, like every other chrome-density icon button in the app. The coarse
						     door below rests this one open, and `.ui-icon-btn`'s base paint (a filled
						     surface plus a border) would then draw a 44×44 box on every console row —
						     the highest repetition rate anywhere in the UI, against app.css's own rule
						     that a surface step is what carries separation. On a fine pointer it was
						     invisible (16px at `opacity: 0`), which is why nobody saw it. -->
						<IconButton
							class="console-copy-btn"
							variant="ghost"
							size="sm"
							density="chrome"
							data-testid="console-copy"
							title="Copy message"
							label="Copy message"
							onmousedown={(ev) => ev.stopPropagation()}
							onclick={(ev) => {
								ev.stopPropagation();
								copy(row.e.text, row.e.uid);
							}}>{copiedUid === row.e.uid ? '✓' : '⧉'}</IconButton
						>
					</div>
				</div>
			{/each}
			<div style="height:{bottomPad}px"></div>
		{/if}
	</div>

	{#if !stuck && layout.n > 0}
		<IconButton
			class="to-bottom-fab"
			data-testid="console-to-bottom"
			title="Scroll to bottom"
			label="Scroll to bottom"
			onclick={scrollToBottom}>↓</IconButton
		>
	{/if}

	{#if dragActive}
		<div class="drop-hint" class:active={over} data-testid="node-drop-hint"></div>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
		height: 100%;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
	.fl {
		color: var(--text-muted);
		font-size: var(--fs-small);
	}
	.fn {
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text);
		max-width: 140px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The virtual scroller keeps its own DOM handle (scrollTop / measured heights / onscroll), so
	   the container stays a native div rather than the ScrollArea component; it wears the shared
	   `.thin-scrollbar` skin (app.css) instead of restating it. */
	.scroll {
		flex: 1;
		overflow-y: auto;
		overflow-x: hidden;
		min-height: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.row {
		display: flex;
		align-items: flex-start;
		gap: var(--space-5);
		/* The 2px sides are mirrored by `PAD = 4` in consoleRowHeight.ts (and `line-height: 16px` by
		   its `LINE_H`):
		   the pre-measurement height estimate is computed in px, so a rem here would make the
		   estimate wrong at every root size but 14. */
		padding: 2px var(--space-6);
		border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
		box-sizing: border-box;
	}
	.row.toggleable {
		cursor: pointer;
	}
	.row.toggleable:hover {
		background: color-mix(in srgb, var(--accent) 7%, transparent);
	}
	.row.err {
		background: color-mix(in srgb, var(--danger) 9%, transparent);
		color: var(--danger);
	}
	.row.err.toggleable:hover {
		background: color-mix(in srgb, var(--danger) 16%, transparent);
	}
	.caret {
		flex: 0 0 auto;
		width: 10px;
		line-height: 16px;
		color: var(--text-muted);
		font-size: var(--fs-micro);
	}
	.node {
		flex: 0 0 auto;
		background: transparent;
		border: none;
		padding: 0;
		line-height: 16px;
		color: var(--accent);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		cursor: pointer;
		max-width: 160px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.txt {
		flex: 1 1 auto;
		min-width: 0;
		margin: 0;
		font-size: var(--fs-small);
		line-height: 16px;
		white-space: pre-wrap;
		word-break: break-word;
		overflow: hidden;
		color: inherit;
		user-select: text;
		cursor: text;
	}
	.txt.clamp {
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
	}
	.actions {
		flex: 0 0 auto;
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
	/* Hover-only per-message copy (an IconButton). Always occupies its slot (no reflow on hover);
	   only fades in — and becomes clickable — when the row is hovered/focused.
	   A row IS a chrome strip: it is one 16px text line tall, shorter than --hit, so the button
	   states its box through the primitive's `density="chrome"` seam. Anything taller drives the
	   row's height instead of the text and desyncs `estimateRowHeight`'s model. IconButton restores
	   the --hit floor under a coarse pointer by itself, and the row grows with it — which is right
	   (the row is the tap target) and is why the estimate carries the same floor (C16, closed). */
	.row :global(.console-copy-btn) {
		--icon-btn-size: 16px;
		opacity: 0;
		pointer-events: none;
		transition: opacity var(--dur-fast) var(--ease);
	}
	.row:hover :global(.console-copy-btn),
	.row:focus-within :global(.console-copy-btn) {
		opacity: 1;
		pointer-events: auto;
	}
	/* Touch: copying a log line is an ACTION, and it was reachable by hover or keyboard focus only —
	   neither of which a phone has, so it did not exist there at all (C15). It rests open instead.
	   The button already occupies its slot when hidden, so nothing reflows and the row-height model
	   is untouched. Its own `--hit` floor (IconButton's coarse `::after`) is what makes it tappable. */
	@media (hover: none) and (pointer: coarse) {
		.row :global(.console-copy-btn) {
			opacity: 1;
			pointer-events: auto;
		}
	}
	/* Appears only while scrolled up; jumps back to the live tail. Round FAB chrome on top of
	   the IconButton primitive. */
	.wrap :global(.to-bottom-fab) {
		position: absolute;
		right: 12px;
		bottom: 12px;
		border-radius: 999px;
		box-shadow: var(--shadow-1);
		z-index: 2;
	}
	.drop-hint {
		position: absolute;
		inset: 4px;
		pointer-events: none;
		border: 2px dashed color-mix(in srgb, var(--accent) 55%, transparent);
		border-radius: var(--radius-sm);
		z-index: var(--z-drag-ghost);
	}
	.drop-hint.active {
		border-style: solid;
		background: color-mix(in srgb, var(--accent) 16%, transparent);
	}
</style>
