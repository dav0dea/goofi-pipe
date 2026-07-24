<!-- Console panel — a Chrome-devtools-style log of every node's stdout/stderr,
     Shows all nodes
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

	const LINE_H = 16; // px per text line
	const PAD = 4; // row vertical padding
	const ROW = LINE_H + PAD; // a single-line row
	const COLLAPSE_LINES = 3; // lines shown before a row collapses
	const OVERSCAN = 8;

	// Per-row expansion + measured geometry, keyed by uid. Panel-local: wrapped
	// heights depend on *this* panel's width, so they can't live in the store.
	let expanded = $state(new Set<number>());
	let measured = $state(new Map<number, { h: number; trunc: boolean }>());

	function estimateH(e: ConsoleEntry, exp: boolean): number {
		const lines = exp ? e.lines : Math.min(e.lines, COLLAPSE_LINES);
		return lines * LINE_H + PAD;
	}
	function heightOf(e: ConsoleEntry): number {
		return measured.get(e.uid)?.h ?? estimateH(e, expanded.has(e.uid));
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

	// Subscribe to SSE for exactly what this panel shows: one node when filtered,
	// every node when not. Released on unmount so closing the console disconnects.
	$effect(() => {
	});
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
		for (let i = 0; i < n; i++) cum[i + 1] = cum[i] + heightOf(v!.get(i));
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
	<div class="bar">
		<button
			class="chip"
			class:on={showStdout}
			onclick={() => (showStdout = !showStdout)}
			title="Show stdout">out</button
		>
		<button
			class="chip err"
			class:on={showStderr}
			onclick={() => (showStderr = !showStderr)}
			title="Show stderr">err</button
		>
		<span class="spacer"></span>
		{#if filterName}
			<span class="fl">filtering</span>
			<span class="fn" title={filterLabel}>{filterLabel}</span>
			<button class="clearf" title="Show all nodes" aria-label="Clear filter" onclick={clearFilter}
				>✕</button
			>
		{/if}
	</div>

	<div class="scroll" bind:this={scrollEl} bind:clientHeight={viewportH} onscroll={onScroll}>
		{#if layout.n === 0}
			<div class="empty">No output{filterName ? ' for this node' : ''} yet.</div>
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
							<span class="cnt" data-testid="console-count" title="{row.e.count} occurrences"
								>×{row.e.count}</span
							>
						{/if}
						<button
							class="copy"
							data-testid="console-copy"
							title="Copy message"
							aria-label="Copy message"
							onmousedown={(ev) => ev.stopPropagation()}
							onclick={(ev) => {
								ev.stopPropagation();
								copy(row.e.text, row.e.uid);
							}}>{copiedUid === row.e.uid ? '✓' : '⧉'}</button
						>
					</div>
				</div>
			{/each}
			<div style="height:{bottomPad}px"></div>
		{/if}
	</div>

	{#if !stuck && layout.n > 0}
		<button
			class="to-bottom"
			data-testid="console-to-bottom"
			title="Scroll to bottom"
			aria-label="Scroll to bottom"
			onclick={scrollToBottom}>↓</button
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
	.bar {
		display: flex;
		align-items: center;
		gap: 6px;
		flex: 0 0 auto;
		padding: 4px 8px;
		background: var(--bg-elev-1);
		border-bottom: 1px solid var(--border);
		font-size: 0.78rem;
	}
	.spacer {
		flex: 1 1 auto;
	}
	.chip {
		font-family: var(--font-mono);
		font-size: 10px;
		padding: 1px 8px;
		border-radius: 999px;
		border: 1px solid var(--border);
		background: transparent;
		color: var(--text-faint);
		cursor: pointer;
	}
	.chip.on {
		color: var(--text);
		border-color: var(--border-strong);
		background: var(--bg-elev-3);
	}
	.chip.err.on {
		color: var(--danger);
		border-color: color-mix(in srgb, var(--danger) 50%, transparent);
		background: color-mix(in srgb, var(--danger) 14%, transparent);
	}
	.fl {
		color: var(--text-faint);
	}
	.fn {
		font-family: var(--font-mono);
		color: var(--text);
		max-width: 140px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.clearf {
		width: 18px;
		height: 18px;
		display: grid;
		place-items: center;
		padding: 0;
		font-size: 0.7rem;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-faint);
		cursor: pointer;
	}
	.clearf:hover {
		color: var(--danger);
		background: var(--bg-elev-2);
	}
	.scroll {
		flex: 1;
		overflow-y: auto;
		overflow-x: hidden;
		min-height: 0;
		font-family: var(--font-mono);
		font-size: 11px;
	}
	.row {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		padding: 2px 10px;
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
		color: var(--text-faint);
		font-size: 9px;
	}
	.node {
		flex: 0 0 auto;
		background: transparent;
		border: none;
		padding: 0;
		line-height: 16px;
		color: var(--accent);
		font-family: var(--font-mono);
		font-size: 10px;
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
		font-family: var(--font-mono);
		font-size: 11px;
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
		gap: 4px;
	}
	.cnt {
		line-height: 16px;
		font-size: 10px;
		color: var(--text);
		background: var(--bg-elev-3);
		border: 1px solid var(--border-strong);
		border-radius: 999px;
		padding: 0 6px;
	}
	/* Hover-only per-message copy. Always occupies its slot (no reflow on hover);
	   only fades in — and becomes clickable — when the row is hovered/focused. */
	.copy {
		width: 18px;
		height: 18px;
		display: grid;
		place-items: center;
		padding: 0;
		font-size: 11px;
		line-height: 1;
		background: var(--bg-elev-3);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.1s;
	}
	.row:hover .copy,
	.row:focus-within .copy {
		opacity: 1;
		pointer-events: auto;
	}
	.copy:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.empty {
		color: var(--text-faint);
		font-size: 11px;
		padding: 10px;
	}
	/* Appears only while scrolled up; jumps back to the live tail. */
	.to-bottom {
		position: absolute;
		right: 12px;
		bottom: 12px;
		width: 30px;
		height: 30px;
		display: grid;
		place-items: center;
		padding: 0;
		font-size: 15px;
		line-height: 1;
		border-radius: 999px;
		border: 1px solid var(--border-strong);
		background: var(--bg-elev-3);
		color: var(--text);
		cursor: pointer;
		box-shadow: var(--shadow-1);
		z-index: 2;
	}
	.to-bottom:hover {
		border-color: var(--accent);
		color: var(--accent);
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
