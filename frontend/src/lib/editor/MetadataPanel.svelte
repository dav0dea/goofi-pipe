<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { metaEntries, formatMetaValue, metaPreview, isLarge } from './metaFormat';
	import { nodeStatsRows } from './nodeStats';

	type Props = {
		node: NodeInstanceInfo;
		/** Show the "Metadata" header + slot dropdown. True in the editor's
		 * slide-in inspector; false in the dedicated Metadata panel, which drives
		 * the slot from its own header bar via `slot`. */
		showHeader?: boolean;
		/** Externally-controlled slot (used with `showHeader = false`). */
		slot?: string | null;
	};
	const { node, showHeader = true, slot: slotProp = null }: Props = $props();

	const slots = $derived(Object.keys(node.output_slots ?? {}));
	let internalSlot = $state<string | null>(null);
	let lastFrame = $state<DataFrame | null>(null);

	// Own the slot only in header mode; otherwise the parent controls it.
	$effect(() => {
		if (!showHeader) return;
		const fst = slots[0] ?? null;
		if (internalSlot === null || !slots.includes(internalSlot)) internalSlot = fst;
	});

	const activeSlot = $derived(showHeader ? internalSlot : slotProp);

	$effect(() => {
		lastFrame = null;
		const slot = activeSlot;
		if (!slot) return;
		// The inspector only reads frame.meta. It shares the slot's single reduced
		// stream — no viewer contributes a ViewSpec on its behalf, so if the inspector
		// is the ONLY subscriber the frame arrives full-resolution; that's fine, the
		// meta it reads is identical either way.
		const unsub = subscribeFrames(node.uid, slot, (f) => {
			lastFrame = f;
		});
		return () => unsub();
	});

	// Derive the rendered fields ONCE per frame (the panel re-renders at the data
	// rate). Each field's body/preview is precomputed so the template doesn't
	// re-format — and the capped formatter bounds the cost. `defaultOpen` is only
	// the INITIAL collapse state (large fields start collapsed).
	const fields = $derived(
		metaEntries(lastFrame?.meta).map(([key, value]) => ({
			key,
			body: formatMetaValue(value),
			preview: metaPreview(value),
			defaultOpen: !isLarge(value)
		}))
	);

	// The user's per-field collapse choice, keyed by field name, persisted for the
	// life of this viewer. Without it, binding `open` to the per-frame-derived
	// default would re-expand a manually-collapsed field on the next node tick.
	let manualOpen = $state<Record<string, boolean>>({});

	// Node-level execution telemetry (update rate + mean process() time), pushed on
	// the status plane independent of the data frame — so it shows even while we're
	// still waiting for the first frame. Empty until the node's first NODE_STATS.
	const statsRows = $derived(nodeStatsRows(node.stats));

	function isOpen(key: string, defaultOpen: boolean): boolean {
		return manualOpen[key] ?? defaultOpen;
	}
	function onToggle(key: string, e: Event): void {
		manualOpen[key] = (e.currentTarget as HTMLDetailsElement).open;
	}
</script>

<section class="panel" class:bare={!showHeader}>
	{#if showHeader}
		<header>
			<span>Metadata</span>
			{#if slots.length > 0}
				<select bind:value={internalSlot}>
					{#each slots as s}
						<option value={s}>{s}</option>
					{/each}
				</select>
			{/if}
		</header>
	{/if}

	{#if statsRows.length > 0}
		<dl class="stats" data-testid="node-stats">
			{#each statsRows as row (row.label)}
				<div class="stat">
					<dt>{row.label}</dt>
					<dd>{row.value}</dd>
				</div>
			{/each}
		</dl>
	{/if}

	{#if lastFrame}
		{#if fields.length === 0}
			<div class="hint">No metadata</div>
		{:else}
			<div class="meta-tree">
				{#each fields as f (f.key)}
					<details
						class="meta-field"
						open={isOpen(f.key, f.defaultOpen)}
						ontoggle={(e) => onToggle(f.key, e)}
					>
						<summary>
							<span class="mk">{f.key}</span>
							<span class="mp">{f.preview}</span>
						</summary>
						<div class="mv">{f.body}</div>
					</details>
				{/each}
			</div>
		{/if}
	{:else}
		<div class="hint">Waiting for data…</div>
	{/if}
</section>

<style>
	.panel {
		padding: 12px;
		border-top: 1px solid var(--border);
	}
	/* Dedicated Metadata panel: no stacked-inspector divider or header. */
	.panel.bare {
		border-top: none;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-weight: 600;
		margin-bottom: 8px;
	}
	header select {
		font-family: var(--font-mono);
		font-size: 10px;
		padding: 2px 6px;
	}
	/* Node execution telemetry — a compact key/value strip directly under the
	   "Metadata" heading, updated ~1 Hz from the node's NODE_STATS push. */
	.stats {
		margin: 0 0 8px;
		padding: 0 0 8px;
		display: flex;
		flex-wrap: wrap;
		gap: 2px 16px;
		border-bottom: 1px solid var(--border);
	}
	.stats .stat {
		display: flex;
		gap: 6px;
		align-items: baseline;
		font-family: var(--font-mono);
	}
	.stats dt {
		font-size: 10px;
		color: var(--text-dim);
	}
	.stats dd {
		margin: 0;
		font-size: 11px;
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	.meta-tree {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	/* One collapsible section per top-level meta field. */
	.meta-field > summary {
		display: flex;
		align-items: baseline;
		gap: 8px;
		cursor: pointer;
		padding: 3px 2px;
		font-family: var(--font-mono);
		font-size: 11px;
		border-radius: 3px;
		list-style-position: inside;
	}
	.meta-field > summary:hover {
		background: var(--bg-elev-2);
	}
	.mk {
		color: var(--text);
		font-weight: 600;
	}
	.mp {
		color: var(--text-faint);
		font-size: 10px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.mv {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-dim);
		/* Preserve the dict indentation/newlines, but wrap long inline lists so
		   they fill the panel width instead of one entry per line. */
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: 2px 0 6px 16px;
	}
	.hint {
		color: var(--text-faint);
		font-size: 11px;
	}
</style>
