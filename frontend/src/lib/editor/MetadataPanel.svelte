<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { metaEntries, formatMetaValue, metaPreview } from './metaFormat';
	import { nodeStatsRows } from './nodeStats';
	import { Icon, Select, EmptyState } from '$lib/ui';

	type Props = {
		node: NodeInstanceInfo;
		/** Show the "Metadata" header + slot dropdown. True in the editor's
		 * slide-in inspector; false in the dedicated Metadata panel, which drives
		 * the slot from its own header bar via `slotName`. */
		showHeader?: boolean;
		/** Externally-controlled slot (used with `showHeader = false`). NOT named `slot`:
		 * that is Svelte's legacy slot attribute, which must be a static value. */
		slotName?: string | null;
	};
	const { node, showHeader = true, slotName = null }: Props = $props();

	const slots = $derived(Object.keys(node.output_slots ?? {}));
	let internalSlot = $state<string | null>(null);
	let lastFrame = $state<DataFrame | null>(null);

	// Own the slot only in header mode; otherwise the parent controls it.
	$effect(() => {
		if (!showHeader) return;
		const fst = slots[0] ?? null;
		if (internalSlot === null || !slots.includes(internalSlot)) internalSlot = fst;
	});

	const activeSlot = $derived(showHeader ? internalSlot : slotName);

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
	// re-format — and the capped formatter bounds the cost.
	const fields = $derived(
		metaEntries(lastFrame?.meta).map(([key, value]) => ({
			key,
			body: formatMetaValue(value),
			preview: metaPreview(value)
		}))
	);

	// Node-level execution telemetry (the measured update rate), pushed on the status
	// plane independent of the data frame — so it shows even while we're still waiting
	// for the first frame. Empty until the node's first NODE_STATS.
	const statsRows = $derived(nodeStatsRows(node.stats));
</script>

<section class="panel" class:bare={!showHeader}>
	{#if showHeader}
		<header>
			<span>Metadata</span>
			{#if slots.length > 0}
				<Select
					class="slot-select"
					value={internalSlot ?? ''}
					onChange={(v) => (internalSlot = v)}
					options={slots}
				/>
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
			<EmptyState>
				{#snippet hint()}No metadata{/snippet}
			</EmptyState>
		{:else}
			<div class="meta-tree">
				<!-- No `open` binding, deliberately. Every field starts collapsed, which is
				     `<details>`'s own default, so the keyed element OWNS the user's choice from there
				     on. A reactive `open` cannot: this panel re-renders at the data rate and Svelte
				     re-assigns the attribute on every one of those renders, while the `toggle` event
				     reporting a click fires ASYNCHRONOUSLY — a frame landing in that gap put the stale
				     value back and silently undid the click. -->
				{#each fields as f (f.key)}
					<details class="meta-field">
						<summary>
							<span class="caret"><Icon name="chevron-right" /></span>
							<span class="mk">{f.key}</span>
							<span class="mp">{f.preview}</span>
						</summary>
						<div class="mv">{f.body}</div>
					</details>
				{/each}
			</div>
		{/if}
	{:else}
		<EmptyState>
			{#snippet hint()}Waiting for data…{/snippet}
		</EmptyState>
	{/if}
</section>

<style>
	.panel {
		padding: var(--space-6);
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
		margin-bottom: var(--space-5);
	}
	/* Bare header picker — sits at natural width on the right, not stretched across the bar. */
	header :global(.slot-select) {
		flex: 0 0 auto;
	}
	/* Node execution telemetry — a compact key/value strip directly under the
	   "Metadata" heading, updated 2 Hz from the node's NODE_STATS push. */
	.stats {
		margin: 0 0 var(--space-5);
		padding: 0 0 var(--space-5);
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1) var(--space-7);
		border-bottom: 1px solid var(--border);
	}
	.stats .stat {
		display: flex;
		gap: var(--space-3);
		align-items: baseline;
		font-family: var(--font-mono);
	}
	.stats dt {
		font-size: var(--fs-micro);
		color: var(--text-dim);
	}
	.stats dd {
		margin: 0;
		font-size: var(--fs-small);
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	.meta-tree {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	/* One collapsible section per top-level meta field. The native marker is off app-wide (app.css)
	   — it was the last thing in the UI the BROWSER drew, in its own shape and its own ink — so the
	   affordance is the app's chevron below, turned by the `[open]` state `<details>` already owns. */
	.meta-field > summary {
		display: flex;
		align-items: baseline;
		gap: var(--space-5);
		cursor: pointer;
		padding: var(--space-2) var(--space-1);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		border-radius: var(--radius-sm);
	}
	.meta-field > summary:hover {
		background: var(--surface-2);
	}
	/* `align-self`, so the row's own baseline alignment between the key and its preview — two
	   different type sizes — survives an icon that has no baseline of its own. Reduced-motion is
	   neutralised globally (F, app.css), so no per-component guard. */
	.caret {
		display: flex;
		align-self: center;
		flex: 0 0 auto;
		font-size: var(--fs-micro);
		color: var(--text-muted);
		transition: transform var(--dur-slow) var(--ease);
	}
	.meta-field[open] > summary .caret {
		transform: rotate(90deg);
	}
	.mk {
		color: var(--text);
		font-weight: 600;
	}
	.mp {
		color: var(--text-muted);
		font-size: var(--fs-micro);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.mv {
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-dim);
		/* Preserve the dict indentation/newlines, but wrap long inline lists so
		   they fill the panel width instead of one entry per line. */
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: var(--space-1) 0 var(--space-3) var(--space-7);
	}
</style>
