<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { bindViewer, dropRate } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { metaEntries, formatMetaValue, metaPreview } from './metaFormat';
	import { nodeStatsRows } from './nodeStats';
	import { Icon, Select, EmptyState } from '$lib/ui';

	type Props = {
		node: NodeInstanceInfo;
		/** Show the "Metadata" header and slot dropdown, i.e. own the slot rather than take it. */
		showHeader?: boolean;
		/** Externally-controlled slot. NOT named `slot`: that is Svelte's legacy slot attribute. */
		slotName?: string | null;
	};
	const { node, showHeader = true, slotName = null }: Props = $props();

	const slots = $derived(Object.keys(node.output_slots ?? {}));
	let internalSlot = $state<string | null>(null);
	let lastFrame = $state<DataFrame | null>(null);
	/** This panel's identity in the slot's viewer registry; it binds with a null spec, so it
	 *  constrains nothing a real viewer asked for. */
	const token =
		typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `md-${Math.random()}`;

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
		return bindViewer(node.uid, slot, token, null, (f: DataFrame) => {
			lastFrame = f;
		});
	});

	// Format once per frame, not per render: this panel re-renders at the data rate.
	const fields = $derived(
		metaEntries(lastFrame?.meta).map(([key, value]) => ({
			key,
			body: formatMetaValue(value),
			preview: metaPreview(value)
		}))
	);

	// Polled, not derived: a rate must keep falling when frames stop, and only a frame re-renders.
	let drops = $state<number | null>(null);
	$effect(() => {
		const slot = activeSlot;
		drops = null;
		if (!slot) return;
		const id = setInterval(() => (drops = dropRate(node.uid, slot)), 250);
		return () => clearInterval(id);
	});

	const statsRows = $derived(nodeStatsRows(node.stats, drops));
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
				<!-- No `open` binding: `<details>` owns the user's choice. A reactive one is undone by
				     the next frame, because `toggle` fires asynchronously. -->
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
	header :global(.slot-select) {
		flex: 0 0 auto;
	}
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
	/* The native marker is off app-wide, so the chevron below is the affordance. */
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
	/* `align-self`, so the row's baseline alignment survives an icon that has no baseline. */
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
		/* Keep the dict indentation, but still wrap a long inline list. */
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: var(--space-1) 0 var(--space-3) var(--space-7);
	}
</style>
