<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { resolveKind } from '$lib/viewers/kind';
	import { metaEntries, formatMetaValue, metaPreview, isLarge } from './metaFormat';

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
		// The inspector only reads frame.meta, which is identical across viewer kinds;
		// subscribe with the slot's dtype-default kind so the wire frame stays small.
		const kind = resolveKind(node.output_slots?.[slot] ?? null, undefined);
		const unsub = subscribeFrames(node.name, slot, kind, (f) => {
			lastFrame = f;
		});
		return () => unsub();
	});

	// Derive the rendered fields ONCE per frame (the panel re-renders at the data
	// rate). Each field's body/preview/collapse state is precomputed so the
	// template doesn't re-format — and the capped formatter bounds the cost.
	const fields = $derived(
		metaEntries(lastFrame?.meta).map(([key, value]) => ({
			key,
			body: formatMetaValue(value),
			preview: metaPreview(value),
			open: !isLarge(value)
		}))
	);
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

	{#if lastFrame}
		{#if fields.length === 0}
			<div class="hint">No metadata</div>
		{:else}
			<div class="meta-tree">
				{#each fields as f (f.key)}
					<details class="meta-field" open={f.open}>
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
