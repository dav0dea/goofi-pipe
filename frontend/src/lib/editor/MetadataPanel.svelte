<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { resolveKind } from '$lib/viewers/kind';
	import { onMount } from 'svelte';

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

	function fmtMeta(meta: Record<string, unknown>): string {
		const safe = JSON.parse(
			JSON.stringify(meta, (_k, v) => {
				if (typeof v === 'bigint') return Number(v);
				return v;
			})
		);
		return JSON.stringify(safe, null, 2);
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

	{#if lastFrame}
		<div class="meta-line">dtype: <code>{lastFrame.dtype}</code></div>
		{#if lastFrame.dtype === 'ARRAY'}
			<div class="meta-line">
				shape:
				<code
					>[{(lastFrame.data as { shape: number[] }).shape.join(', ')}]</code
				>
			</div>
			<div class="meta-line">
				np dtype: <code>{(lastFrame.data as { dtype: string }).dtype}</code>
			</div>
		{/if}
		<pre class="meta-block">{fmtMeta(lastFrame.meta)}</pre>
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
	.meta-line {
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text-dim);
		margin: 2px 0;
	}
	code {
		color: var(--text);
	}
	.meta-block {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-dim);
		background: var(--bg-elev-2);
		border-radius: 4px;
		padding: 6px;
		margin-top: 8px;
		max-height: 200px;
		overflow: auto;
		white-space: pre-wrap;
	}
	.hint {
		color: var(--text-faint);
		font-size: 11px;
	}
</style>
