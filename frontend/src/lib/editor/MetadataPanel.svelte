<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeData } from '$lib/api/data';
	import type { DataFrame } from '$lib/codec/decode';
	import { onMount } from 'svelte';

	type Props = { node: NodeInstanceInfo };
	const { node }: Props = $props();

	const slots = $derived(Object.keys(node.output_slots ?? {}));
	let activeSlot = $state<string | null>(null);
	let lastFrame = $state<DataFrame | null>(null);

	$effect(() => {
		const fst = slots[0] ?? null;
		if (activeSlot === null || !slots.includes(activeSlot)) activeSlot = fst;
	});

	let cleanup: (() => void) | null = null;
	$effect(() => {
		cleanup?.();
		cleanup = null;
		lastFrame = null;
		if (!activeSlot) return;
		const slot = activeSlot;
		const unsub = subscribeData(node.name, slot, (f) => {
			lastFrame = f;
		});
		cleanup = unsub;
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

<section class="panel">
	<header>
		<span>Inspector</span>
		{#if slots.length > 0}
			<select bind:value={activeSlot}>
				{#each slots as s}
					<option value={s}>{s}</option>
				{/each}
			</select>
		{/if}
	</header>

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
