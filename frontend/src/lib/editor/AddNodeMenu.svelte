<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { categoryColor, dtypeColor } from './categoryColor';
	import { rankNodeTypes } from './nodeSearch';
	import type { NodeTypeInfo } from '$lib/api/control';
	import type { SlotClickSeed } from '$lib/stores/ui.svelte';
	import { onMount, tick } from 'svelte';

	type Props = {
		onPick: (type: NodeTypeInfo) => void;
		onClose: () => void;
		seed?: SlotClickSeed | null;
	};
	const { onPick, onClose, seed = null }: Props = $props();

	const g = graph();
	let query = $state('');
	let listEl = $state<HTMLDivElement | null>(null);
	let inputEl = $state<HTMLInputElement | null>(null);
	let highlighted = $state(0);

	/** When seeded, narrow `nodeTypes` to those that expose at least one
	 * opposite-side slot of matching dtype. */
	function matchesSeed(t: NodeTypeInfo): boolean {
		if (!seed) return true;
		const candidates = seed.side === 'source' ? t.input_slots : t.output_slots;
		return Object.values(candidates).includes(seed.dtype);
	}

	const filtered = $derived.by(() => {
		const types = (g.nodeTypes ?? []).filter(matchesSeed);
		// Rank by match quality (name ≫ category ≫ doc) so the closest hit leads;
		// an empty query is returned untouched and grouped by category below.
		return rankNodeTypes(types, query);
	});

	// Group by category, but only when no query — when filtering, a flat
	// list keeps result density high.
	const groups = $derived.by(() => {
		if (query) return null;
		const out: Record<string, NodeTypeInfo[]> = {};
		for (const t of filtered) {
			(out[t.category] ??= []).push(t);
		}
		return out;
	});

	$effect(() => {
		// Reset highlight whenever the filtered list changes.
		highlighted = 0;
		// Drop the unused-deps lint by depending on filtered.length explicitly
		void filtered.length;
	});

	onMount(() => {
		tick().then(() => inputEl?.focus());
	});

	function pick(t: NodeTypeInfo): void {
		onPick(t);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			onClose();
		} else if (e.key === 'Enter') {
			e.preventDefault();
			const t = filtered[highlighted];
			if (t) pick(t);
		} else if (e.key === 'ArrowDown') {
			e.preventDefault();
			highlighted = Math.min(filtered.length - 1, highlighted + 1);
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			highlighted = Math.max(0, highlighted - 1);
		}
	}
</script>

<div class="add-menu" role="dialog" aria-label="Add node">
	{#if seed}
		<div class="seed-chip" data-testid="add-menu-seed">
			<span class="seed-arrow">{seed.side === 'source' ? '→' : '←'}</span>
			<span class="seed-from">from</span>
			<span class="seed-ref">{seed.node}.{seed.slot}</span>
			<span class="seed-dtype" style="color: {dtypeColor(seed.dtype)};">
				{seed.dtype.toLowerCase()}
			</span>
		</div>
	{/if}
	<input
		bind:this={inputEl}
		bind:value={query}
		onkeydown={onKeydown}
		placeholder={seed
			? `compatible with ${seed.dtype.toLowerCase()}…`
			: 'Type to search nodes…'}
		spellcheck="false"
		autocomplete="off"
		data-testid="add-menu-search"
	/>

	<div class="list" bind:this={listEl} data-testid="add-menu-list">
		{#if groups}
			{#each Object.entries(groups) as [cat, items] (cat)}
				<div class="group-header" style="--cat: {categoryColor(cat)};">
					<span class="dot"></span>{cat}
				</div>
				{#each items as t (t.type)}
					<button
						type="button"
						class="item"
						class:hl={filtered[highlighted]?.type === t.type}
						title={t.doc}
						onmouseenter={() => (highlighted = filtered.indexOf(t))}
						onclick={() => pick(t)}
					>
						<span class="cat-dot" style="background: {categoryColor(cat)};"></span>
						<span class="t-name">{t.type}</span>
						<span class="t-cat">{cat}</span>
					</button>
				{/each}
			{/each}
		{:else}
			{#each filtered as t, idx (t.type)}
				<button
					type="button"
					class="item"
					class:hl={idx === highlighted}
					title={t.doc}
					onmouseenter={() => (highlighted = idx)}
					onclick={() => pick(t)}
				>
					<span class="cat-dot" style="background: {categoryColor(t.category)};"></span>
					<span class="t-name">{t.type}</span>
					<span class="t-cat">{t.category}</span>
				</button>
			{/each}
			{#if filtered.length === 0}
				<div class="empty">No matches.</div>
			{/if}
		{/if}
	</div>
</div>

<style>
	.add-menu {
		background: var(--surface-glass);
		backdrop-filter: blur(8px);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-2);
		overflow: hidden;
		font-size: 12px;
	}
	input {
		width: 100%;
		border: none;
		background: transparent;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		border-radius: 0;
		font-size: 13px;
	}
	input:focus {
		border-color: transparent;
		border-bottom-color: var(--accent);
	}
	.list {
		max-height: 360px;
		overflow-y: auto;
		padding: 4px 0;
	}
	.group-header {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 12px;
		color: var(--text-faint);
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.group-header .dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--cat, var(--text-faint));
	}
	.item {
		background: transparent;
		color: var(--text);
		border: none;
		border-radius: 0;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 12px;
		width: 100%;
		text-align: left;
		font-family: var(--font-mono);
		font-size: 11px;
		cursor: pointer;
	}
	.item.hl {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.cat-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.t-name {
		flex-grow: 1;
	}
	.t-cat {
		color: var(--text-faint);
		font-size: 9px;
		text-transform: lowercase;
	}
	.empty {
		text-align: center;
		color: var(--text-faint);
		padding: 16px;
	}
	.seed-chip {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 12px;
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-bottom: 1px solid var(--border);
		font-size: 10px;
		font-family: var(--font-mono);
	}
	.seed-arrow {
		color: var(--accent);
	}
	.seed-from {
		color: var(--text-faint);
	}
	.seed-ref {
		color: var(--text);
	}
	.seed-dtype {
		margin-left: auto;
	}
</style>
