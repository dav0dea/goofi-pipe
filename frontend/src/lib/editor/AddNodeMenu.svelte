<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from './categoryColor';
	import { rankNodeTypes } from './nodeSearch';
	import type { NodeTypeInfo } from '$lib/api/control';
	import type { SlotClickSeed } from '$lib/stores/ui.svelte';
	import { onMount, tick } from 'svelte';
	import { EmptyState, MODE_ATTRS } from '$lib/ui';

	type Props = {
		onPick: (type: NodeTypeInfo) => void;
		onClose: () => void;
		seed?: SlotClickSeed | null;
		/** Extra synthetic types prepended to the list (the In/Out boundary nodes,
		 * passed only when the editor is inside a sub-patch). */
		extraTypes?: NodeTypeInfo[];
	};
	const { onPick, onClose, seed = null, extraTypes = [] }: Props = $props();

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
		const types = [...extraTypes, ...(g.nodeTypes ?? [])].filter(matchesSeed);
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
		// Unavailable types (missing deps per the registry probe) render greyed
		// and are not addable — adding one could only fail at spawn.
		if (!t.available) return;
		onPick(t);
	}

	function itemTitle(t: NodeTypeInfo): string {
		return t.available ? t.doc : `missing dependency: ${t.missing_deps.join(', ')}`;
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
	<!-- Native, not `TextInput`: this filters per keystroke and owns Enter/Arrow/Escape, which a
	     commit-on-blur primitive cannot express. It still speaks the primitive's keyboard vocabulary
	     rather than shipping the UA defaults (sentence caps + a spellcheck squiggle on a node type). -->
	<input
		{...MODE_ATTRS.search}
		bind:this={inputEl}
		bind:value={query}
		onkeydown={onKeydown}
		placeholder={seed
			? `compatible with ${seed.dtype.toLowerCase()}…`
			: 'Type to search nodes…'}
		autocomplete="off"
		data-testid="add-menu-search"
	/>

	<div class="list" bind:this={listEl} data-testid="add-menu-list">
		{#if groups}
			{#each Object.entries(groups) as [cat, items] (cat)}
				<div class="group-header">
					<span class="dot"></span>{cat}
				</div>
				{#each items as t (t.type)}
					<button
						type="button"
						class="item"
						class:hl={filtered[highlighted]?.type === t.type}
						class:unavailable={!t.available}
						title={itemTitle(t)}
						onmouseenter={() => (highlighted = filtered.indexOf(t))}
						onclick={() => pick(t)}
					>
						<span class="cat-dot"></span>
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
					class:unavailable={!t.available}
					title={itemTitle(t)}
					onmouseenter={() => (highlighted = idx)}
					onclick={() => pick(t)}
				>
					<span class="cat-dot"></span>
					<span class="t-name">{t.type}</span>
					<span class="t-cat">{t.category}</span>
				</button>
			{/each}
			{#if filtered.length === 0}
				<EmptyState>
					{#snippet hint()}No matches.{/snippet}
				</EmptyState>
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
		font-size: var(--fs-body);
	}
	input {
		width: 100%;
		border: none;
		background: transparent;
		padding: var(--space-5) var(--space-6);
		border-bottom: 1px solid var(--border);
		border-radius: 0;
		font-size: var(--fs-body);
	}
	input:focus {
		border-color: transparent;
		border-bottom-color: var(--accent);
	}
	.list {
		/* Viewport-relative as well as fixed: 360px is most of a phone's height, and this menu opens
		   with its search focused, so the soft keyboard is on its way up as it lands. */
		max-height: min(360px, 45dvh);
		overflow-y: auto;
		padding: var(--space-2) 0;
	}
	/* Touch: a scoped `input` rule is (0,1,1) and out-specifies app.css's coarse 16px floor (0,0,1),
	   so this one field defeated it — and it is the field the add-node flow focuses on open, i.e. the
	   most reliable force-zoom in the app. The threshold is absolute, not a type rung. */
	@media (hover: none) and (pointer: coarse) {
		input {
			font-size: 16px;
		}
	}
	.group-header {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-2) var(--space-6);
		color: var(--text-muted);
		font-size: var(--fs-micro);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.group-header .dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--text-muted);
	}
	/* A full-bleed menu row, deliberately square (the wash runs edge to edge of the menu surface).
	   The fade on that wash is declared here because M-Task 7 stripped the base `button` skin it
	   used to come from — without it the highlight snaps between rows as the cursor sweeps. */
	.item {
		background: transparent;
		color: var(--text);
		border: none;
		border-radius: 0;
		display: flex;
		align-items: center;
		gap: var(--space-5);
		padding: var(--space-3) var(--space-6);
		width: 100%;
		text-align: left;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		cursor: pointer;
		transition: background var(--dur-fast) var(--ease);
	}
	.item.hl {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	/* Missing dependency (registry availability probe): visible but not addable;
	   the title names the missing package. */
	.item.unavailable {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
	/* Categories group the palette but no longer colour it (D4), so the dot is one neutral
	   ink stated here — not a per-item inline style routed through a function that ignored
	   its argument. */
	.cat-dot {
		width: 6px;
		height: 6px;
		background: var(--text-muted);
		border-radius: 50%;
		flex-shrink: 0;
	}
	.t-name {
		flex-grow: 1;
	}
	.t-cat {
		color: var(--text-muted);
		font-size: var(--fs-micro);
		text-transform: lowercase;
	}
	.seed-chip {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-6);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-bottom: 1px solid var(--border);
		font-size: var(--fs-micro);
		font-family: var(--font-mono);
	}
	.seed-arrow {
		color: var(--accent);
	}
	.seed-from {
		color: var(--text-muted);
	}
	.seed-ref {
		color: var(--text);
	}
	.seed-dtype {
		margin-left: auto;
	}
</style>
