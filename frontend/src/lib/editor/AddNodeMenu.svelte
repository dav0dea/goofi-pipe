<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { notify } from '$lib/stores/notify.svelte';
	import { dtypeColor } from './categoryColor';
	import { rankNodeTypes } from './nodeSearch';
	import { nodeTypeTitle } from './nodeTypeTitle';
	import { nodeTypeSource } from './nodeTypeSource';
	import type { NodeTypeInfo } from '$lib/api/control';
	import type { SlotClickSeed } from '$lib/stores/ui.svelte';
	import { onMount, tick } from 'svelte';
	import { EmptyState, Icon, IconButton, MODE_ATTRS } from '$lib/ui';

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

	/** The seed carries the node's UID (the routing key the pick auto-wires through). The chip is a
	 * label, so it shows the display name — falling back to the uid only if it resolves to nothing. */
	const seedName = $derived(seed ? (g.nodeById(seed.node)?.name ?? seed.node) : null);

	/** When seeded, narrow `nodeTypes` to those that expose at least one
	 * opposite-side slot of matching dtype. */
	function matchesSeed(t: NodeTypeInfo): boolean {
		if (!seed) return true;
		const candidates = seed.side === 'source' ? t.input_slots : t.output_slots;
		return Object.values(candidates).includes(seed.dtype);
	}

	const filtered = $derived.by(() => {
		const types = [...extraTypes, ...(g.nodeTypes ?? [])].filter(matchesSeed);
		// Rank by match quality (name ≫ category ≫ doc) so the closest hit leads. ONE flat list,
		// queried or not (the user's call): category headers split a short palette into shorter
		// pieces and pushed the node you wanted below the fold. Category still ranks a search — it
		// describes a node even where it no longer sorts it.
		return rankNodeTypes(types, query);
	});

	let rescanning = $state(false);

	/** Re-derive the palette from the node files on disk and say what moved. There is no watcher
	 * (decision, 2026-08-09), so this button is how a human picks up a file they just wrote or
	 * edited; an agent calls the same op through the façade. */
	async function rescan(): Promise<void> {
		rescanning = true;
		try {
			const d = await g.rescanNodes();
			const parts = [
				d.added.length && `${d.added.length} added`,
				d.changed.length && `${d.changed.length} reloaded`,
				d.removed.length && `${d.removed.length} removed`
			].filter(Boolean);
			notify().raise(parts.length ? `Nodes: ${parts.join(', ')}` : 'Nodes: no changes');
		} catch (e) {
			notify().failure('Rescan', e);
		} finally {
			rescanning = false;
		}
	}

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
			<span class="seed-ref">{seedName}.{seed.slot}</span>
			<span class="seed-dtype" style="color: {dtypeColor(seed.dtype)};">
				{seed.dtype.toLowerCase()}
			</span>
		</div>
	{/if}
	<div class="search-row">
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
		<!-- Always visible, never hover-only (D-R7): this is the only door onto a node file written
		     or edited since the app started, and touch has to reach it too. -->
		<IconButton
			label="Rescan node files"
			size="sm"
			density="chrome"
			disabled={rescanning}
			onclick={rescan}
			data-testid="add-menu-rescan"
		>
			<Icon name="refresh-cw" />
		</IconButton>
	</div>

	<div class="list" bind:this={listEl} data-testid="add-menu-list">
		{#each filtered as t, idx (t.type)}
			<button
				type="button"
				class="item"
				class:hl={idx === highlighted}
				class:unavailable={!t.available}
				title={nodeTypeTitle(t)}
				onmouseenter={() => (highlighted = idx)}
				onclick={() => pick(t)}
			>
				<span class="cat-dot"></span>
				<span class="t-name">{t.type}</span>
				<span class="t-cat">{nodeTypeSource(t)}</span>
			</button>
		{/each}
		{#if filtered.length === 0}
			<EmptyState>
				{#snippet hint()}No matches.{/snippet}
			</EmptyState>
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
	/* The search and its rescan share one strip, so the border below belongs to the row rather
	   than to the field — the field would otherwise underline only its own width. */
	.search-row {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding-right: var(--space-4);
		border-bottom: 1px solid var(--border);
	}
	input {
		flex: 1;
		min-width: 0;
		border: none;
		background: transparent;
		padding: var(--space-5) var(--space-6);
		border-radius: 0;
		font-size: var(--fs-body);
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
	/* One neutral ink, stated here — not a per-item inline style routed through a function that
	   ignored its argument. Categories no longer colour the palette (D4) and no longer group it. */
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
