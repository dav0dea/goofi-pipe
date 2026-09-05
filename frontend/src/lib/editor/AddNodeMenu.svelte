<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { notify } from '$lib/stores/notify.svelte';
	import { dtypeColor } from './categoryColor';
	import { rankNodeTypes } from './nodeSearch';
	import { bareName } from './typeId';
	import { nodeTypeTitle } from './nodeTypeTitle';
	import { nodeTypeSource } from './nodeTypeSource';
	import type { NodeTypeInfo } from '$lib/api/control';
	import { boundaryType } from '$lib/api/vocab';
	import type { SlotClickSeed } from '$lib/stores/ui.svelte';
	import { seedSlot } from './seedSlot';
	import { onMount, tick } from 'svelte';
	import { EmptyState, Icon, IconButton, MODE_ATTRS } from '$lib/ui';

	type Props = {
		onPick: (type: NodeTypeInfo) => void;
		onClose: () => void;
		seed?: SlotClickSeed | null;
		/** Synthetic types prepended to the list: the In/Out boundaries, inside a sub-patch only. */
		/** Offer the sub-patch boundary types — they are a port OF a sub-patch, so they exist only
		 * inside one. */
		boundary?: boolean;
	};
	const { onPick, onClose, seed = null, boundary = false }: Props = $props();

	const g = graph();
	let query = $state('');
	let listEl = $state<HTMLDivElement | null>(null);
	let inputEl = $state<HTMLInputElement | null>(null);
	let highlighted = $state(0);

	const seedName = $derived(seed ? (g.nodeById(seed.node)?.name ?? seed.node) : null);

	function matchesSeed(t: NodeTypeInfo): boolean {
		return !seed || seedSlot(seed, t) !== undefined;
	}

	const filtered = $derived.by(() => {
		const types = (g.nodeTypes ?? [])
			.filter((t) => boundary || !boundaryType(t.type))
			.filter(matchesSeed);
		return rankNodeTypes(types, query);
	});

	let rescanning = $state(false);

	/** Re-derive the palette from the node files on disk and say what moved. */
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
		highlighted = 0;
		// The effect's only dependency; without it the list would not re-highlight.
		void filtered.length;
	});

	onMount(() => {
		tick().then(() => inputEl?.focus());
	});

	function pick(t: NodeTypeInfo): void {
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
		<!-- Native, not `TextInput`: this filters per keystroke and owns Enter/Arrow/Escape. -->
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
		<!-- Always visible, never hover-only: touch has to reach the rescan too. -->
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
				<span class="t-name">{bareName(t.type)}</span>
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
		/* Viewport-relative as well as fixed: the soft keyboard is rising as this menu lands. */
		max-height: min(360px, 45dvh);
		overflow-y: auto;
		padding: var(--space-2) 0;
	}
	/* The scoped `input` rule out-specifies app.css's coarse 16px floor, so restate it here or iOS
	   force-zooms on focus. 16px is an absolute threshold, never a type rung. */
	@media (hover: none) and (pointer: coarse) {
		input {
			font-size: 16px;
		}
	}
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
	.item.unavailable {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
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
