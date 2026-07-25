<!-- The content a freshly-split panel starts as: a grid of buttons, one per
     registered panel type, that turns this empty panel into the chosen content.
     The choices are discovered from the panel registry — the single source of
     truth — so any registered panel (including future mods) shows up here
     automatically, with no list to keep in sync. -->
<script lang="ts">
	import type { PanelProps } from './registry';
	import { listPanelTypes } from './registry';
	import { EMPTY_PANEL_TYPE } from './model';
	import { workspace } from './workspace.svelte';
	import { EmptyState } from '$lib/ui';

	let { panelId }: PanelProps = $props();
	const ws = workspace();

	// Every registered type except 'empty' itself — picking one converts this
	// panel into that content.
	const choices = listPanelTypes().filter((t) => t.id !== EMPTY_PANEL_TYPE);
</script>

<div class="empty" data-testid="empty-panel">
	<EmptyState>
		{#snippet title()}Choose panel content{/snippet}
		<div class="grid">
			{#each choices as t (t.id)}
				<button
					class="choice"
					data-panel-choice={t.id}
					title={t.title}
					onclick={() => ws.setType(panelId, t.id)}
				>
					<span class="icon">{t.icon ?? '▢'}</span>
					<span class="label">{t.title}</span>
				</button>
			{/each}
		</div>
	</EmptyState>
</div>

<style>
	/* Stretches so the EmptyState spans the panel width — the grid below the prompt sizes
	   against it (and centres itself at its max-width), rather than shrinking to content. */
	.empty {
		display: flex;
		flex-direction: column;
		justify-content: center;
		height: 100%;
		background: var(--bg);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
		gap: var(--space-5);
		width: 100%;
		max-width: 340px;
	}
	/* A tile, not a Button: an icon-over-label card whose whole face is the affordance. The
	   surface step carries the separation (D5) and the border is the hover accent alone. */
	.choice {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-4);
		padding: var(--space-6) var(--space-5);
		background: var(--surface-1);
		border: 1px solid transparent;
		border-radius: var(--radius-md);
		color: var(--text);
		cursor: pointer;
		transition:
			border-color var(--dur-slow) var(--ease),
			background var(--dur-slow) var(--ease);
	}
	.choice:hover {
		border-color: var(--accent);
		background: var(--surface-2);
	}
	.icon {
		font-size: var(--fs-title);
		line-height: 1;
		color: var(--text-dim);
	}
	.label {
		font-size: var(--fs-small);
	}
</style>
