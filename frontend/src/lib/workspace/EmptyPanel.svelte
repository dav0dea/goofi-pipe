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

	let { panelId }: PanelProps = $props();
	const ws = workspace();

	// Every registered type except 'empty' itself — picking one converts this
	// panel into that content.
	const choices = listPanelTypes().filter((t) => t.id !== EMPTY_PANEL_TYPE);
</script>

<div class="empty" data-testid="empty-panel">
	<div class="prompt">Choose panel content</div>
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
</div>

<style>
	.empty {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 14px;
		height: 100%;
		padding: 16px;
		background: var(--bg);
	}
	.prompt {
		color: var(--text-faint);
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
		gap: 10px;
		width: 100%;
		max-width: 340px;
	}
	.choice {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
		padding: 14px 10px;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		cursor: pointer;
		transition:
			border-color 100ms ease,
			background 100ms ease;
	}
	.choice:hover {
		border-color: var(--accent);
		background: var(--bg-elev-2);
	}
	.icon {
		font-size: 1.3rem;
		line-height: 1;
		color: var(--text-dim);
	}
	.label {
		font-size: 0.78rem;
	}
</style>
