<!-- ChoiceGrid — the icon-over-label tile grid an empty surface offers its choices with. -->
<script module lang="ts">
	/** One tile; `id` keys the list. */
	export interface Choice {
		id: string;
		label: string;
		icon?: string;
		/** Native tooltip: what the label had no room for. */
		title?: string;
		testid?: string;
		choose: () => void;
	}
</script>

<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'panelty';

	let {
		choices,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & { choices: Choice[] } = $props();
</script>

<div {...rest} class={`ui-choice-grid ${klass}`.trim()}>
	{#each choices as c (c.id)}
		<button class="choice" title={c.title} data-testid={c.testid} onclick={c.choose}>
			<span class="icon"><Icon name={c.icon ?? 'square-dashed'} /></span>
			<span class="label">{c.label}</span>
		</button>
	{/each}
</div>

<style>
	.ui-choice-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
		gap: var(--space-5);
		width: 100%;
		max-width: 340px;
	}
	/* A tile, not a Button, so it states its own face — a button inherits no font. */
	.choice {
		font: inherit;
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
		display: flex;
		font-size: var(--fs-title);
		line-height: 1;
		color: var(--text-dim);
	}
	.label {
		font-size: var(--fs-small);
	}
</style>
