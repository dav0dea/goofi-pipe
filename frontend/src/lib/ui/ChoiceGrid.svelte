<!--
  ChoiceGrid — the icon-over-label tile grid an empty surface offers its choices with (spec §2.5's
  companion): a responsive `auto-fit` grid of cards whose whole face is the affordance.

  It was the empty panel's, and only the empty panel's; the agent panel's launcher had its own row
  of Buttons, which is why the two "pick something to put here" surfaces did not look like one
  thing. A tile is NOT a `Button`: the surface step carries the separation (D5) and the border is
  the hover accent alone. `class` merged, `data-testid` (+ any attribute) forwarded onto the grid.
-->
<script module lang="ts">
	/** One tile. `id` keys the list; `testid` stamps the tile a consumer's e2e names. */
	export interface Choice {
		id: string;
		label: string;
		icon?: string;
		/** Native tooltip — the version string, the path, whatever the label had no room for. */
		title?: string;
		testid?: string;
		choose: () => void;
	}
</script>

<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'tatami';

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
	/* A tile, not a Button: an icon-over-label card whose whole face is the affordance. The
	   surface step carries the separation (D5) and the border is the hover accent alone. It owns
	   its font for the same reason a menu row does: buttons don't inherit one, and a complete
	   bespoke face does not lean on app.css's base reset to supply half of it. */
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
