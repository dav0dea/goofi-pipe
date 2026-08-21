<!-- The app-wide paint rate in the TopBar, ticked off a timer rather than a perpetual rAF. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { perfStats } from '$lib/api/perfStats.svelte';

	const p = perfStats();

	onMount(() => {
		const id = setInterval(() => p.tick(), 250);
		return () => clearInterval(id);
	});

	const active = $derived(p.fps > 0.05);
</script>

{#if active}
	<span
		class="hud"
		data-testid="perf-hud"
		title="Screen paints per second, app-wide — capped at 30, and it does not climb with node count."
	>
		<span class="fps">{p.fps.toFixed(0)} fps</span>
	</span>
{/if}

<style>
	.hud {
		display: inline-flex;
		align-items: center;
		gap: var(--space-3);
		font-size: var(--fs-chrome);
		/* Tabular figures hold the column still as the counter ticks. */
		font-variant-numeric: tabular-nums;
		color: var(--text-dim);
		white-space: nowrap;
	}
</style>
