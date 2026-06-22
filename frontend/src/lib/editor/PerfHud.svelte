<!--
  Perf HUD (backlog #12) — a compact frame-rate / drop indicator in the TopBar.
  Reads the app-wide perf stats singleton, which the frame layer (frames.ts)
  feeds on every delivered / dropped frame. We drive `tick()` on a 250ms timer
  (the meter's own window is 500ms) so the numbers refresh without a perpetual
  rAF. Hidden when idle (no frames flowing) so a static patch shows nothing.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { perfStats } from '$lib/api/perfStats.svelte';

	const p = perfStats();

	onMount(() => {
		const id = setInterval(() => p.tick(), 250);
		return () => clearInterval(id);
	});

	const active = $derived(p.fps > 0.05 || p.dps > 0.05);
</script>

{#if active}
	<span
		class="hud"
		class:dropping={p.dps > 0.5}
		data-testid="perf-hud"
		title="Frames painted per second across all visible viewers (drops = latest-wins coalesced frames)"
	>
		<span class="fps">{p.fps.toFixed(0)} fps</span>
		{#if p.dps > 0.5}
			<span class="drop">▾ {p.dps.toFixed(0)}/s</span>
		{/if}
	</span>
{/if}

<style>
	.hud {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-dim);
		padding: 2px 6px;
		border-radius: 4px;
		background: color-mix(in srgb, var(--text) 6%, transparent);
		white-space: nowrap;
	}
	.hud.dropping {
		color: var(--warning);
		background: color-mix(in srgb, var(--warning) 14%, transparent);
	}
	.drop {
		font-weight: 600;
	}
</style>
