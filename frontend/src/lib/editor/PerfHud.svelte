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
	import { Icon } from '$lib/ui';

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
		data-testid="perf-hud"
		title="Frames painted per second across all visible viewers (drops = latest-wins coalesced frames)"
	>
		<span class="fps">{p.fps.toFixed(0)} fps</span>
		<!-- This slot stays present at zero. Occasional coalescing is normal, and mounting/unmounting
		     the counter made that normal fluctuation shove the FPS readout sideways every 500ms. -->
		<span class="drop" data-testid="perf-drops"
			><Icon name="chevron-down" /> {p.dps.toFixed(0)}/s</span
		>
	</span>
{/if}

<style>
	/* Quiet text, exactly like the patch name beside it — the boxed pill is gone (Phil,
	   2026-08-08: the bar reads calmer as one text row than as a row of mixed containers),
	   and the size is the bar's shared integer chrome size so its baseline is the same one. */
	.hud {
		display: inline-flex;
		align-items: center;
		gap: var(--space-3);
		font-size: var(--fs-chrome);
		/* The counters change every frame, and the bar reads as chrome (sans) — proportional digits
		   would make the whole row twitch as they tick. Tabular figures hold each column still. */
		font-variant-numeric: tabular-nums;
		color: var(--text-dim);
		white-space: nowrap;
	}
	/* Weight, not a status ink. A "drop" is latest-wins coalescing — `frames.ts` states that it
	   costs nothing ("no viewer is starved and no queue grows") and the tooltip above says so to the
	   user. This used to flip the whole HUD to `--warning`, the same amber the bar beside it paints
	   on a LOST CONTROL CONNECTION, so a healthy 2-3 node patch sat amber at ~4% coalescing. There
	   is no threshold that makes a designed behaviour a fault; pinned in theme/styleDrift.test.ts.
	   It remains in the row at zero so its changing value cannot reflow the FPS counter. */
	.drop {
		font-weight: 600;
	}
</style>
