<!--
  Perf HUD (backlog #12) — the app-wide paint rate, in the TopBar.
  Reads the perf stats singleton, which the frame layer (frames.ts) feeds once per
  PAINT — one rAF flush, however many streams it repainted. The word "fps" is kept
  because it is literally what the number is; it used to be bumped per SLOT, which
  made it the sum of the open streams' rates (~30 x N) under a label naming the
  30 fps cap.

  The drop counter that sat beside it is GONE (Phil, 2026-08-10). It summed
  coalesced frames across every stream, so it was a total standing next to a
  number that is emphatically not a total — the two could not be read against each
  other. A drop belongs to the stream whose frame was overwritten, so it now lives
  per (node, slot) in the Metadata panel beside that node's update rate.

  We drive `tick()` on a 250ms timer (the meter's own window is 500ms) so the
  numbers refresh without a perpetual rAF. Hidden when idle (no frames flowing)
  so a static patch shows nothing.
-->
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
</style>
