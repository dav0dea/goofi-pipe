<!--
  TouchDesigner-style "zoom out to leave a sub-patch". When the editor is inside a
  sub-patch and the user zooms out far enough, it pops up one level. Must live
  INSIDE <SvelteFlow> so useViewport()/useSvelteFlow() resolve from the provider
  context (the parent panel's <script> can't see it — same reason FitToGraph is a
  child). Renders nothing.

  Threshold is ADAPTIVE: it records the fit-zoom right after entering and pops when
  the zoom drops below half of it — a fixed absolute zoom pops too eagerly on large
  sub-patches and too late on tiny ones. Re-framing on exit (the parent panel's
  exit re-fits) pushes zoom back above the threshold, and a one-shot guard prevents
  an immediate second pop — so you cleanly surface one level at a time.
-->
<script lang="ts">
	import { useViewport } from '@xyflow/svelte';

	let { entered, onExit }: { entered: string | null; onExit: () => void } = $props();

	const EXIT_RATIO = 0.5; // pop when zoom falls below half the entry fit-zoom
	const vp = useViewport();
	let baseline = $state<number | null>(null);
	let armed = false; // plain (not a reactive dep) — only zoom changes drive the check

	$effect(() => {
		// Re-baseline whenever we descend into a (new) sub-patch. The enter-fit is
		// scheduled by the panel (~60ms); record the settled zoom a bit after.
		const cur = entered;
		baseline = null;
		armed = false;
		if (!cur) return;
		const t = setTimeout(() => {
			baseline = vp.current.zoom;
			armed = true;
		}, 160);
		return () => clearTimeout(t);
	});

	$effect(() => {
		const z = vp.current.zoom; // reactive: fires continuously while zooming
		if (!entered || !armed || baseline == null) return;
		if (z < baseline * EXIT_RATIO) {
			armed = false; // one-shot; the exit re-fit restores a higher zoom
			onExit();
		}
	});
</script>
