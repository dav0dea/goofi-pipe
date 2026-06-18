<!--
  TouchDesigner-style "zoom out to leave a sub-patch". When inside a sub-patch and
  the user zooms out far enough, it pops up one level. Must live INSIDE <SvelteFlow>
  so the viewport/store hooks resolve from the provider context (the parent panel's
  <script> can't see them — same reason FitToGraph is a child). Renders nothing.

  The threshold is computed LIVE from the actual content: exit when zoomed to half
  the "all-node-fit" zoom (the zoom at which every currently-visible node just fits
  the viewport, capped the same way the enter-fit is). That makes it size-aware —
  a large sub-patch has a smaller fit zoom, so you must zoom out further before it
  pops — and robust to panning/zooming after entry. A short arm delay + a one-shot
  guard (the exit re-fit restores a higher zoom) prevent an immediate second pop.
-->
<script lang="ts">
	import { useViewport, useSvelteFlow, useStore, getViewportForBounds } from '@xyflow/svelte';

	let { entered, onExit }: { entered: string | null; onExit: () => void } = $props();

	const EXIT_RATIO = 0.5; // pop at half the all-node-fit zoom ("50% past fit")
	const FIT_MAX_ZOOM = 1; // must match the editor's FIT_OPTIONS maxZoom
	const FIT_PADDING = 0.18; // must match the editor's FIT_OPTIONS padding

	const vp = useViewport();
	const store = useStore();
	const { getNodesBounds } = useSvelteFlow();
	let armed = false; // plain (not a reactive dep) — only zoom changes drive the check

	$effect(() => {
		// Arm shortly after descending so the enter-fit settles first; disarm at top.
		const cur = entered;
		armed = false;
		if (!cur) return;
		const t = setTimeout(() => {
			armed = true;
		}, 220);
		return () => clearTimeout(t);
	});

	$effect(() => {
		const z = vp.current.zoom; // reactive: fires continuously while zooming
		if (!entered || !armed) return;
		const w = store.width;
		const h = store.height;
		const ids = store.nodes.map((n) => n.id);
		if (!w || !h || ids.length === 0) return;
		const bounds = getNodesBounds(ids);
		if (!bounds.width || !bounds.height) return;
		const fitZoom = getViewportForBounds(bounds, w, h, 0.05, FIT_MAX_ZOOM, FIT_PADDING).zoom;
		if (z < fitZoom * EXIT_RATIO) {
			armed = false; // one-shot; the exit re-fit restores a higher zoom
			onExit();
		}
	});
</script>
