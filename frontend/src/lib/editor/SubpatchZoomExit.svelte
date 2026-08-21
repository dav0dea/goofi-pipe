<!-- Zoom out far enough to leave a sub-patch. Must live INSIDE <SvelteFlow> for the hooks to
     resolve, and renders nothing. The threshold is derived live from the content's fit zoom. -->
<script lang="ts">
	import { useViewport, useSvelteFlow, useStore, getViewportForBounds } from '@xyflow/svelte';

	let { entered, onExit }: { entered: string | null; onExit: () => void } = $props();

	const EXIT_RATIO = 0.5; // pop at half the all-node-fit zoom ("50% past fit")
	const FIT_MAX_ZOOM = 1; // must match the editor's FIT_OPTIONS maxZoom
	const FIT_PADDING = 0.18; // must match the editor's FIT_OPTIONS padding

	const vp = useViewport();
	const store = useStore();
	const { getNodesBounds } = useSvelteFlow();
	let armed = false; // plain, not reactive: only a zoom change drives the check

	$effect(() => {
		// Arm shortly after descending, so the enter-fit settles first.
		const cur = entered;
		armed = false;
		if (!cur) return;
		const t = setTimeout(() => {
			armed = true;
		}, 220);
		return () => clearTimeout(t);
	});

	$effect(() => {
		const z = vp.current.zoom;
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
