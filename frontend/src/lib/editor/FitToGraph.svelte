<script lang="ts">
	/**
	 * Fits the viewport to the whole graph after a *wholesale* load (patch open,
	 * example, reconnect to a populated backend) — and never when nodes are added
	 * one at a time.
	 *
	 * This replaces SvelteFlow's declarative `fitView` init prop, which queues a
	 * one-shot fit that stays parked while the graph is empty (`nodesInitialized`
	 * is false with zero nodes) and then fires on the first interactively-placed
	 * node, yanking the camera to center it. Driving the fit off `graph.loadEpoch`
	 * — which bumps only on snapshot replacement — keeps the good "fit on load"
	 * behaviour while leaving interactive placement alone.
	 *
	 * The epoch it has fitted for lives with the panel's camera, not here: a layout
	 * reshape remounts this component, and a fresh -1 would re-fit a graph the user
	 * has since panned away from.
	 *
	 * Must be rendered inside <SvelteFlow>: `useSvelteFlow()` reads the flow
	 * instance from context, which the parent panel's <script> can't see (the
	 * provider is set up by a child component in its template). `fitView()` itself
	 * queues until the new nodes are measured, so we don't wait on that here.
	 */
	import { untrack } from 'svelte';
	import { useSvelteFlow, type FitViewOptions } from '@xyflow/svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { camera } from './camera';

	let { panelId, options }: { panelId: string; options: FitViewOptions } = $props();

	const { fitView } = useSvelteFlow();
	const g = graph();
	// One panel per instance, fixed at mount — a panel id is the component's identity.
	const cam = camera(untrack(() => panelId));

	$effect(() => {
		const epoch = g.loadEpoch;
		if (epoch === cam.fittedEpoch) return;
		cam.fittedEpoch = epoch;
		// An empty load (fresh blank session) has nothing to fit — and crucially
		// must not arm a fit that the first placed node would then satisfy.
		if (g.nodes.length > 0) void fitView(options);
	});
</script>
