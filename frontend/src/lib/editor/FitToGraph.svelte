<script lang="ts">
	/**
	 * Fits the viewport to the whole graph after a wholesale load, never on an interactive add.
	 * Must be rendered INSIDE <SvelteFlow>: `useSvelteFlow()` reads the flow from context.
	 */
	import { untrack } from 'svelte';
	import { useSvelteFlow, type FitViewOptions } from '@xyflow/svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { camera } from './camera';

	let { panelId, options }: { panelId: string; options: FitViewOptions } = $props();

	const { fitView } = useSvelteFlow();
	const g = graph();
	const cam = camera(untrack(() => panelId));

	$effect(() => {
		const epoch = g.loadEpoch;
		if (epoch === cam.fittedEpoch) return;
		cam.fittedEpoch = epoch;
		// An empty load must not arm a fit that the first placed node would then satisfy.
		if (g.nodes.length > 0) void fitView(options);
	});
</script>
