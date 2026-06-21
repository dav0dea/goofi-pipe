<script lang="ts">
	/**
	 * Bridges the SvelteFlow imperative API up to the parent panel's <script>,
	 * which can't call useSvelteFlow() itself — the flow context is provided by a
	 * child in the panel's template, below the panel script. Render this inside
	 * <SvelteFlow> and bind the functions you need (report B9: paste needs
	 * screenToFlowPosition to anchor in flow space, not screen space).
	 */
	import { useSvelteFlow } from '@xyflow/svelte';

	type ScreenToFlow = (p: { x: number; y: number }) => { x: number; y: number };

	let { screenToFlowPosition = $bindable() }: { screenToFlowPosition?: ScreenToFlow } = $props();

	const flow = useSvelteFlow();
	screenToFlowPosition = flow.screenToFlowPosition;
</script>
