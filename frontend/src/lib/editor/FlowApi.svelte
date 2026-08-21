<script lang="ts">
	/**
	 * Bridges the SvelteFlow imperative API up to the parent panel's <script>, which cannot call
	 * useSvelteFlow() itself. Render this inside <SvelteFlow> and bind the functions you need.
	 */
	import { useSvelteFlow } from '@xyflow/svelte';
	import type { FlowViewport } from './doubleTapZoom';

	type ScreenToFlow = (p: { x: number; y: number }) => { x: number; y: number };

	let {
		screenToFlowPosition = $bindable(),
		getViewport = $bindable(),
		setViewport = $bindable()
	}: {
		screenToFlowPosition?: ScreenToFlow;
		/** The pan/zoom matrix and the way to write it, so no caller writes the transform directly. */
		getViewport?: () => FlowViewport;
		setViewport?: (v: FlowViewport) => void;
	} = $props();

	const flow = useSvelteFlow();
	screenToFlowPosition = flow.screenToFlowPosition;
	getViewport = flow.getViewport;
	setViewport = flow.setViewport;
</script>
