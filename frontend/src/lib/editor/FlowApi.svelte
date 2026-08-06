<script lang="ts">
	/**
	 * Bridges the SvelteFlow imperative API up to the parent panel's <script>,
	 * which can't call useSvelteFlow() itself — the flow context is provided by a
	 * child in the panel's template, below the panel script. Render this inside
	 * <SvelteFlow> and bind the functions you need (report B9: paste needs
	 * screenToFlowPosition to anchor in flow space, not screen space).
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
		/** The pan/zoom matrix, and the way to write it — the double-tap zoom drives the viewport
		 *  through SvelteFlow's own helpers rather than writing the transform behind its back. */
		getViewport?: () => FlowViewport;
		setViewport?: (v: FlowViewport) => void;
	} = $props();

	const flow = useSvelteFlow();
	screenToFlowPosition = flow.screenToFlowPosition;
	getViewport = flow.getViewport;
	setViewport = flow.setViewport;
</script>
