<script lang="ts">
	import type { Guide } from './snap';

	// The alignment snap-guide overlay, shared by the drag-snap (NodeEditorPanel) and the
	// placement-snap (PlacementPreview) so both render identically. The caller owns the
	// ViewportPortal + the `guides.length > 0` guard (PlacementPreview's portal wraps its
	// whole ghost, not just the guides).
	let { guides, testid }: { guides: Guide[]; testid: string } = $props();
</script>

<svg class="snap-guides" data-testid={testid}>
	{#each guides as gd, i (i)}
		{#if gd.x !== undefined}
			<line x1={gd.x} x2={gd.x} y1={-5000} y2={5000} stroke="var(--accent)" stroke-width="1" stroke-opacity={gd.opacity} />
		{:else if gd.y !== undefined}
			<line x1={-5000} x2={5000} y1={gd.y} y2={gd.y} stroke="var(--accent)" stroke-width="1" stroke-opacity={gd.opacity} />
		{/if}
	{/each}
</svg>

<style>
	.snap-guides {
		position: absolute;
		left: 0;
		top: 0;
		width: 1px;
		height: 1px;
		overflow: visible;
		pointer-events: none;
	}
</style>
