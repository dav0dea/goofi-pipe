<!-- ScrollArea — the panel-body vertical scroller, wearing the global `.thin-scrollbar` skin. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		children?: Snippet;
	} = $props();
</script>

<div {...rest} class={`ui-scrollarea thin-scrollbar ${klass}`.trim()}>
	{@render children?.()}
</div>

<style>
	.ui-scrollarea {
		/* `min-height: 0` is what lets a flex child scroll instead of pushing the layout. */
		flex: 1 1 auto;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
	}
</style>
