<!--
  ScrollArea — the panel-body vertical scroller (spec §2.3). It OWNS `min-height: 0` so it can
  shrink below its content inside a flex column (the classic "flex child won't scroll" trap) and
  `overflow-y: auto` so the overflow scrolls. `flex: 1 1 auto` makes it fill the remaining column
  space by default — its dominant use — while a consumer can bound it with an explicit height via
  the merged `class`. Snippet `children`, merged `class`, forwarded `data-testid` via `...rest`.

  The scrollbar is thin and matches the app scrollbar palette (F surface/border tokens): a slim
  Firefox `scrollbar-width: thin` plus a WebKit thumb, so it reads as the app's, not the browser's.
-->
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

<div {...rest} class={`ui-scrollarea ${klass}`.trim()}>
	{@render children?.()}
</div>

<style>
	.ui-scrollarea {
		/* Fill the remaining column space and own min-height:0 so overflow scrolls instead of
		   pushing the layout (the reason a flex-child scroller needs this). */
		flex: 1 1 auto;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		/* Thin scrollbar matching the app palette (F tokens), not the browser default. */
		scrollbar-width: thin;
		scrollbar-color: var(--surface-3) transparent;
	}
	.ui-scrollarea::-webkit-scrollbar {
		width: 8px;
	}
	.ui-scrollarea::-webkit-scrollbar-track {
		background: transparent;
	}
	.ui-scrollarea::-webkit-scrollbar-thumb {
		background: var(--surface-3);
		border-radius: var(--radius-md);
	}
	.ui-scrollarea::-webkit-scrollbar-thumb:hover {
		background: var(--border-strong);
	}
</style>
