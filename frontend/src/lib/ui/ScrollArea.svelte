<!--
  ScrollArea — the panel-body vertical scroller (spec §2.3). It OWNS `min-height: 0` so it can
  shrink below its content inside a flex column (the classic "flex child won't scroll" trap) and
  `overflow-y: auto` so the overflow scrolls. `flex: 1 1 auto` makes it fill the remaining column
  space by default — its dominant use — while a consumer can bound it with an explicit height via
  the merged `class`. Snippet `children`, merged `class`, forwarded `data-testid` via `...rest`.

  The scrollbar skin is the global `.thin-scrollbar` (app.css) — shared with the one scroller that
  cannot be this component (ConsolePanel's virtual list owns its own scrollTop handle), so the app
  has one slim scrollbar, not a copy per scroller.
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

<div {...rest} class={`ui-scrollarea thin-scrollbar ${klass}`.trim()}>
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
	}
</style>
