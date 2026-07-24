<!--
  Stack — a vertical flex column (spec §2.3). The `gap` prop is token-valued (a spacing-scale
  key → an F `--space-N`, resolved by the pure `resolveSpace`), and `align`/`justify` are short
  string unions mapped to CSS flexbox. Snippet `children`, merged `class`, forwarded `data-testid`
  (and any other attribute) via `...rest`.

  It OWNS `min-width: 0` AND `min-height: 0` on itself so that, when nested as a flex item, it can
  shrink below its content on both axes — this is the point: it replaces the swarm of defensive
  `min-*: 0` declarations that consumers otherwise sprinkle to make text ellipsis / scroll areas
  behave. Prop values flow through inline CSS vars, so a consumer's own `style`/`class` still merges.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import {
		resolveSpace,
		alignItems,
		justifyContent,
		type SpaceScale,
		type AlignSetting,
		type JustifySetting
	} from './layout';

	let {
		gap = 4,
		align = 'stretch',
		justify = 'start',
		class: klass = '',
		style: styleAttr = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		gap?: SpaceScale | string;
		align?: AlignSetting;
		justify?: JustifySetting;
		children?: Snippet;
	} = $props();

	const vars = $derived(
		`--stack-gap:${resolveSpace(gap)};--stack-align:${alignItems(align)};--stack-justify:${justifyContent(justify)}`
	);
</script>

<div {...rest} class={`ui-stack ${klass}`.trim()} style={`${vars};${styleAttr ?? ''}`}>
	{@render children?.()}
</div>

<style>
	.ui-stack {
		display: flex;
		flex-direction: column;
		/* Own both axes so a nested Stack shrinks under its content (replaces the defensive min-*:0). */
		min-width: 0;
		min-height: 0;
		gap: var(--stack-gap);
		align-items: var(--stack-align);
		justify-content: var(--stack-justify);
	}
</style>
