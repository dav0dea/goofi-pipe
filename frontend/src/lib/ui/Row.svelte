<!--
  Row — a horizontal flex row (spec §2.3). Sibling of Stack: the same token-valued `gap` and the
  same `align`/`justify` short unions, mapped to CSS flexbox. Snippet `children`, merged `class`,
  forwarded `data-testid` (and any other attribute) via `...rest`.

  It OWNS `min-width: 0` on itself so that, when nested as a flex item, it can shrink below its
  content along the main axis — the point of the primitive (it replaces the defensive `min-width:0`
  consumers otherwise add so a child's text can ellipsis instead of forcing overflow). Prop values
  flow through inline CSS vars, so a consumer's own `style`/`class` still merges. `align` defaults to
  `center` (the common row baseline — vertically centred items).
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
		align = 'center',
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
		`--row-gap:${resolveSpace(gap)};--row-align:${alignItems(align)};--row-justify:${justifyContent(justify)}`
	);
</script>

<div {...rest} class={`ui-row ${klass}`.trim()} style={`${vars};${styleAttr ?? ''}`}>
	{@render children?.()}
</div>

<style>
	.ui-row {
		display: flex;
		flex-direction: row;
		/* Own the main axis so a nested Row shrinks under its content (replaces the defensive min-width:0). */
		min-width: 0;
		gap: var(--row-gap);
		align-items: var(--row-align);
		justify-content: var(--row-justify);
	}
</style>
