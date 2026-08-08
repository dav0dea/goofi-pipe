<!--
  Icon — the app's ONE icon renderer. Takes a name from the vendored Lucide table (`icons.ts`) and
  draws it in Lucide's own frame: a 24-box, a 2px stroke with round caps and joins, no fill.

  It paints in `currentColor` and sizes at `1em`, so it inherits the colour AND the size of whatever
  control it sits in — exactly as the text glyph it replaced did, which is why a consumer needs no
  rule of its own. A control that wants a different weight moves its own `font-size`.

  `aria-hidden` by default: an icon is decoration beside a label, or the visible content of an
  `IconButton` whose `label` prop is already the accessible name. It is spelled before `...rest` so
  a caller that really does mean the icon to be announced can say so.

  `{@html}` is safe by construction here — `name` is a `keyof typeof ICONS`, so the markup is always
  one of the vendored constants, and `icons.test.ts` holds those to self-closing drawing tags with
  no attributes beyond geometry.
-->
<script lang="ts">
	import type { SVGAttributes } from 'svelte/elements';
	import { ICONS, type IconName } from './icons';

	let {
		name,
		class: klass = '',
		...rest
	}: SVGAttributes<SVGSVGElement> & { name: IconName } = $props();
</script>

<svg
	xmlns="http://www.w3.org/2000/svg"
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width="2"
	stroke-linecap="round"
	stroke-linejoin="round"
	aria-hidden="true"
	focusable="false"
	{...rest}
	class={`ui-icon ${klass}`.trim()}
	data-icon={name}>{@html ICONS[name]}</svg
>

<style>
	.ui-icon {
		display: inline-block;
		vertical-align: middle;
		width: 1em;
		height: 1em;
		/* Never squeezed by a flex row it happens to sit in — an icon is its own size or it is a
		   different icon. */
		flex: 0 0 auto;
	}
</style>
