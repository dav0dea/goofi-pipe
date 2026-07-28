<!--
  Trigger — the action-button param control (spec §2.2): a value-less "do it now" button (the goofi
  `trigger` param). Quiet by default so it sits in the same visual register as the other controls,
  warming to the accent on hover and filling accent on press. Dumb: it forwards `onclick` (and every
  other button attribute) via `...rest`; N wires it to fire the param. `children` is the label.

  Fully self-styled from F tokens (like `Button`) so it renders correctly independent of the app.css
  base `button` rule — which M strips last. `class` merged, `data-testid` forwarded.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	let {
		type = 'button',
		class: klass = '',
		children,
		...rest
	}: HTMLButtonAttributes & {
		children?: Snippet;
	} = $props();
</script>

<button {...rest} {type} class={`ui-trigger ${klass}`.trim()}>
	{@render children?.()}
</button>

<style>
	.ui-trigger {
		font-family: var(--font-mono);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		min-height: var(--hit);
		padding: var(--space-3) var(--space-6);
		font-size: var(--fs-small);
		background: var(--surface-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		color: var(--text);
		letter-spacing: 0.02em;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			border-color var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-trigger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 14%, var(--surface-2));
		border-color: color-mix(in srgb, var(--accent) 55%, var(--border-strong));
	}
	/* Press fills accent — the tactile "fired" confirmation. */
	.ui-trigger:active:not(:disabled) {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--on-accent);
	}
	.ui-trigger:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
</style>
