<!--
  Disclosure — one collapse control (spec §2.5): a caret + a `summary` snippet that toggles the
  `children` in and out of the DOM. The WAI-ARIA disclosure pattern — a real `<button>` whose
  `aria-expanded` mirrors the open state and `aria-controls` names the region it reveals, so the
  affordance is never behind `:hover` alone (it is a visible, keyboard-operable button).

  `open` is `$bindable` (use `bind:open`) AND reports through `onToggle(next)` — a controlled parent
  and a bound one both work. The caret rotates on open; the global `prefers-reduced-motion` guard (F,
  app.css) neutralises that transition, so no per-component guard is needed. `class` merged,
  `data-testid` (and any other attribute) forwarded via `...rest`.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		open = $bindable(false),
		onToggle,
		summary,
		children,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		open?: boolean;
		onToggle?: (open: boolean) => void;
		summary: Snippet;
		children?: Snippet;
	} = $props();

	const bodyId = $props.id();

	function toggle(): void {
		open = !open;
		onToggle?.(open);
	}
</script>

<div {...rest} class={`ui-disclosure ${klass}`.trim()}>
	<button
		type="button"
		class="ui-disclosure-summary"
		aria-expanded={open}
		aria-controls={bodyId}
		onclick={toggle}
	>
		<span class="ui-disclosure-caret" class:open aria-hidden="true">▶</span>
		<span class="ui-disclosure-label">{@render summary()}</span>
	</button>
	{#if open}
		<div id={bodyId} class="ui-disclosure-body">{@render children?.()}</div>
	{/if}
</div>

<style>
	.ui-disclosure {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	/* The summary is a full-width, self-styled button (does not lean on the app.css base rule) so its
	   caret + label read as one clickable header. */
	.ui-disclosure-summary {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		width: 100%;
		min-height: var(--hit);
		padding: var(--space-2) var(--space-3);
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		font-weight: 600;
		text-align: left;
		cursor: pointer;
		transition: background var(--dur-fast) var(--ease);
	}
	.ui-disclosure-summary:hover {
		background: var(--surface-2);
	}
	.ui-disclosure-summary:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: -2px;
	}
	.ui-disclosure-label {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The caret points right when closed, rotates down when open. Reduced-motion is handled globally. */
	.ui-disclosure-caret {
		flex-shrink: 0;
		display: inline-block;
		font-size: var(--fs-micro);
		color: var(--text-muted);
		transition: transform var(--dur-slow) var(--ease);
	}
	.ui-disclosure-caret.open {
		transform: rotate(90deg);
	}
	.ui-disclosure-body {
		min-width: 0;
		padding: var(--space-3) var(--space-3) var(--space-3) var(--space-7);
	}
</style>
