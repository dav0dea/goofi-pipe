<!-- Disclosure — one collapse control: a caret plus a `summary` that toggles `children` in and
     out of the DOM. `open` is bindable and also reports through `onToggle`. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'panelty';

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
		<span class="ui-disclosure-caret" class:open><Icon name="chevron-right" /></span>
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
		font-family: var(--font-sans);
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
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -2px;
	}
	.ui-disclosure-label {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.ui-disclosure-caret {
		flex-shrink: 0;
		display: flex;
		align-items: center;
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
