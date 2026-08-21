<!-- Select — a dumb dropdown: `value` in, `onChange` out, plus an optional ⟳ re-scan. A native
     <select> commits on `change`, so it needs no `useLiveValue` latch. -->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'panelty';
	import { IconButton } from 'panelty';
		import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		options,
		labels,
		onRefresh,
		refreshing = false,
		refreshTestid,
		density = 'comfortable',
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		value: string;
		onChange: (v: string) => void;
		options: string[];
		/** Box density; `chrome` is the compact toolbar face. */
		density?: 'comfortable' | 'chrome';
		/** Display text per option; the committed value stays the raw option key. */
		labels?: Record<string, string>;
		/** Provide to show the ⟳ re-scan button; kept even when `options` is empty. */
		onRefresh?: () => void;
		/** A ⟳ re-scan is in flight. */
		refreshing?: boolean;
		/** `data-testid` stamped on the ⟳ button. */
		refreshTestid?: string;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	// A stale-but-live value stays selectable; the truthy guard keeps a blank one out of the list.
	const items = $derived(!value || options.includes(value) ? options : [value, ...options]);
</script>

<div {...rest} class={`ui-select ${density === 'chrome' ? 'd-chrome ' : ''}${klass}`.trim()}>
	<select
		id={fieldId}
		class="ui-select-input"
		{value}
		disabled={refreshing}
		onchange={(e) => onChange((e.currentTarget as HTMLSelectElement).value)}
	>
		{#each items as opt (opt)}
			<option value={opt}>{labels?.[opt] ?? opt}</option>
		{/each}
	</select>
	{#if onRefresh}
		<IconButton
			size="sm"
			label={refreshing ? 'Re-scanning…' : 'Re-scan for options'}
			disabled={refreshing}
			aria-busy={refreshing}
			data-testid={refreshTestid}
			onclick={() => onRefresh()}
		>
			{#if refreshing}
				<span class="ui-select-spinner" aria-hidden="true"></span>
			{:else}
				<Icon name="refresh-cw" />
			{/if}
		</IconButton>
	{/if}
</div>

<style>
	.ui-select {
		display: flex;
		align-items: stretch;
		gap: var(--space-2);
		flex: 1 1 auto;
		min-width: 0;
	}
	.ui-select-input {
		flex: 1 1 auto;
		min-width: 0;
		color: var(--text);
	}
	.ui-select-input:disabled {
		opacity: var(--disabled-opacity);
		cursor: default;
	}
	.ui-select.d-chrome,
	.ui-select.d-chrome .ui-select-input {
		flex: 0 0 auto;
	}
	.ui-select.d-chrome .ui-select-input {
		appearance: none;
		font-size: var(--fs-small);
		/* The content box is pinned to the line height and the UA's <select> height floor released,
		   or the value text rides high in an over-tall control. */
		box-sizing: content-box;
		height: 1.5em;
		line-height: 1.5em;
		min-height: 0;
		text-align: center;
		text-align-last: center;
		color: var(--text-dim);
		background: color-mix(in srgb, var(--bg) 55%, transparent);
		border-radius: var(--radius-sm);
		padding: 0 var(--space-2);
		cursor: pointer;
	}
	/* Gated on a real hover: `:hover` still matches for a phone's synthetic pointer. */
	@media (hover: hover) {
		.ui-select.d-chrome .ui-select-input:hover {
			color: var(--text);
			border-color: var(--accent);
		}
	}
	/* Touch: the rules above release app.css's --hit floor and drop below the 16px iOS force-zooms
	   under, so both are restored here; `--select-*` is the frozen slot header's opt-out. */
	@media (hover: none) and (pointer: coarse) {
		.ui-select.d-chrome .ui-select-input {
			/* Less this element's own 1px border per side: the box above is a CONTENT box, and it is
			   the rendered control that must measure --hit. */
			min-height: var(--select-min-h, calc(var(--hit) - 2px));
			font-size: var(--select-fs, 16px);
		}
	}
	/* A bare ring, not the ⟳ icon: a circle rotates dead-centred where the glyph wobbles. */
	.ui-select-spinner {
		width: 0.85em;
		height: 0.85em;
		border-radius: 50%;
		box-sizing: border-box;
		border: 1.5px solid color-mix(in srgb, var(--accent) 40%, transparent);
		border-top-color: var(--accent);
		animation: ui-select-spin 0.8s linear infinite;
	}
	@keyframes ui-select-spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
