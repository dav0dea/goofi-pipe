<!--
  Select — a dumb dropdown (spec §2.2): `value` in, `onChange` out, plus an optional refresh (⟳)
  affordance for device / stream pickers. A native <select> commits on `change` and can't echo-jump
  mid-interaction, so it needs no `useLiveValue` latch. A truthy current value is kept in the list
  (prepended when the options don't contain it) so a stale-but-live value still renders — but an
  EMPTY/falsy value is NOT prepended, so it never shows as a blank leading option (the empty-value
  rule N's deleted `selectOptions` helper used to own, now enforced at this one P source of truth).

  The ⟳ appears only when `onRefresh` is given — even with an EMPTY option list, so a scan that found
  nothing is still re-runnable (the device/stream re-scan). It is the `IconButton` primitive; while a
  refresh is in flight the <select> dims and the button spins a CSS ring. Pass `refreshTestid` to stamp
  a `data-testid` on that ⟳ button (a consumer delegating its own refresh affordance keeps its testid).
  The <select> claims the enclosing Field's label id. `class` merged, `data-testid` (and any other
  attribute) forwarded onto the wrapper.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import IconButton from './IconButton.svelte';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		options,
		labels,
		onRefresh,
		refreshing = false,
		refreshTestid,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		value: string;
		onChange: (v: string) => void;
		options: string[];
		/** Optional display text per option value (`labels[opt] ?? opt`) — the committed value stays the
		 * raw option key while the dropdown shows a friendlier label (e.g. a port's user name + dtype). */
		labels?: Record<string, string>;
		/** Provide to show the ⟳ re-scan button (kept even when `options` is empty). */
		onRefresh?: () => void;
		/** A ⟳ re-scan is in flight — dim the select and spin the button. */
		refreshing?: boolean;
		/** `data-testid` stamped on the ⟳ button (undefined → no attribute). */
		refreshTestid?: string;
	} = $props();

	const fieldId = claimFieldControlId();
	// Keep the live value selectable even if it isn't among the options (a stale-but-live value). The
	// truthy guard keeps an empty/falsy value from being prepended as a blank leading option.
	const items = $derived(!value || options.includes(value) ? options : [value, ...options]);
</script>

<div {...rest} class={`ui-select ${klass}`.trim()}>
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
				⟳
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
	/* Inherits the app-wide select chrome + coarse --hit floor; just fill the row. */
	.ui-select-input {
		flex: 1 1 auto;
		min-width: 0;
		color: var(--text);
	}
	.ui-select-input:disabled {
		opacity: var(--disabled-opacity);
		cursor: default;
	}
	/* The ⟳ glyph is swapped for a CSS ring while spinning — a geometric circle rotates dead-centred
	   where a text glyph pivots off its baseline (same ring as the node boot-spinner). */
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
