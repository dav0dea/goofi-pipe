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

  `density="chrome"` is the compact face a toolbar strip wears — the same axis `IconButton` spends
  `density` on, and for the same reason: a `Bar` is shorter than a form row, so a dropdown wearing the
  form control's box is the tallest thing in it. It was the viewer-type dropdown's own hardcoded
  `<select class="kind">`, which is why that one looked right and every other bar dropdown did not;
  stated here, one dropdown wears it in every strip and the frozen node-slot header keeps its opt-out
  through `--select-min-h` / `--select-fs`.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'tatami';
	import { IconButton } from 'tatami';
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
		/** Box density. `chrome` is the compact toolbar face; `comfortable` the form-row one. */
		density?: 'comfortable' | 'chrome';
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

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	// Keep the live value selectable even if it isn't among the options (a stale-but-live value). The
	// truthy guard keeps an empty/falsy value from being prepended as a blank leading option.
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
	/* Chrome density: a toolbar dropdown is sized by its own line, not by the form-row box app.css
	   gives every <select>. It also stops filling the row — a strip's controls sit side by side. */
	.ui-select.d-chrome,
	.ui-select.d-chrome .ui-select-input {
		flex: 0 0 auto;
	}
	.ui-select.d-chrome .ui-select-input {
		appearance: none;
		font-size: var(--fs-small);
		/* The value text rode high at this smaller font-size because the browser floors a <select>'s
		   height (UA min-height), leaving the short line box top-aligned in an over-tall control. Pin
		   the content box to exactly the line height (content-box so they match precisely) and release
		   the UA floor, so the single value line is vertically centered at any font-size. */
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
	/* Hover feedback, gated on the device having a hover to give — `:hover` still MATCHES for a
	   synthetic pointer on a phone, so ungated this lit the picker for a state a finger is never in.
	   A hover-CAPABILITY query, not a pointer one: D-R7's coarse idiom below is untouched. */
	@media (hover: hover) {
		.ui-select.d-chrome .ui-select-input:hover {
			color: var(--text);
			border-color: var(--accent);
		}
	}
	/* Touch: this rule out-specifies app.css's `select` floor (0,0,1) on BOTH counts — the
	   `min-height: 0` above releases the --hit floor, and --fs-small is 11.5px at the coarse root
	   size, which iOS force-zooms on focus. Restored here.
	   `--select-*` is the frozen host's opt-out, the same seam IconButton spends `--icon-btn-size`
	   on: a strip that is shorter than --hit BY CONSTRUCTION states the compact box, and every other
	   host takes the floor. Exactly one host does — SlotViewer's 24px `--node-u` slot header, which
	   is frozen canvas geometry; every docked panel bar is real chrome and takes it. */
	@media (hover: none) and (pointer: coarse) {
		.ui-select.d-chrome .ui-select-input {
			/* Less the 1px border on each side, because the box above is a CONTENT box: what has to
			   measure --hit is the rendered control, and a chrome strip is exactly --panel-header-h
			   tall under a coarse pointer — which IS --hit, so there is no room for a border outside
			   the floor. Floored at --hit flat, this control was 46px in a 44px bar and every panel
			   toolbar stood 2px taller than the panel header it matches. The subtrahend is this
			   element's own border (app.css: `input, select, textarea`), not a spacing rung; if it
			   ever moves, `touch-panel-bar.spec.ts` is what says so. */
			min-height: var(--select-min-h, calc(var(--hit) - 2px));
			font-size: var(--select-fs, 16px);
		}
	}
	/* The ⟳ icon is swapped for a CSS ring while spinning — a bare circle rotates dead-centred
	   where an icon's own asymmetry wobbles (same ring as the node boot-spinner). */
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
