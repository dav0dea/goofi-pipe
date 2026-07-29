<!--
  Toggle — the on/off switch control (spec §2.1/§2.2): `value` in, `onChange` out. A native
  `<input type=checkbox>` (visually hidden, so keyboard focus + the app :focus-visible ring still
  work) behind a track + knob painted from F tokens. A checkbox commits atomically on change and
  can't echo-jump mid-interaction, so it needs no `useLiveValue` latch.

  The root is a real `<label>` wrapping the checkbox (its own switch label); it also claims an
  enclosing `<Field>`'s label id so the Field label toggles it too. `class` merged, `data-testid`
  (and any other attribute) forwarded via `...rest`.
-->
<script lang="ts">
	import type { HTMLLabelAttributes } from 'svelte/elements';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		class: klass = '',
		...rest
	}: Omit<HTMLLabelAttributes, 'onchange'> & {
		value: boolean;
		onChange: (v: boolean) => void;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
</script>

<label {...rest} class={`ui-toggle ${klass}`.trim()}>
	<input
		id={fieldId}
		class="ui-toggle-input"
		type="checkbox"
		checked={value}
		onchange={(e) => onChange((e.currentTarget as HTMLInputElement).checked)}
	/>
	<span class="ui-toggle-track" aria-hidden="true"></span>
</label>

<style>
	.ui-toggle {
		position: relative;
		display: inline-block;
		flex-shrink: 0;
		width: 2.4rem;
		height: var(--hit);
	}
	/* The real control: full-box, transparent — it carries focus + the app focus ring, and its
	   :checked state drives the track/knob below. */
	.ui-toggle-input {
		position: absolute;
		inset: 0;
		margin: 0;
		padding: 0;
		width: 100%;
		height: 100%;
		opacity: 0;
		cursor: pointer;
	}
	.ui-toggle-track {
		position: absolute;
		top: 50%;
		left: 0;
		width: 100%;
		height: 1.35rem;
		transform: translateY(-50%);
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: 999px;
		transition: background var(--dur-fast) var(--ease);
		pointer-events: none;
	}
	/* The knob. */
	.ui-toggle-track::before {
		content: '';
		position: absolute;
		top: 50%;
		left: 0.15rem;
		width: 1rem;
		height: 1rem;
		transform: translateY(-50%);
		border-radius: 50%;
		background: var(--text-dim);
		transition:
			transform var(--dur-slow) var(--ease),
			background var(--dur-fast) var(--ease);
	}
	.ui-toggle-input:checked ~ .ui-toggle-track {
		background: color-mix(in srgb, var(--accent) 35%, transparent);
		border-color: var(--accent);
	}
	.ui-toggle-input:checked ~ .ui-toggle-track::before {
		transform: translate(calc(2.4rem - 1.3rem), -50%);
		background: var(--accent);
	}
	/* Keyboard focus rings the track (the visible surrogate for the hidden checkbox). */
	.ui-toggle-input:focus-visible ~ .ui-toggle-track {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	/* Coarse-pointer hit-rect guarantee (mirrors IconButton, down to the two-clause gate D-R7
	   standardises on): extend the clickable area outward to at least --hit WITHOUT widening the
	   painted 2.4rem track (the knob's checked translate is tied to that width, so growing the box
	   would distort the switch). A no-op under a fine pointer. */
	@media (hover: none) and (pointer: coarse) {
		.ui-toggle::after {
			content: '';
			position: absolute;
			inset: calc((var(--hit) - 100%) / -2);
		}
	}
</style>
