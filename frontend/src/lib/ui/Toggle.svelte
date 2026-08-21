<!-- Toggle — the on/off switch: a transparent native checkbox, so focus and the focus ring stay
     real, behind a painted track and knob. -->
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
	/* The track is the visible surrogate for the transparent checkbox, so it takes the ring. */
	.ui-toggle-input:focus-visible ~ .ui-toggle-track {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: 2px;
	}
	/* The hit rect grows outward, never the painted track: the knob's translate is tied to its
	   2.4rem width. */
	@media (hover: none) and (pointer: coarse) {
		.ui-toggle::after {
			content: '';
			position: absolute;
			inset: calc((var(--hit) - 100%) / -2);
		}
	}
</style>
