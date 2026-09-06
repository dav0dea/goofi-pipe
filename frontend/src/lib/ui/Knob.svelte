<!-- Knob — a dumb rotary control: `value` in, `onChange` out. A vertical drag turns it, and the
     arrow keys step it, so it is reachable without a pointer. The latch is released on pointer-up
     OR pointer-cancel, because a touch pan the UA claims fires cancel and never up. -->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { turnedBy } from './knob';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		min = 0,
		max = 1,
		step = 0,
		disabled = false,
		label,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		value: number;
		onChange: (v: number) => void;
		min?: number;
		max?: number;
		step?: number;
		disabled?: boolean;
		label?: string;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);

	const span = $derived(max - min || 1);
	const fraction = $derived(Math.min(1, Math.max(0, (value - min) / span)));
	// A knob sweeps 270°, from the lower left round to the lower right.
	const angle = $derived(-135 + fraction * 270);
	const stp = $derived(step > 0 ? step : span / 200);

	let from: { y: number; value: number } | null = $state(null);

	function down(e: PointerEvent): void {
		if (disabled) return;
		from = { y: e.clientY, value };
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		e.preventDefault();
	}

	function move(e: PointerEvent): void {
		if (!from) return;
		onChange(turnedBy(from.value, e.clientY - from.y, min, max, stp));
	}

	function up(): void {
		from = null;
	}

	function key(e: KeyboardEvent): void {
		if (disabled) return;
		const by = e.key === 'ArrowUp' || e.key === 'ArrowRight' ? 1 : e.key === 'ArrowDown' || e.key === 'ArrowLeft' ? -1 : 0;
		if (by === 0) return;
		e.preventDefault();
		onChange(turnedBy(value + by * stp, 0, min, max, stp));
	}
</script>

<div {...rest} class={`ui-knob ${klass}`.trim()} class:disabled>
	<div
		id={fieldId}
		class="ui-knob-dial"
		role="slider"
		tabindex={disabled ? -1 : 0}
		aria-label={label}
		aria-valuemin={min}
		aria-valuemax={max}
		aria-valuenow={value}
		aria-disabled={disabled}
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={up}
		onkeydown={key}
	>
		<svg viewBox="0 0 40 40" aria-hidden="true">
			<circle class="ui-knob-track" cx="20" cy="20" r="16" />
			<line
				class="ui-knob-pointer"
				x1="20"
				y1="20"
				x2={20 + 13 * Math.sin((angle * Math.PI) / 180)}
				y2={20 - 13 * Math.cos((angle * Math.PI) / 180)}
			/>
		</svg>
	</div>
</div>

<style>
	/* The dial fills the box it is given and stays round, and is never smaller than a tap target. */
	.ui-knob {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		container-type: size;
	}
	.ui-knob-dial {
		width: max(var(--hit), min(100cqw, 100cqh));
		aspect-ratio: 1;
		touch-action: none;
		cursor: ns-resize;
		border-radius: 50%;
	}
	.ui-knob-dial:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: 2px;
	}
	.ui-knob.disabled .ui-knob-dial {
		cursor: default;
		opacity: 0.5;
	}
	svg {
		display: block;
		width: 100%;
		height: 100%;
	}
	.ui-knob-track {
		fill: var(--surface-2);
		stroke: var(--border);
		stroke-width: 2;
	}
	.ui-knob-pointer {
		stroke: var(--accent);
		stroke-width: 3;
		stroke-linecap: round;
	}
</style>
