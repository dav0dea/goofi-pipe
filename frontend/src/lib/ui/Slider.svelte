<!-- Slider — a dumb range control: `value` in, `onChange` out. The latch is released on pointer-up
     OR pointer-cancel, because a touch pan the UA claims fires cancel and never up. -->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { useLiveValue } from './liveValue.svelte';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		min = 0,
		max = 1,
		step,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		value: number;
		onChange: (v: number) => void;
		min?: number;
		max?: number;
		step?: number;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	const live = useLiveValue<number>(
		() => value,
		(v) => onChange(v)
	);

	// Auto-extend so an out-of-range live value never clips at an edge.
	const lo = $derived(Math.min(min, live.value));
	const hi = $derived(Math.max(max, live.value));
	// A default step gives the range ~200 stops across the span.
	const stp = $derived(step ?? Math.max((hi - lo) / 200, 1e-6));

	function fmtBound(v: number): string {
		if (!Number.isFinite(v)) return '';
		if (Number.isInteger(v)) return String(v);
		return String(Number(v.toFixed(3))); // trim trailing zeros
	}
</script>

<div {...rest} class={`ui-slider ${klass}`.trim()}>
	<span class="ui-slider-bound" aria-hidden="true">{fmtBound(lo)}</span>
	<input
		id={fieldId}
		class="ui-slider-range"
		type="range"
		min={lo}
		max={hi}
		step={stp}
		value={live.value}
		onpointerdown={() => live.begin()}
		onpointerup={() => live.end()}
		onpointercancel={() => live.end()}
		oninput={(e) => live.commit(Number((e.currentTarget as HTMLInputElement).value))}
	/>
	<span class="ui-slider-bound" aria-hidden="true">{fmtBound(hi)}</span>
</div>

<style>
	.ui-slider {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-width: 0;
		flex: 1 1 auto;
	}
	.ui-slider-range {
		flex: 1 1 auto;
		min-width: 0;
		accent-color: var(--accent);
		background: transparent;
		padding: 0;
		border: none;
		/* A vertical touch gesture scrolls; a horizontal one drags the thumb. */
		touch-action: pan-y;
	}
	.ui-slider-bound {
		flex-shrink: 0;
		min-width: 1rem;
		text-align: center;
		color: var(--text-muted);
		font-size: var(--fs-micro);
		font-variant-numeric: tabular-nums;
	}
</style>
