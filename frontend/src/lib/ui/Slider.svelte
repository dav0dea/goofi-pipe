<!--
  Slider — a dumb range control (spec §2.2): `value` in, `onChange` out. `min`/`max`/`step`, an
  accent track, and the min/max bound labels. It opts into `useLiveValue` so a live backend echo can't
  yank the thumb mid-drag: `editing` is latched on pointer-down and released on pointer-up, and each
  drag step commits live.

  The track auto-extends when the live value lies outside [min, max] (as goofi3 / ParamField do) so a
  value of 5 on a [0, 1] slider renders in range instead of clipping at the edge. The wrapping row is
  the root — `class` merged, `data-testid` (and any other attribute) forwarded via `...rest`; the
  range input claims the enclosing Field's label id so clicking the label focuses it.
-->
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
	// A default step gives the range ~200 stops across the (possibly extended) span.
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
		/* Let vertical scroll gestures pass through on touch while horizontal drags the thumb. */
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
