<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { onMount, onDestroy, untrack } from 'svelte';
	import uPlot from 'uplot';
	import 'uplot/dist/uPlot.min.css';
	import { decimateMinMax } from './decimate';
	import { readEnvelope, envelopeBand } from './envelope';
	import { formatTick as fmtTick } from './format';
	import { SERIES } from './palette';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	// Settings come from the slot's cog menu (persisted per node+slot). Reading
	// them as derived values means the rebuild effect re-makes the plot when any
	// of them change.
	const logX = $derived(Boolean(settings.logX));
	const logY = $derived(Boolean(settings.logY));
	const yAuto = $derived(settings.yAuto !== false);
	const yMin = $derived(Number(settings.yMin ?? -1));
	const yMax = $derived(Number(settings.yMax ?? 1));
	const showPoints = $derived(Boolean(settings.points));

	// Plain mirrors of the plot-shaping settings (seeded with the schema defaults;
	// the settings $effect refreshes them from the live values before any data is
	// drawn). The data path (makePlot / indexAxis) reads ONLY these, never the
	// reactive $derived above — so pushing a frame never makes the frame effect
	// depend on a setting, which would force a redundant rebuild on every change.
	let mLogX = false;
	let mLogY = false;
	let mYAuto = true;
	let mYMin = -1;
	let mYMax = 1;
	let mPoints = false;

	let container: HTMLDivElement | null = $state(null);
	let plot: uPlot | null = null;
	let resizer: ResizeObserver | null = null;
	let lastNSeries = 0;

	// A size-one array (a single scalar) can't be drawn as a line. Rather than a
	// misleading scrolling history (which reads as a 1-D array), we draw it as a
	// vertical bar at x = value on a locked [0,1] y-axis — it slides left/right as
	// the value changes, unmistakably distinct from an array's line plot. The
	// value (x) axis uses the cog Y-range when set, else a self-calibrating span
	// from the values seen so far. `plotIsScalar` tracks the plot's current build
	// so we only rebuild when switching between the two modes.
	let scalarMode = false;
	let plotIsScalar = false;
	let lastScalar = 0;
	let scalarMin = Infinity;
	let scalarMax = -Infinity;

	// Reused x-index array, rebuilt only when the sample count (or base) changes.
	// uPlot keeps a reference to its data, so a stable index avoids allocating a
	// fresh array every frame. `base` is the sample-index origin chosen by the
	// caller (see pushData).
	let xsCache: Float64Array | null = null;
	let xsLen = -1;
	let xsBase = -1;
	function indexAxis(m: number, base: number): Float64Array {
		if (xsLen !== m || xsBase !== base || !xsCache) {
			const a = new Float64Array(m);
			for (let i = 0; i < m; i++) a[i] = i + base;
			xsCache = a;
			xsLen = m;
			xsBase = base;
		}
		return xsCache;
	}

	// Cursor hover values, updated by the uPlot setCursor hook. Rendered
	// as a top-right floating chip. `null` index = mouse not over plot.
	let cursorIdx = $state<number | null>(null);
	let cursorValues = $state<(number | null)[]>([]);
	let cursorXValue = $state<number | null>(null);

	function buildSeries(nSeries: number): uPlot.Series[] {
		const out: uPlot.Series[] = [{ label: 'x' }];
		for (let i = 0; i < nSeries; i++) {
			out.push({
				label: `c${i}`,
				stroke: SERIES[i % SERIES.length],
				width: 1,
				points: { show: mPoints, size: 4 }
			});
		}
		return out;
	}

	/** uPlot draw hook: paint corner tick labels INSIDE the plot bbox so the
	 * canvas can fill the entire viewer body without sacrificing axis info.
	 * Anchors min/max for x and y at the canvas corners; tiny font, dim
	 * colour — readable but unobtrusive. */
	function drawCornerTicks(u: uPlot): void {
		const ctx = u.ctx;
		// uPlot doesn't pre-scale its ctx, so canvas coords are device pixels.
		// Multiply font + padding by devicePixelRatio for crisp text.
		const r = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
		const xMin = u.scales.x.min ?? 0;
		const xMax = u.scales.x.max ?? 1;
		const yMin = u.scales.y.min ?? 0;
		const yMax = u.scales.y.max ?? 1;
		const left = u.bbox.left;
		const top = u.bbox.top;
		const right = u.bbox.left + u.bbox.width;
		const bottom = u.bbox.top + u.bbox.height;

		ctx.save();
		ctx.font = `${10 * r}px "JetBrains Mono", ui-monospace, monospace`;
		ctx.fillStyle = 'rgba(208, 208, 208, 0.55)';
		const pad = 4 * r;

		if (scalarMode) {
			// Only the value (x) range is meaningful — y is the locked [0,1] bar
			// height. Label the value span at the bottom corners.
			ctx.textBaseline = 'bottom';
			ctx.textAlign = 'left';
			ctx.fillText(fmtTick(xMin), left + pad, bottom - pad);
			ctx.textAlign = 'right';
			ctx.fillText(fmtTick(xMax), right - pad, bottom - pad);
			ctx.restore();
			return;
		}

		// y-axis: max at top-left, min at bottom-left
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(fmtTick(yMax), left + pad, top + pad);
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(yMin), left + pad, bottom - pad);

		// x-axis: min and max on the bottom, right-justified for max
		ctx.textAlign = 'right';
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(xMax), right - pad, bottom - pad);

		ctx.restore();
	}

	function makePlot(width: number, height: number, nSeries: number): void {
		plot?.destroy();
		lastNSeries = nSeries;
		plotIsScalar = scalarMode;
		// `size: 0` reclaims the axis margin so the canvas plot area fills
		// the viewer body. Grid stays via a thin transparent grid stroke.
		// Tick text is painted inside the canvas via the `draw` hook.
		const noMarginAxis: uPlot.Axis = {
			show: true,
			size: 0,
			gap: 0,
			stroke: 'transparent',
			ticks: { show: false },
			grid: { show: true, stroke: 'rgba(255,255,255,0.05)' },
			values: () => []
		};
		// uPlot.Scale.distr: 1 = linear, 3 = log10. The x index is 1-based under
		// log-x (see indexAxis) so no x is <= 0; on log-y, uPlot simply skips any
		// non-positive samples (gaps) rather than collapsing the scale.
		// Scalar mode flips the axes' roles: the single value lives on x (so the
		// bar slides), and y is locked to [0,1] just to draw a full-height bar.
		const scales: uPlot.Options['scales'] = scalarMode
			? {
					x: { time: false, range: () => scalarXRange() },
					y: { auto: false, range: [0, 1] }
				}
			: {
					x: { time: false, distr: mLogX ? 3 : 1 },
					// Manual Y range (from the cog menu) pins the scale; otherwise
					// uPlot auto-fits to the data.
					y: mYAuto
						? { auto: true, distr: mLogY ? 3 : 1 }
						: { auto: false, distr: mLogY ? 3 : 1, range: [mYMin, mYMax] }
				};
		const series: uPlot.Series[] = scalarMode
			? [{ label: 'x' }, { label: 'v', stroke: SERIES[0], width: 2, points: { show: false } }]
			: buildSeries(nSeries);
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [2, 2, 2, 2],
			series,
			axes: [noMarginAxis, noMarginAxis],
			scales,
			cursor: {
				show: true,
				drag: { x: false, y: false, setScale: false },
				points: { show: true, size: 5 },
				// The node is CSS-scaled (and re-positioned) by the SvelteFlow zoom,
				// which uPlot's cached over-rect doesn't track — so the crosshair
				// drifts. Recompute the position straight from the live pointer event
				// and the current rect, dividing the scale back out, and ignore the
				// value uPlot derived from its stale rect.
				//
				// Crucially this hook also decides when the crosshair HIDES. We can't
				// lean on uPlot's own mouseleave here: it doesn't fire reliably for the
				// CSS-scaled over element, and the live data stream re-commits the
				// cursor on every frame (uPlot repaints it whenever `cursor.left >= 0`),
				// which together strand a stale crosshair at the last position after the
				// pointer leaves. So we gate on our own `pointerInside` tracking and
				// return uPlot's hidden sentinel (negative coords) whenever the pointer
				// isn't over the viewer — or when uPlot itself signals a hide.
				move: (u, left, top) => {
					const e = lastMove;
					if (!pointerInside || left < 0 || top < 0 || !e) return [-10, -10];
					const r = u.over.getBoundingClientRect();
					const sx = u.over.offsetWidth ? r.width / u.over.offsetWidth : 1;
					const sy = u.over.offsetHeight ? r.height / u.over.offsetHeight : 1;
					return [(e.clientX - r.left) / sx, (e.clientY - r.top) / sy];
				}
			},
			legend: { show: false },
			hooks: {
				draw: [drawCornerTicks],
				setCursor: [
					(u) => {
						const idx = u.cursor.idx ?? null;
						if (idx == null) {
							cursorIdx = null;
							cursorXValue = null;
							cursorValues = [];
							return;
						}
						cursorIdx = idx;
						cursorXValue = (u.data[0] as ArrayLike<number>)[idx] ?? null;
						const vals: (number | null)[] = [];
						for (let s = 1; s < u.data.length; s++) {
							const arr = u.data[s] as ArrayLike<number> | null;
							vals.push(arr ? arr[idx] ?? null : null);
						}
						cursorValues = vals;
					}
				]
			}
		};
		if (!container) return;
		plot = new uPlot(opts, scalarMode ? [[], []] : [[]], container);
	}

	/** Push aligned data, rebuilding the plot if the mode or series count changed. */
	function setSeries(xs: ArrayLike<number>, ySeries: ArrayLike<number>[]): void {
		if (!plot || !container) return;
		if (plotIsScalar !== scalarMode || ySeries.length !== plot.series.length - 1) {
			makePlot(container.clientWidth || 200, container.clientHeight || 120, ySeries.length);
		}
		if (!plot) return;
		plot.setData([xs, ...ySeries] as unknown as uPlot.AlignedData);
	}

	/** The value (x) axis span: the manual cog Y-range when set, else a
	 * self-calibrating span from the values seen so far (so the bar visibly
	 * slides). Degenerate spans get a unit of padding. */
	function scalarXRange(): [number, number] {
		if (!mYAuto && isFinite(mYMin) && isFinite(mYMax) && mYMin !== mYMax) {
			return mYMin < mYMax ? [mYMin, mYMax] : [mYMax, mYMin];
		}
		const lo = scalarMin;
		const hi = scalarMax;
		if (!isFinite(lo) || !isFinite(hi)) return [lastScalar - 1, lastScalar + 1];
		if (lo === hi) return [lo - 1, hi + 1];
		const pad = (hi - lo) * 0.05;
		return [lo - pad, hi + pad];
	}

	/** Draw a single scalar as a vertical bar at x = value (two points: y 0→1). */
	function drawScalar(value: number): void {
		scalarMode = true;
		lastScalar = value;
		if (isFinite(value)) {
			if (value < scalarMin) scalarMin = value;
			if (value > scalarMax) scalarMax = value;
		}
		if (!plot || !container) return;
		if (!plotIsScalar) makePlot(container.clientWidth || 200, container.clientHeight || 120, 1);
		if (!plot) return;
		plot.setData([[value, value], [0, 1]] as unknown as uPlot.AlignedData);
	}

	function pushData(arr: ArrayData, envelope: { origLen: number } | null = null): void {
		if (!plot || !container) return;
		const shape = arr.shape;
		const flatLen = arr.values.length;

		// A single scalar → vertical bar at x = value (see drawScalar).
		if (flatLen === 1) {
			drawScalar(Number(arr.values[0]));
			return;
		}
		scalarMode = false;
		scalarMin = Infinity;
		scalarMax = -Infinity;

		// nSeries (channels) × m (samples per channel) from the shape.
		let nSeries: number;
		let m: number;
		if (shape.length <= 1) {
			nSeries = 1;
			m = flatLen;
		} else if (shape.length === 2) {
			nSeries = shape[0];
			m = shape[1];
		} else {
			return; // higher-D handled by parent fallback
		}

		// Hand typed arrays straight to uPlot — no per-frame number[] copy (the wire is
		// f32-only, so `values` is always a Float32Array of real numbers).
		const v = arr.values as ArrayLike<number> & {
			subarray?: (start: number, end: number) => ArrayLike<number>;
		};
		const ySeries: ArrayLike<number>[] = [];
		if (nSeries === 1) {
			ySeries.push(v);
		} else {
			for (let c = 0; c < nSeries; c++) {
				if (v.subarray) {
					ySeries.push(v.subarray(c * m, (c + 1) * m));
				} else {
					const row = new Array<number>(m);
					for (let i = 0; i < m; i++) row[i] = v[c * m + i];
					ySeries.push(row);
				}
			}
		}

		// One x-index origin for BOTH the decimated and non-decimated paths: 1 under
		// log-x so no x is <= 0 (uPlot's log10(0) = -Infinity collapses the x-scale
		// and blanks the plot), else 0. Deciding it here keeps the two x-generators
		// from drifting — the decimation path once bypassed this and silently
		// reintroduced the log-x empty-plot bug.
		const base = mLogX ? 1 : 0;
		// The server already envelope-reduced the last axis (thalamus): the m samples ARE the
		// [min,max] band pairs. Draw them directly over the reconstructed original x-span —
		// re-decimating already-reduced data would blur the peaks the envelope preserved.
		if (envelope) {
			const band = envelopeBand(ySeries, envelope.origLen, base);
			setSeries(band.xs, band.ys);
			return;
		}
		// When there are far more samples than pixel columns, min/max-decimate to
		// the display budget — preserves spikes, cuts uPlot's per-frame work from
		// O(samples) to O(pixels) (report A15).
		const cols = container.clientWidth || 800;
		if (m > cols * 2) {
			const dec = decimateMinMax(ySeries, m, cols, base);
			setSeries(dec.xs, dec.ys);
			return;
		}
		setSeries(indexAxis(m, base), ySeries);
	}

	// The single "draw this frame" seam, used by BOTH the data effect and the settings-rebuild
	// re-seed. It must live in one place: if the data plane envelope-reduced the last axis, the
	// received min/max band is drawn directly (thalamus G3) rather than re-decimated — and BOTH
	// draw paths must compute that, or a settings change would redraw an enveloped frame through
	// the raw path (wrong x-span/labels, hidpi double-decimation).
	function drawFrame(f: DataFrame): void {
		const arr = f.data as ArrayData;
		pushData(arr, readEnvelope(f.meta, arr.shape.length));
	}

	$effect(() => {
		if (frame) drawFrame(frame);
	});

	$effect(() => {
		// Refresh the plain mirrors from the reactive settings (this read is what
		// makes the effect re-run on any change), then re-render with them.
		mLogX = logX;
		mLogY = logY;
		mYAuto = yAuto;
		mYMin = yMin;
		mYMax = yMax;
		mPoints = showPoints;
		if (!plot || !container) return;
		if (scalarMode) {
			// Y is locked and the value axis reads the live mirrors via the range
			// function, so a redraw (not a rebuild) picks up the new settings.
			drawScalar(lastScalar);
		} else {
			makePlot(container.clientWidth || 200, container.clientHeight || 120, lastNSeries || 1);
			// Re-seed the rebuilt plot with the LATEST frame, but read it untracked:
			// this effect must depend on the settings only, never on `frame`. Once a
			// setting is toggled (plot now exists, so we reach this branch instead of
			// early-returning at mount), a tracked `frame` read would make every data
			// frame re-run this effect and rebuild the whole plot — thrashing uPlot
			// (and, under a log scale, breaking the hover crosshair).
			const f = untrack(() => frame);
			if (f) drawFrame(f);
		}
	});

	// The live pointer event for the cursor.move hook — captured before uPlot's
	// own (over-bound) handler runs, so move() can recompute the position from
	// the current rect instead of uPlot's transform-stale one. `pointerInside`
	// tracks whether the pointer is currently over the viewer; the move hook
	// reads it to decide whether to project the crosshair or hide it.
	let lastMove: MouseEvent | null = null;
	let pointerInside = false;
	function captureMove(e: MouseEvent): void {
		lastMove = e;
		pointerInside = true;
	}
	// Pointer left the viewer: drop the cached event and retract the crosshair
	// immediately (don't wait for the next data frame — the stream may be slow
	// or paused). setCursor → move() returns the hidden sentinel, which nulls
	// cursor.idx and clears the value chip via the setCursor hook.
	function handleLeave(): void {
		pointerInside = false;
		lastMove = null;
		plot?.setCursor({ left: -10, top: -10 });
	}

	onMount(() => {
		if (!container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, 1);
		container.addEventListener('mousemove', captureMove, true);
		container.addEventListener('mouseleave', handleLeave);
		resizer = new ResizeObserver(() => {
			if (!container || !plot) return;
			plot.setSize({ width: container.clientWidth, height: container.clientHeight });
		});
		resizer.observe(container);
	});

	onDestroy(() => {
		container?.removeEventListener('mousemove', captureMove, true);
		container?.removeEventListener('mouseleave', handleLeave);
		resizer?.disconnect();
		plot?.destroy();
		plot = null;
	});
</script>

<div class="container" bind:this={container}>
	{#if cursorIdx !== null && cursorValues.length > 0}
		<div class="cursor-chip" data-testid="cursor-chip">
			<span class="cursor-x">x={cursorXValue !== null ? fmtTick(cursorXValue) : '—'}</span>
			{#each cursorValues as v, i (i)}
				<span class="cursor-y" style="color: {SERIES[i % SERIES.length]};">
					{v !== null ? fmtTick(v) : '—'}
				</span>
			{/each}
		</div>
	{/if}
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		min-height: 100px;
		min-width: 80px;
		position: relative;
	}
	:global(.uplot, .uplot *, .uplot *::before, .uplot *::after) {
		font-family: var(--font-mono);
	}
	.cursor-chip {
		/* Bottom-center: corners are taken by the min/max corner ticks
		   (`drawCornerTicks`) and the cursor-chip would otherwise overlap
		   them on every hover. */
		position: absolute;
		bottom: 4px;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		gap: var(--space-3);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		padding: var(--space-1) var(--space-3);
		background: color-mix(in srgb, var(--bg) 78%, transparent);
		border: 1px solid color-mix(in srgb, var(--text-muted) 30%, transparent);
		border-radius: 3px;
		color: var(--text-dim);
		pointer-events: none;
		max-width: calc(100% - 8px);
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
		z-index: 2;
	}
	.cursor-x {
		color: var(--text-muted);
	}
	.cursor-y {
		font-variant-numeric: tabular-nums;
	}
</style>
