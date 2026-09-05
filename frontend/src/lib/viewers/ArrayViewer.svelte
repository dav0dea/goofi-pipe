<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { onMount, onDestroy, untrack } from 'svelte';
	import uPlot from 'uplot';
	import 'uplot/dist/uPlot.min.css';
	import { decimateMinMax } from './decimate';
	import { readEnvelope, envelopeBand } from './envelope';
	import { formatTick as fmtTick } from './format';
	import { logSafe, logSplits } from './logScale';
	import { SERIES, AXIS_INK, tickFont } from './palette';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const logX = $derived(Boolean(settings.logX));
	const logY = $derived(Boolean(settings.logY));
	const yAuto = $derived(settings.yAuto !== false);
	const yMin = $derived(Number(settings.yMin ?? -1));
	const yMax = $derived(Number(settings.yMax ?? 1));
	const showPoints = $derived(Boolean(settings.points));

	// Plain mirrors: the data path reads only these, so pushing a frame never depends on a setting.
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

	// A lone scalar is drawn as a vertical bar at x = value, not as a misleading scrolling history.
	let scalarMode = false;
	let plotIsScalar = false;
	let lastScalar = 0;
	let scalarMin = Infinity;
	let scalarMax = -Infinity;

	// uPlot keeps a reference to its data, so a stable index avoids a fresh array every frame.
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

	/** uPlot draw hook: paint the corner tick labels INSIDE the plot bbox, so the canvas fills the body. */
	function drawCornerTicks(u: uPlot): void {
		const ctx = u.ctx;
		// uPlot does not pre-scale its ctx, so canvas coords are device pixels.
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
		ctx.font = tickFont(10 * r);
		ctx.fillStyle = AXIS_INK;
		const pad = 4 * r;

		if (scalarMode) {
			// Only the value (x) range is meaningful; y is the locked bar height.
			ctx.textBaseline = 'bottom';
			ctx.textAlign = 'left';
			ctx.fillText(fmtTick(xMin), left + pad, bottom - pad);
			ctx.textAlign = 'right';
			ctx.fillText(fmtTick(xMax), right - pad, bottom - pad);
			ctx.restore();
			return;
		}

		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(fmtTick(yMax), left + pad, top + pad);
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(yMin), left + pad, bottom - pad);

		ctx.textAlign = 'right';
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(xMax), right - pad, bottom - pad);

		ctx.restore();
	}

	/** The index axis. A log-x index is 1-based, but a frame with NO samples hands uPlot a null
	 * extent, and its own log range walks up from zero — so the floor is stated here too. */
	function xScale(): uPlot.Scale {
		return mLogX
			? { time: false, distr: 3, range: (_u, lo, hi) => logSafe(lo, hi) }
			: { time: false, distr: 1 };
	}

	/** The value axis. Only the log form is floored — a linear one takes the range as given. */
	function yScale(): uPlot.Scale {
		const distr = mLogY ? 3 : 1;
		if (!mLogY) {
			return mYAuto ? { auto: true, distr } : { auto: false, distr, range: [mYMin, mYMax] };
		}
		return mYAuto
			? { auto: true, distr, range: (_u, lo, hi) => logSafe(lo, hi) }
			: { auto: false, distr, range: logSafe(mYMin, mYMax) };
	}

	function makePlot(width: number, height: number, nSeries: number): void {
		plot?.destroy();
		lastNSeries = nSeries;
		plotIsScalar = scalarMode;
		// `size: 0` reclaims the axis margin; the `draw` hook paints the ticks inside the canvas.
		const axis = (log: boolean): uPlot.Axis => ({
			show: true,
			size: 0,
			gap: 0,
			stroke: 'transparent',
			ticks: { show: false },
			grid: { show: true, stroke: 'rgba(255,255,255,0.05)' },
			values: () => [],
			...(log ? { splits: (_u: uPlot, _i: number, lo: number, hi: number) => logSplits(lo, hi) } : {})
		});
		const axes = scalarMode ? [axis(false), axis(false)] : [axis(mLogX), axis(mLogY)];
		// distr 3 = log10.
		const scales: uPlot.Options['scales'] = scalarMode
			? {
					x: { time: false, range: () => scalarXRange() },
					y: { auto: false, range: [0, 1] }
				}
			: {
					x: xScale(),
					y: yScale()
				};
		const series: uPlot.Series[] = scalarMode
			? [{ label: 'x' }, { label: 'v', stroke: SERIES[0], width: 2, points: { show: false } }]
			: buildSeries(nSeries);
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [2, 2, 2, 2],
			series,
			axes,
			scales,
			cursor: {
				show: true,
				drag: { x: false, y: false, setScale: false },
				points: { show: true, size: 5 },
				// The SvelteFlow zoom CSS-scales the node and uPlot's cached over-rect does not track it, so
				// recompute from the live pointer; this hook also owns the hide, as mouseleave is unreliable here.
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

	/** The value (x) axis span: the manual cog Y-range when set, else a self-calibrating one. */
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

		if (flatLen === 1) {
			drawScalar(Number(arr.values[0]));
			return;
		}
		scalarMode = false;
		scalarMin = Infinity;
		scalarMax = -Infinity;

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

		// One origin for both x-generators: 1 under log-x, since log10(0) collapses uPlot's x-scale.
		const base = mLogX ? 1 : 0;
		// Already an envelope band: re-decimating reduced data would blur the peaks it preserved.
		if (envelope) {
			const band = envelopeBand(ySeries, envelope.origLen, base);
			setSeries(band.xs, band.ys);
			return;
		}
		const cols = container.clientWidth || 800;
		if (m > cols * 2) {
			const dec = decimateMinMax(ySeries, m, cols, base);
			setSeries(dec.xs, dec.ys);
			return;
		}
		setSeries(indexAxis(m, base), ySeries);
	}

	// The single "draw this frame" seam: both the data effect and the settings re-seed go through it.
	function drawFrame(f: DataFrame): void {
		const arr = f.data as ArrayData;
		pushData(arr, readEnvelope(f.meta, arr.shape.length));
	}

	$effect(() => {
		if (frame) drawFrame(frame);
	});

	$effect(() => {
		mLogX = logX;
		mLogY = logY;
		mYAuto = yAuto;
		mYMin = yMin;
		mYMax = yMax;
		mPoints = showPoints;
		if (!plot || !container) return;
		if (scalarMode) {
			// The value axis reads the live mirrors through its range fn, so a redraw is enough.
			drawScalar(lastScalar);
		} else {
			makePlot(container.clientWidth || 200, container.clientHeight || 120, lastNSeries || 1);
			// Untracked: this effect must depend on the settings alone, or every frame would rebuild the plot.
			const f = untrack(() => frame);
			if (f) drawFrame(f);
		}
	});

	// Captured before uPlot's own over-bound handler, so move() can use the current rect.
	// POINTER, not mouse: `mousemove` is dispatched for a mouse alone, so touch had no readout.
	let lastMove: MouseEvent | null = null;
	let pointerInside = false;
	function captureMove(e: PointerEvent): void {
		lastMove = e;
		pointerInside = true;
		// Touch and pen produce no `mousemove` while down, so drive the cursor from the pointer event.
		if (e.pointerType !== 'mouse') plot?.setCursor({ left: 0, top: 0 });
	}
	// Retract at once rather than on the next data frame, which may be slow or paused.
	function handleLeave(e: PointerEvent): void {
		// A touch pointer is destroyed on release, so retracting on its `pointerleave` would flash the readout away.
		if (e.type === 'pointerleave' && e.pointerType !== 'mouse') return;
		pointerInside = false;
		lastMove = null;
		plot?.setCursor({ left: -10, top: -10 });
	}

	onMount(() => {
		if (!container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, 1);
		// `pointerdown` too: a tap is a press with no motion, and it is the whole gesture on touch.
		container.addEventListener('pointermove', captureMove, true);
		container.addEventListener('pointerdown', captureMove, true);
		container.addEventListener('pointerleave', handleLeave);
		container.addEventListener('pointercancel', handleLeave);
		resizer = new ResizeObserver(() => {
			if (!container || !plot) return;
			plot.setSize({ width: container.clientWidth, height: container.clientHeight });
		});
		resizer.observe(container);
	});

	onDestroy(() => {
		container?.removeEventListener('pointermove', captureMove, true);
		container?.removeEventListener('pointerdown', captureMove, true);
		container?.removeEventListener('pointerleave', handleLeave);
		container?.removeEventListener('pointercancel', handleLeave);
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
		/* Bottom-center: the corners are taken by the min/max corner ticks. */
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
