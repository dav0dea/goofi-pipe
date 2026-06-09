<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './viewerSettings.svelte';
	import { onMount, onDestroy } from 'svelte';
	import uPlot from 'uplot';
	import 'uplot/dist/uPlot.min.css';

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
	const histLen = $derived(Math.max(2, Math.floor(Number(settings.history ?? 256))));

	let container: HTMLDivElement | null = $state(null);
	let plot: uPlot | null = null;
	let resizer: ResizeObserver | null = null;
	let lastNSeries = 0;

	// Single-sample streams (m === 1: a lone scalar, or N channels with one value
	// each) have nothing to draw as a line on their own, so we roll each series'
	// value into a rolling history and plot THAT as a time-series. `scalarMode`
	// records whether the last frame was such a stream, so a settings rebuild
	// re-renders the accumulated history instead of re-ingesting (which would
	// double-count) the last frame.
	let scalarMode = false;
	let scalarHist: number[][] = [];

	// Reused x-index array (0..m-1), rebuilt only when the sample count changes.
	// uPlot keeps a reference to its data, so a stable index avoids allocating a
	// fresh array every frame.
	let xsCache: Float64Array | null = null;
	let xsLen = -1;
	function indexAxis(m: number): Float64Array {
		if (xsLen !== m || !xsCache) {
			const a = new Float64Array(m);
			for (let i = 0; i < m; i++) a[i] = i;
			xsCache = a;
			xsLen = m;
		}
		return xsCache;
	}

	// Cursor hover values, updated by the uPlot setCursor hook. Rendered
	// as a top-right floating chip. `null` index = mouse not over plot.
	let cursorIdx = $state<number | null>(null);
	let cursorValues = $state<(number | null)[]>([]);
	let cursorXValue = $state<number | null>(null);

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	const PALETTE = [
		'#7ab7ff',
		'#b58cff',
		'#5dd09a',
		'#ffb761',
		'#ff7aa2',
		'#9aa3b3',
		'#c5c8d6',
		'#6e7686'
	];

	function buildSeries(nSeries: number): uPlot.Series[] {
		const out: uPlot.Series[] = [{ label: 'x' }];
		for (let i = 0; i < nSeries; i++) {
			out.push({
				label: `c${i}`,
				stroke: PALETTE[i % PALETTE.length],
				width: 1,
				points: { show: showPoints, size: 4 }
			});
		}
		return out;
	}

	function fmtTick(v: number): string {
		if (!Number.isFinite(v)) return '';
		const abs = Math.abs(v);
		if (abs === 0) return '0';
		if (abs >= 10000 || abs < 0.01) return v.toExponential(1);
		if (abs >= 100) return v.toFixed(0);
		if (abs >= 1) return v.toFixed(2);
		return v.toFixed(3);
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
		// uPlot.Scale.distr: 1 = linear, 3 = log10. Log scales reject
		// non-positive values, so we silently clamp via the data path
		// (filter happens in pushData when logY is on).
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [2, 2, 2, 2],
			series: buildSeries(nSeries),
			axes: [noMarginAxis, noMarginAxis],
			scales: {
				x: { time: false, distr: logX ? 3 : 1 },
				// Manual Y range (from the cog menu) pins the scale; otherwise
				// uPlot auto-fits to the data.
				y: yAuto
					? { auto: true, distr: logY ? 3 : 1 }
					: { auto: false, distr: logY ? 3 : 1, range: [yMin, yMax] }
			},
			cursor: {
				show: true,
				drag: { x: false, y: false, setScale: false },
				points: { show: true, size: 5 },
				// The node is CSS-scaled (and re-positioned) by the SvelteFlow zoom,
				// which uPlot's cached over-rect doesn't track — so the crosshair
				// drifts. Recompute the position straight from the live pointer event
				// and the current rect, dividing the scale back out, and ignore the
				// value uPlot derived from its stale rect.
				move: (u, left, top) => {
					const e = lastMove;
					if (!e) return [left, top];
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
		plot = new uPlot(opts, [[]], container);
	}

	/** Push aligned data, rebuilding the plot if the series count changed. */
	function setSeries(xs: ArrayLike<number>, ySeries: ArrayLike<number>[]): void {
		if (!plot || !container) return;
		if (ySeries.length !== plot.series.length - 1) {
			makePlot(container.clientWidth || 200, container.clientHeight || 120, ySeries.length);
		}
		if (!plot) return;
		plot.setData([xs, ...ySeries] as unknown as uPlot.AlignedData);
	}

	function pushData(arr: ArrayData): void {
		if (!plot || !container) return;
		const shape = arr.shape;
		const flatLen = arr.values.length;
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

		// One sample per series → roll it into a scrolling time-series instead of
		// trying (and failing) to draw a one-point line.
		if (m === 1) {
			ingestScalarFrame(arr, nSeries);
			return;
		}
		scalarMode = false;
		scalarHist = [];

		// Hand typed arrays straight to uPlot — no per-frame number[] copy. BigInt
		// dtypes (i8/u8) aren't numbers to uPlot, so only those fall back to a copy.
		const v = arr.values as ArrayLike<number> & {
			subarray?: (start: number, end: number) => ArrayLike<number>;
		};
		const isBig = flatLen > 0 && typeof v[0] === 'bigint';
		const xs = indexAxis(m);
		const ySeries: ArrayLike<number>[] = [];
		if (nSeries === 1) {
			ySeries.push(isBig ? Array.from(v, Number) : v);
		} else {
			for (let c = 0; c < nSeries; c++) {
				if (!isBig && v.subarray) {
					ySeries.push(v.subarray(c * m, (c + 1) * m));
				} else {
					const row = new Array<number>(m);
					for (let i = 0; i < m; i++) row[i] = Number(v[c * m + i]);
					ySeries.push(row);
				}
			}
		}
		setSeries(xs, ySeries);
	}

	/** Append this frame's per-series scalar(s) to the rolling history. */
	function ingestScalarFrame(arr: ArrayData, nSeries: number): void {
		scalarMode = true;
		const v = arr.values as ArrayLike<number>;
		const isBig = v.length > 0 && typeof v[0] === 'bigint';
		if (scalarHist.length !== nSeries) scalarHist = Array.from({ length: nSeries }, () => []);
		for (let s = 0; s < nSeries; s++) {
			scalarHist[s].push(isBig ? Number(v[s]) : (v[s] as number));
		}
		renderScalar();
	}

	/** Draw the accumulated scalar history (also trims it to the current window). */
	function renderScalar(): void {
		if (!plot || scalarHist.length === 0) return;
		for (const h of scalarHist) if (h.length > histLen) h.splice(0, h.length - histLen);
		setSeries(indexAxis(scalarHist[0].length), scalarHist);
	}

	$effect(() => {
		if (frame) pushData(asArray(frame.data));
	});

	$effect(() => {
		// Rebuild whenever any plot-shaping setting changes. Reading them all up
		// front makes the dependency set explicit.
		void [logX, logY, yAuto, yMin, yMax, showPoints, histLen];
		if (!plot || !container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, lastNSeries || 1);
		// In scalar mode re-render the accumulated history (re-ingesting the last
		// frame would double-count it); otherwise re-push the current frame.
		if (scalarMode) renderScalar();
		else if (frame) pushData(asArray(frame.data));
	});

	// The live pointer event for the cursor.move hook — captured before uPlot's
	// own (over-bound) handler runs, so move() can recompute the position from
	// the current rect instead of uPlot's transform-stale one.
	let lastMove: MouseEvent | null = null;
	function captureMove(e: MouseEvent): void {
		lastMove = e;
	}

	onMount(() => {
		if (!container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, 1);
		container.addEventListener('mousemove', captureMove, true);
		resizer = new ResizeObserver(() => {
			if (!container || !plot) return;
			plot.setSize({ width: container.clientWidth, height: container.clientHeight });
		});
		resizer.observe(container);
	});

	onDestroy(() => {
		container?.removeEventListener('mousemove', captureMove, true);
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
				<span class="cursor-y" style="color: {PALETTE[i % PALETTE.length]};">
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
		gap: 6px;
		font-family: var(--font-mono);
		font-size: 9px;
		padding: 2px 6px;
		background: color-mix(in srgb, var(--bg) 78%, transparent);
		border: 1px solid color-mix(in srgb, var(--text-faint) 30%, transparent);
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
		color: var(--text-faint);
	}
	.cursor-y {
		font-variant-numeric: tabular-nums;
	}
</style>
