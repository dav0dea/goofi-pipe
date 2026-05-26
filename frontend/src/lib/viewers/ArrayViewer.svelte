<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import { onMount, onDestroy } from 'svelte';
	import uPlot from 'uplot';
	import 'uplot/dist/uPlot.min.css';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	let container: HTMLDivElement | null = $state(null);
	let plot: uPlot | null = null;
	let resizer: ResizeObserver | null = null;

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	function buildSeries(nSeries: number): uPlot.Series[] {
		const palette = [
			'#7ab7ff',
			'#b58cff',
			'#5dd09a',
			'#ffb761',
			'#ff7aa2',
			'#9aa3b3',
			'#c5c8d6',
			'#6e7686'
		];
		const out: uPlot.Series[] = [{ label: 'x' }];
		for (let i = 0; i < nSeries; i++) {
			out.push({
				label: `c${i}`,
				stroke: palette[i % palette.length],
				width: 1,
				points: { show: false }
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
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [2, 2, 2, 2],
			series: buildSeries(nSeries),
			axes: [noMarginAxis, noMarginAxis],
			scales: { x: { time: false }, y: { auto: true } },
			cursor: { show: false },
			legend: { show: false },
			hooks: { draw: [drawCornerTicks] }
		};
		if (!container) return;
		plot = new uPlot(opts, [[]], container);
	}

	function pushData(arr: ArrayData): void {
		if (!plot || !container) return;
		const shape = arr.shape;
		const flatLen = arr.values.length;
		let xs: number[];
		let ySeries: number[][];
		if (shape.length === 0 || shape.length === 1) {
			xs = new Array(flatLen);
			for (let i = 0; i < flatLen; i++) xs[i] = i;
			ySeries = [Array.from(arr.values as ArrayLike<number>)];
		} else if (shape.length === 2) {
			const [n, m] = shape;
			xs = new Array(m);
			for (let i = 0; i < m; i++) xs[i] = i;
			ySeries = [];
			for (let c = 0; c < n; c++) {
				const row: number[] = new Array(m);
				for (let i = 0; i < m; i++) row[i] = Number((arr.values as ArrayLike<number>)[c * m + i]);
				ySeries.push(row);
			}
		} else {
			return; // higher-D handled by parent fallback
		}

		const expectedSeries = ySeries.length;
		const currentSeries = plot.series.length - 1;
		if (expectedSeries !== currentSeries) {
			const w = container.clientWidth || 200;
			const h = container.clientHeight || 120;
			makePlot(w, h, expectedSeries);
		}
		if (!plot) return;
		plot.setData([xs, ...ySeries] as unknown as uPlot.AlignedData);
	}

	$effect(() => {
		if (frame) pushData(asArray(frame.data));
	});

	onMount(() => {
		if (!container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, 1);
		resizer = new ResizeObserver(() => {
			if (!container || !plot) return;
			plot.setSize({ width: container.clientWidth, height: container.clientHeight });
		});
		resizer.observe(container);
	});

	onDestroy(() => {
		resizer?.disconnect();
		plot?.destroy();
		plot = null;
	});
</script>

<div class="container" bind:this={container}></div>

<style>
	.container {
		width: 100%;
		height: 100%;
		min-height: 100px;
		min-width: 80px;
	}
	:global(.uplot, .uplot *, .uplot *::before, .uplot *::after) {
		font-family: var(--font-mono);
	}
</style>
