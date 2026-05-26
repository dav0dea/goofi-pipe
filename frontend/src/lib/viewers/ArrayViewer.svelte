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

	/** Draw a handful of tick labels INSIDE the plot area.
	 *
	 * Default uPlot reserves a left + bottom gutter for axis labels. With many
	 * small viewers per node, that gutter eats most of the canvas. We hide
	 * the native axes (size=0, show=false) and use the `draw` hook to render
	 * 3 X ticks along the bottom edge and 2 Y ticks (min/max) at the top-
	 * and bottom-left, all overlaid on the plot itself.
	 */
	function drawInsideTicks(u: uPlot): void {
		const ctx = u.ctx;
		const x = u.scales.x;
		const y = u.scales.y;
		if (x?.min === undefined || x?.max === undefined) return;
		if (y?.min === undefined || y?.max === undefined) return;

		const plotL = u.bbox.left;
		const plotT = u.bbox.top;
		const plotW = u.bbox.width;
		const plotH = u.bbox.height;

		const dpr = window.devicePixelRatio || 1;
		ctx.save();
		ctx.font = `${10 * dpr}px "JetBrains Mono", ui-monospace, monospace`;
		ctx.fillStyle = 'rgba(150, 158, 175, 0.7)';
		ctx.textBaseline = 'bottom';

		// X ticks: 3 evenly spaced labels (left / mid / right) along the
		// bottom edge of the plot, anchored just inside the right edge so
		// they don't overflow.
		const xMin = x.min;
		const xMax = x.max;
		const xs = [xMin, (xMin + xMax) / 2, xMax];
		const xAnchors: CanvasTextAlign[] = ['left', 'center', 'right'];
		for (let i = 0; i < xs.length; i++) {
			ctx.textAlign = xAnchors[i];
			const label = fmt(xs[i]);
			let px = plotL + ((xs[i] - xMin) / (xMax - xMin)) * plotW;
			// Inset the edges by a couple of pixels so text doesn't clip.
			if (i === 0) px += 3;
			if (i === xs.length - 1) px -= 3;
			ctx.fillText(label, px, plotT + plotH - 2);
		}

		// Y ticks: min/max in the top-left and bottom-left of the plot.
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(fmt(y.max), plotL + 4, plotT + 2);
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmt(y.min), plotL + 4, plotT + plotH - 12 * dpr);
		ctx.restore();
	}

	function fmt(v: number): string {
		const av = Math.abs(v);
		if (av === 0) return '0';
		if (av >= 1000 || av < 0.01) return v.toExponential(1);
		if (av >= 10) return v.toFixed(0);
		return v.toPrecision(2);
	}

	function makePlot(width: number, height: number, nSeries: number): void {
		plot?.destroy();
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [2, 2, 2, 2],
			series: buildSeries(nSeries),
			// Axes hidden — labels are drawn inside the plot via the `draw`
			// hook below. We still want gridlines, so keep `show: true` for
			// the side rules but force size=0 so no gutter is reserved.
			axes: [
				{ show: false, size: 0 },
				{ show: false, size: 0 }
			],
			scales: { x: { time: false }, y: { auto: true } },
			cursor: { show: false },
			legend: { show: false },
			hooks: {
				draw: [(u) => drawInsideTicks(u)]
			}
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
