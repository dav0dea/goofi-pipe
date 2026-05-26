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

	function buildSeries(nPairs: number): uPlot.Series[] {
		const palette = [
			'#7ab7ff',
			'#b58cff',
			'#5dd09a',
			'#ffb761',
			'#ff7aa2',
			'#9aa3b3',
			'#c5c8d6'
		];
		const out: uPlot.Series[] = [{ label: 'x' }];
		for (let i = 0; i < nPairs; i++) {
			out.push({
				label: `trace ${i}`,
				stroke: palette[i % palette.length],
				width: 1,
				points: { show: true, size: 3, stroke: palette[i % palette.length] }
			});
		}
		return out;
	}

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

		const xs = [x.min, x.max];
		const xAlign: CanvasTextAlign[] = ['left', 'right'];
		for (let i = 0; i < xs.length; i++) {
			ctx.textAlign = xAlign[i];
			let px = plotL + ((xs[i] - x.min) / (x.max - x.min)) * plotW;
			if (i === 0) px += 3;
			if (i === xs.length - 1) px -= 3;
			ctx.fillText(fmt(xs[i]), px, plotT + plotH - 2);
		}
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

	function makePlot(width: number, height: number, nPairs: number): void {
		plot?.destroy();
		if (!container) return;
		plot = new uPlot(
			{
				width: Math.max(60, width),
				height: Math.max(60, height),
				padding: [2, 2, 2, 2],
				series: buildSeries(nPairs),
				axes: [
					{ show: false, size: 0 },
					{ show: false, size: 0 }
				],
				scales: { x: { time: false, auto: true }, y: { auto: true } },
				cursor: { show: false },
				legend: { show: false },
				hooks: { draw: [(u) => drawInsideTicks(u)] }
			},
			[[]],
			container
		);
	}

	function pushData(arr: ArrayData): void {
		if (!plot || !container) return;
		const shape = arr.shape;
		if (shape.length !== 2 || shape[0] < 2) return;
		const [nDim, nT] = shape;
		// Strategy: take consecutive pairs (0,1), (2,3), ... If odd dim count
		// the last dim is dropped.
		const nPairs = Math.floor(nDim / 2);
		const currentSeries = plot.series.length - 1;
		if (nPairs !== currentSeries) {
			makePlot(container.clientWidth || 200, container.clientHeight || 120, nPairs);
		}
		if (!plot) return;
		const src = arr.values as ArrayLike<number>;
		const xs: number[] = new Array(nT);
		const out: number[][] = [];
		// uPlot wants aligned data; pick the x of the first pair as the x-axis.
		// For trajectories that's fine — every pair is the same time-axis
		// length so we expose pair 0's x.
		for (let i = 0; i < nT; i++) xs[i] = Number(src[i]);
		for (let p = 0; p < nPairs; p++) {
			const row = new Array<number>(nT);
			for (let i = 0; i < nT; i++) row[i] = Number(src[(p * 2 + 1) * nT + i]);
			out.push(row);
		}
		plot.setData([xs, ...out] as unknown as uPlot.AlignedData);
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
</style>
