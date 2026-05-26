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

	function fmtTick(v: number): string {
		if (!Number.isFinite(v)) return '';
		const abs = Math.abs(v);
		if (abs === 0) return '0';
		if (abs >= 10000 || abs < 0.01) return v.toExponential(1);
		if (abs >= 100) return v.toFixed(0);
		if (abs >= 1) return v.toFixed(2);
		return v.toFixed(3);
	}

	function drawCornerTicks(u: uPlot): void {
		const ctx = u.ctx;
		const r = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
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

	function makePlot(width: number, height: number, nPairs: number): void {
		plot?.destroy();
		if (!container) return;
		const noMarginAxis: uPlot.Axis = {
			show: true,
			size: 0,
			gap: 0,
			stroke: 'transparent',
			ticks: { show: false },
			grid: { show: true, stroke: 'rgba(255,255,255,0.05)' },
			values: () => []
		};
		plot = new uPlot(
			{
				width: Math.max(60, width),
				height: Math.max(60, height),
				padding: [2, 2, 2, 2],
				series: buildSeries(nPairs),
				axes: [noMarginAxis, noMarginAxis],
				scales: { x: { time: false, auto: true }, y: { auto: true } },
				cursor: { show: false },
				legend: { show: false },
				hooks: { draw: [drawCornerTicks] }
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
