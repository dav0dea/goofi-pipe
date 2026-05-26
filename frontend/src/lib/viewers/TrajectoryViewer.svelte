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

	function makePlot(width: number, height: number, nPairs: number): void {
		plot?.destroy();
		if (!container) return;
		plot = new uPlot(
			{
				width: Math.max(60, width),
				height: Math.max(60, height),
				series: buildSeries(nPairs),
				axes: [
					{ stroke: '#9aa3b3', grid: { stroke: 'rgba(255,255,255,0.05)' } },
					{ stroke: '#9aa3b3', grid: { stroke: 'rgba(255,255,255,0.05)' } }
				],
				scales: { x: { time: false, auto: true }, y: { auto: true } },
				cursor: { show: false },
				legend: { show: false }
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
