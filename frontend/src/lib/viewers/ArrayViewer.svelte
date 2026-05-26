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

	function makePlot(width: number, height: number, nSeries: number): void {
		plot?.destroy();
		const opts: uPlot.Options = {
			width: Math.max(60, width),
			height: Math.max(60, height),
			padding: [4, 6, 4, 6],
			series: buildSeries(nSeries),
			axes: [
				{
					stroke: '#9aa3b3',
					grid: { stroke: 'rgba(255,255,255,0.05)' },
					ticks: { stroke: 'rgba(255,255,255,0.05)' }
				},
				{
					stroke: '#9aa3b3',
					grid: { stroke: 'rgba(255,255,255,0.05)' },
					ticks: { stroke: 'rgba(255,255,255,0.05)' }
				}
			],
			scales: { x: { time: false }, y: { auto: true } },
			cursor: { show: false },
			legend: { show: false }
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
