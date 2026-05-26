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

	// Per-axis log-scale toggles. Re-makes the plot when flipped so the
	// scale `distr` switches between linear (1) and log10 (3).
	let logX = $state(false);
	let logY = $state(false);
	let lastNSeries = 0;

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
				y: { auto: true, distr: logY ? 3 : 1 }
			},
			cursor: {
				show: true,
				drag: { x: false, y: false, setScale: false },
				points: { show: true, size: 5 }
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

	$effect(() => {
		// Rebuild whenever the log-scale flags flip. Reading both up front
		// makes the dependency explicit; using a no-op reference of plot
		// guards against running before mount.
		void logX;
		void logY;
		if (!plot || !container) return;
		makePlot(container.clientWidth || 200, container.clientHeight || 120, lastNSeries || 1);
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

<div class="container" bind:this={container}>
	<div class="scale-toggles" role="group" aria-label="scale toggles">
		<button
			class="scale-btn"
			class:on={logX}
			onclick={(e) => {
				e.stopPropagation();
				logX = !logX;
			}}
			title="toggle log scale on x"
			data-testid="log-x-toggle"
		>
			log x
		</button>
		<button
			class="scale-btn"
			class:on={logY}
			onclick={(e) => {
				e.stopPropagation();
				logY = !logY;
			}}
			title="toggle log scale on y"
			data-testid="log-y-toggle"
		>
			log y
		</button>
	</div>
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
	.scale-toggles {
		/* Sits in the top-right of the plot area; pointer-events on the
		   buttons themselves so the rest of the canvas remains hoverable
		   for the uPlot cursor. */
		position: absolute;
		top: 4px;
		right: 4px;
		display: flex;
		gap: 3px;
		z-index: 2;
		pointer-events: none;
	}
	.scale-btn {
		pointer-events: auto;
		font-family: var(--font-mono);
		font-size: 9px;
		letter-spacing: 0.04em;
		padding: 1px 5px;
		border: 1px solid color-mix(in srgb, var(--text-faint) 30%, transparent);
		background: color-mix(in srgb, var(--bg) 60%, transparent);
		color: var(--text-faint);
		border-radius: 3px;
		cursor: pointer;
		transition:
			background 80ms ease,
			color 80ms ease,
			border-color 80ms ease;
		opacity: 0;
	}
	.container:hover .scale-btn,
	.scale-btn.on {
		opacity: 1;
	}
	.scale-btn:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.scale-btn.on {
		background: color-mix(in srgb, var(--accent) 25%, transparent);
		color: var(--accent);
		border-color: var(--accent);
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
