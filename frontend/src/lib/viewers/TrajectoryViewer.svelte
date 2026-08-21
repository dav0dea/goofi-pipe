<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { onMount, onDestroy } from 'svelte';
	import { formatTick as fmtTick } from './format';
	import { SERIES, AXIS_INK, tickFont } from './palette';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const pointSize = $derived(Number(settings.pointSize ?? 2));
	const autoRange = $derived(settings.auto !== false);

	let container: HTMLDivElement | null = $state(null);
	let canvas: HTMLCanvasElement | null = $state(null);
	let ctx: CanvasRenderingContext2D | null = null;
	let resizer: ResizeObserver | null = null;

	// One adaptive range for both axes, so the shape stays undistorted. A custom canvas, not uPlot:
	// uPlot assumes a monotonic x-axis and shares one x-array across series.
	const MARGIN = 0.1;
	const SHRINKING = 0.01;
	const MAX_TRAJ = 64; // guard against n*(n-1)/2 pair explosion for large n
	let vmin: number | null = null;
	let vmax: number | null = null;

	// Kept so a resize or settings change can redraw without waiting for a frame.
	let rows: number[][] = [];

	/** All i<j row pairs → one trajectory each; row i is x, row j is y. */
	function pairList(n: number): [number, number][] {
		const out: [number, number][] = [];
		for (let i = 0; i < n; i++) {
			for (let j = i + 1; j < n; j++) {
				out.push([i, j]);
				if (out.length >= MAX_TRAJ) return out;
			}
		}
		return out;
	}

	function pushData(arr: ArrayData): void {
		const shape = arr.shape;
		if (shape.length !== 2 || shape[0] < 2) {
			rows = [];
			draw();
			return;
		}
		const n = shape[0];
		const nT = shape[1];
		const src = arr.values as ArrayLike<number>;

		const parsed: number[][] = [];
		let mn = Infinity;
		let mx = -Infinity;
		for (let i = 0; i < n; i++) {
			const row = new Array<number>(nT);
			for (let t = 0; t < nT; t++) {
				const v = src[i * nT + t];
				row[t] = v;
				if (Number.isFinite(v)) {
					if (v < mn) mn = v;
					if (v > mx) mx = v;
				}
			}
			parsed.push(row);
		}
		rows = parsed;

		if (Number.isFinite(mn) && Number.isFinite(mx)) {
			const have = vmin !== null && vmax !== null;
			if (autoRange) {
				// Shrink toward the centre first, then grow to fit — the order is load-bearing.
				if (have) {
					const nvmin = (vmin as number) * (1 - SHRINKING) + (vmax as number) * SHRINKING;
					const nvmax = (vmax as number) * (1 - SHRINKING) + nvmin * SHRINKING;
					vmin = nvmin;
					vmax = nvmax;
				}
				if (vmin === null || mn < vmin) vmin = mn;
				if (vmax === null || mx > vmax) vmax = mx;
			} else if (!have) {
				// Frozen mode still needs an initial range from the first frame.
				vmin = mn;
				vmax = mx;
			}
		}
		draw();
	}

	/** The shared [lo, hi] data range for both axes, with margin and a degenerate guard. */
	function rangeLoHi(): [number, number] {
		if (vmin === null || vmax === null) return [-1, 1];
		let lo = vmin - Math.abs(vmax) * MARGIN;
		let hi = vmax + Math.abs(vmax) * MARGIN;
		if (!(hi > lo)) {
			const c = Number.isFinite((lo + hi) / 2) ? (lo + hi) / 2 : 0;
			lo = c - 1;
			hi = c + 1;
		}
		return [lo, hi];
	}

	function drawGrid(
		w: number,
		h: number,
		lo: number,
		hi: number,
		px: (x: number) => number,
		py: (y: number) => number
	): void {
		if (!ctx) return;
		ctx.lineWidth = 1;
		ctx.strokeStyle = 'rgba(255,255,255,0.04)';
		ctx.beginPath();
		const DIV = 4;
		for (let i = 1; i < DIV; i++) {
			const gx = (i / DIV) * w;
			const gy = (i / DIV) * h;
			ctx.moveTo(gx, 0);
			ctx.lineTo(gx, h);
			ctx.moveTo(0, gy);
			ctx.lineTo(w, gy);
		}
		ctx.stroke();
		ctx.strokeStyle = 'rgba(255,255,255,0.10)';
		ctx.beginPath();
		if (lo < 0 && hi > 0) {
			const x0 = px(0);
			ctx.moveTo(x0, 0);
			ctx.lineTo(x0, h);
			const y0 = py(0);
			ctx.moveTo(0, y0);
			ctx.lineTo(w, y0);
		}
		ctx.stroke();
	}

	function drawCornerTicks(w: number, h: number, lo: number, hi: number): void {
		if (!ctx) return;
		ctx.font = tickFont(10);
		ctx.fillStyle = AXIS_INK;
		const pad = 4;
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(fmtTick(hi), pad, pad);
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(lo), pad, h - pad);
		ctx.textAlign = 'right';
		ctx.fillText(fmtTick(hi), w - pad, h - pad);
	}

	function draw(): void {
		if (!ctx || !canvas || !container) return;
		const dpr = window.devicePixelRatio || 1;
		const w = container.clientWidth;
		const h = container.clientHeight;
		// Backing store tracks element × DPR; the transform then lets everything draw in CSS pixels.
		const bw = Math.max(1, Math.round(w * dpr));
		const bh = Math.max(1, Math.round(h * dpr));
		if (canvas.width !== bw || canvas.height !== bh) {
			canvas.width = bw;
			canvas.height = bh;
		}
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, w, h);

		const [lo, hi] = rangeLoHi();
		const span = hi - lo || 1;
		const px = (x: number): number => ((x - lo) / span) * w;
		const py = (y: number): number => h - ((y - lo) / span) * h;

		drawGrid(w, h, lo, hi, px, py);

		const n = rows.length;
		if (n >= 2) {
			const pairs = pairList(n);
			const nT = rows[0].length;
			const drawPts = pointSize > 0 && nT <= 800; // skip per-vertex dots on long paths
			const headR = Math.max(pointSize, 2.5);
			for (let k = 0; k < pairs.length; k++) {
				const [i, j] = pairs[k];
				const xr = rows[i];
				const yr = rows[j];
				const color = SERIES[k % SERIES.length];

				ctx.strokeStyle = color;
				ctx.lineWidth = 1.4;
				ctx.lineJoin = 'round';
				ctx.lineCap = 'round';
				ctx.beginPath();
				let started = false;
				let lastX = NaN;
				let lastY = NaN;
				for (let t = 0; t < nT; t++) {
					const x = xr[t];
					const y = yr[t];
					if (!Number.isFinite(x) || !Number.isFinite(y)) {
						started = false; // break the path across gaps
						continue;
					}
					const cx = px(x);
					const cy = py(y);
					if (!started) {
						ctx.moveTo(cx, cy);
						started = true;
					} else {
						ctx.lineTo(cx, cy);
					}
					lastX = cx;
					lastY = cy;
				}
				ctx.stroke();

				if (drawPts) {
					ctx.fillStyle = color;
					const r = pointSize * 0.6;
					for (let t = 0; t < nT; t++) {
						const x = xr[t];
						const y = yr[t];
						if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
						ctx.beginPath();
						ctx.arc(px(x), py(y), r, 0, Math.PI * 2);
						ctx.fill();
					}
				}

				if (Number.isFinite(lastX) && Number.isFinite(lastY)) {
					ctx.fillStyle = color;
					ctx.beginPath();
					ctx.arc(lastX, lastY, headR, 0, Math.PI * 2);
					ctx.fill();
					ctx.strokeStyle = 'rgba(17,17,17,0.9)';
					ctx.lineWidth = 1;
					ctx.stroke();
				}
			}
		}

		drawCornerTicks(w, h, lo, hi);
	}

	$effect(() => {
		if (frame) pushData(frame.data as ArrayData);
	});

	$effect(() => {
		// Redraw when a style/range setting changes (the read registers the dep).
		void [pointSize, autoRange];
		draw();
	});

	onMount(() => {
		if (!canvas) return;
		ctx = canvas.getContext('2d');
		resizer = new ResizeObserver(() => draw());
		if (container) resizer.observe(container);
		draw();
	});

	onDestroy(() => {
		resizer?.disconnect();
		ctx = null;
	});
</script>

<div class="container" bind:this={container}>
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		min-height: 100px;
		min-width: 80px;
		position: relative;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
</style>
