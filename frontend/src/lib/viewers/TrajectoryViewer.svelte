<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { onMount, onDestroy } from 'svelte';
	import { formatTick as fmtTick } from './format';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	// Cog-menu settings (persisted per node+slot).
	const pointSize = $derived(Number(settings.pointSize ?? 2));
	// Auto adapts the shared range to the data each frame; off freezes it.
	const autoRange = $derived(settings.auto !== false);

	let container: HTMLDivElement | null = $state(null);
	let canvas: HTMLCanvasElement | null = $state(null);
	let ctx: CanvasRenderingContext2D | null = null;
	let resizer: ResizeObserver | null = null;

	// Adaptive range SHARED by both axes — ported from the dearpygui
	// TrajectoryViewer (margin=0.1, shrinking=0.01). It grows instantly to fit
	// new extremes and decays back slowly, so the view tracks the path's span
	// without jitter, and an equal x/y data range keeps the shape undistorted.
	//
	// Why a custom canvas instead of uPlot (which every other line plot uses):
	// uPlot assumes a sorted, monotonic x-axis. It derives the x-range from the
	// FIRST and LAST x samples (`autoScaleX`), which a trajectory's non-monotonic
	// path violates — that was the broken xlim. uPlot also shares one x-array
	// across series, so it can't draw N independent (x,y) paths. A trajectory is
	// simply not a time series, so it gets its own renderer.
	const MARGIN = 0.1;
	const SHRINKING = 0.01;
	const MAX_TRAJ = 64; // guard against n*(n-1)/2 pair explosion for large n
	let vmin: number | null = null;
	let vmax: number | null = null;

	// Last parsed rows (n × nTime), kept so a resize or settings change can
	// redraw without waiting for a fresh data frame.
	let rows: number[][] = [];

	const PALETTE = ['#7ab7ff', '#b58cff', '#5dd09a', '#ffb761', '#ff7aa2', '#9aa3b3', '#c5c8d6'];



	/** All i<j row pairs → one trajectory each, matching the old viewer's
	 * n*(n-1)/2 behaviour (row i is x, row j is y). Capped for safety. */
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
		// Trajectories need a 2-D (dims × frames) array with at least 2 dims.
		if (shape.length !== 2 || shape[0] < 2) {
			rows = [];
			draw();
			return;
		}
		const n = shape[0];
		const nT = shape[1];
		const src = arr.values as ArrayLike<number>;
		const big = src.length > 0 && typeof src[0] === 'bigint';

		const parsed: number[][] = [];
		let mn = Infinity;
		let mx = -Infinity;
		for (let i = 0; i < n; i++) {
			const row = new Array<number>(nT);
			for (let t = 0; t < nT; t++) {
				const v = big ? Number(src[i * nT + t]) : (src[i * nT + t] as number);
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
				// Shrink the live range toward its centre, then grow to fit the new
				// extremes (order ported verbatim from the dearpygui viewer).
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

	/** The shared [lo, hi] data range for both axes, with margin + a degenerate
	 * guard. Mirrors `vmin - |vmax|·margin … vmax + |vmax|·margin`. */
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
		// Faint even grid for spatial reference.
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
		// Origin cross — the key reference for a phase portrait — when 0 is in view.
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
		ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
		ctx.fillStyle = 'rgba(208, 208, 208, 0.55)';
		const pad = 4;
		// y-axis: max top-left, min bottom-left.
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(fmtTick(hi), pad, pad);
		ctx.textBaseline = 'bottom';
		ctx.fillText(fmtTick(lo), pad, h - pad);
		// x-axis: max bottom-right (min shares the bottom-left corner with y-min).
		ctx.textAlign = 'right';
		ctx.fillText(fmtTick(hi), w - pad, h - pad);
	}

	function draw(): void {
		if (!ctx || !canvas || !container) return;
		const dpr = window.devicePixelRatio || 1;
		const w = container.clientWidth;
		const h = container.clientHeight;
		// Keep the backing store in step with the element + DPR, then draw in CSS
		// pixels via a scaled transform (crisp text + lines on HiDPI).
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
				const color = PALETTE[k % PALETTE.length];

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

				// "Head" marker at the current (latest) position — reads the
				// trajectory's live tip at a glance.
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
