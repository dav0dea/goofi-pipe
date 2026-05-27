<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import { EEG_LAYOUT } from './eegLayout';
	import {
		buildLayout,
		buildPixelCache,
		solveWeights,
		evaluateField,
		type TopoLayout,
		type PixelCache
	} from './topomapInterp';
	import { onMount, onDestroy } from 'svelte';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	let canvas: HTMLCanvasElement | null = $state(null);
	let resizer: ResizeObserver | null = null;
	let imageData: ImageData | null = null;
	let size = $state({ w: 200, h: 200 });

	let layout: TopoLayout | null = null;
	let pixelCache: PixelCache | null = null;
	let field: Float32Array | null = null;

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	// 17-point piecewise-linear approximation of matplotlib's `coolwarm` sampled
	// at t = 0, 1/16, …, 1. Visually matches the real cmap; the 256-entry LUT
	// below is built at module init from these control points.
	const COOLWARM_STOPS: ReadonlyArray<readonly [number, number, number]> = [
		[59, 76, 192],
		[78, 104, 216],
		[98, 130, 234],
		[119, 154, 247],
		[141, 176, 254],
		[163, 194, 254],
		[185, 208, 249],
		[204, 217, 237],
		[221, 220, 220],
		[236, 211, 197],
		[245, 196, 172],
		[247, 176, 147],
		[244, 152, 122],
		[235, 125, 98],
		[221, 95, 75],
		[202, 59, 55],
		[180, 4, 38]
	];

	const COOLWARM_LUT = (() => {
		const lut = new Uint8Array(256 * 3);
		const segments = COOLWARM_STOPS.length - 1;
		for (let i = 0; i < 256; i++) {
			const t = (i / 255) * segments;
			const lo = Math.min(segments - 1, Math.floor(t));
			const f = t - lo;
			const a = COOLWARM_STOPS[lo];
			const b = COOLWARM_STOPS[lo + 1];
			lut[i * 3 + 0] = Math.round(a[0] + (b[0] - a[0]) * f);
			lut[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * f);
			lut[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * f);
		}
		return lut;
	})();

	function coolwarmIndex(v: number): number {
		// Map v ∈ [-1, 1] (clamped) to LUT index 0..255.
		const t = v <= -1 ? 0 : v >= 1 ? 1 : (v + 1) * 0.5;
		return (t * 255) | 0;
	}

	function drawMessage(ctx: CanvasRenderingContext2D, w: number, h: number, msg: string): void {
		ctx.fillStyle = '#1c2029';
		ctx.fillRect(0, 0, w, h);
		ctx.fillStyle = '#9aa3b3';
		ctx.font = '11px var(--font-mono)';
		ctx.textAlign = 'center';
		ctx.fillText(msg, w / 2, h / 2);
	}

	function drawHeadDecor(
		ctx: CanvasRenderingContext2D,
		w: number,
		h: number,
		channels: Array<[number, number]>
	): void {
		const cx = w / 2;
		const cy = h / 2;
		const radius = Math.min(w, h) * 0.45;
		ctx.strokeStyle = '#c5c8d6';
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.arc(cx, cy, radius, 0, Math.PI * 2);
		ctx.stroke();
		ctx.beginPath();
		ctx.moveTo(cx - 8, cy - radius);
		ctx.lineTo(cx, cy - radius - 10);
		ctx.lineTo(cx + 8, cy - radius);
		ctx.stroke();
		ctx.fillStyle = '#0e1014';
		for (const p of channels) {
			ctx.beginPath();
			ctx.arc(p[0] * w, p[1] * h, 2, 0, Math.PI * 2);
			ctx.fill();
		}
	}

	function paint(arr: ArrayData, channels: string[]): void {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const w = canvas.width;
		const h = canvas.height;

		const vals = arr.values as ArrayLike<number>;
		const knownIdx: number[] = [];
		const knownPos: Array<[number, number]> = [];
		for (let i = 0; i < channels.length; i++) {
			const name = channels[i];
			const pos = EEG_LAYOUT[name] ?? EEG_LAYOUT[name.toUpperCase()];
			if (!pos) continue;
			knownIdx.push(i);
			knownPos.push(pos);
		}

		if (knownPos.length === 0) {
			drawMessage(ctx, w, h, 'no recognized channels');
			return;
		}
		if (knownPos.length < 3) {
			drawMessage(ctx, w, h, 'need ≥ 3 channels for topomap');
			return;
		}

		const layoutKey = knownPos.map((p) => `${p[0]},${p[1]}`).join('|');
		if (!layout || layout.layoutKey !== layoutKey) {
			try {
				layout = buildLayout(knownPos, layoutKey);
			} catch {
				layout = null;
			}
			pixelCache = null;
			field = null;
		}
		if (!layout) {
			drawMessage(ctx, w, h, 'topomap layout failed');
			return;
		}
		if (
			!pixelCache ||
			pixelCache.width !== w ||
			pixelCache.height !== h ||
			pixelCache.layoutKey !== layoutKey
		) {
			pixelCache = buildPixelCache(layout, w, h);
			field = new Float32Array(pixelCache.count);
		}
		if (!field) return;

		const realVals = new Float64Array(knownIdx.length);
		for (let i = 0; i < knownIdx.length; i++) {
			realVals[i] = Number(vals[knownIdx[i]]);
		}

		const weights = solveWeights(layout, realVals);
		evaluateField(layout, pixelCache, weights, field);

		if (!imageData || imageData.width !== w || imageData.height !== h) {
			imageData = ctx.createImageData(w, h);
		}
		const data = imageData.data;
		data.fill(0);
		const offsets = pixelCache.pixelByteOffsets;
		const count = pixelCache.count;
		const lut = COOLWARM_LUT;
		for (let p = 0; p < count; p++) {
			const off = offsets[p];
			const idx = coolwarmIndex(field[p]) * 3;
			data[off] = lut[idx];
			data[off + 1] = lut[idx + 1];
			data[off + 2] = lut[idx + 2];
			data[off + 3] = 255;
		}
		ctx.putImageData(imageData, 0, 0);
		drawHeadDecor(ctx, w, h, knownPos);
	}

	$effect(() => {
		if (!frame || !canvas) return;
		const arr = asArray(frame.data);
		const channels = ((frame.meta?.channels as { dim0?: string[] }) ?? {}).dim0 ?? [];
		if (canvas.width !== size.w) canvas.width = size.w;
		if (canvas.height !== size.h) canvas.height = size.h;
		paint(arr, channels);
	});

	onMount(() => {
		if (!canvas) return;
		const updateSize = () => {
			if (!canvas) return;
			size = { w: canvas.clientWidth, h: canvas.clientHeight };
		};
		updateSize();
		resizer = new ResizeObserver(updateSize);
		resizer.observe(canvas);
	});
	onDestroy(() => resizer?.disconnect());
</script>

<canvas bind:this={canvas}></canvas>

<style>
	canvas {
		width: 100%;
		height: 100%;
		min-height: 100px;
		min-width: 100px;
		display: block;
	}
</style>
