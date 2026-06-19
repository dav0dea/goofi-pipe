<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { makeLUT } from './colormaps';
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

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const colormap = $derived(String(settings.colormap ?? 'coolwarm'));
	const autoRange = $derived(settings.auto !== false);
	const vmin = $derived(Number(settings.vmin ?? -1));
	const vmax = $derived(Number(settings.vmax ?? 1));
	const contours = $derived(Boolean(settings.contours));

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

	// The active colormap LUT, rebuilt only when the colormap setting changes.
	let lut = makeLUT('coolwarm');
	let lutName = 'coolwarm';
	function lutFor(name: string): Uint8Array {
		if (name !== lutName) {
			lut = makeLUT(name);
			lutName = name;
		}
		return lut;
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
		const L = lutFor(colormap);

		// Value range: scanned from the field when auto, else the manual [vmin,
		// vmax]. Contours posterize the normalized value into bands so their
		// boundaries read as iso-lines.
		let lo = vmin;
		let hi = vmax;
		if (autoRange) {
			lo = Infinity;
			hi = -Infinity;
			for (let p = 0; p < count; p++) {
				const v = field[p];
				if (v < lo) lo = v;
				if (v > hi) hi = v;
			}
			if (!(hi > lo)) {
				lo = -1;
				hi = 1;
			}
		}
		const span = hi - lo || 1;
		for (let p = 0; p < count; p++) {
			const off = offsets[p];
			let t = (field[p] - lo) / span;
			t = t < 0 ? 0 : t > 1 ? 1 : t;
			if (contours) t = Math.round(t * 10) / 10;
			const idx = ((t * 255) | 0) * 3;
			data[off] = L[idx];
			data[off + 1] = L[idx + 1];
			data[off + 2] = L[idx + 2];
			data[off + 3] = 255;
		}
		ctx.putImageData(imageData, 0, 0);
		drawHeadDecor(ctx, w, h, knownPos);
	}

	$effect(() => {
		// Repaint on a new frame or any colormap / range / contour change.
		void [colormap, autoRange, vmin, vmax, contours];
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
