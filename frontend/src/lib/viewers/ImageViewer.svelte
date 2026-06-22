<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { makeLUT } from './colormaps';
	import { GLImageRenderer, glSupports } from './imageGL';
	import { manualGrayDomain, viewInfo } from './viewMeta';
	import { onMount, onDestroy } from 'svelte';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	// The bridge image adapter ships grayscale as uint8 normalized from the source
	// float range, carried in meta.__view__.range — used to map a manual float window.
	const view = $derived(viewInfo(frame.meta));

	// Colormap + value range apply to single-channel (grayscale) frames; true RGB
	// frames are drawn as-is.
	const colormap = $derived(String(settings.colormap ?? 'gray'));
	const autoRange = $derived(settings.auto !== false);
	const vmin = $derived(Number(settings.vmin ?? 0));
	const vmax = $derived(Number(settings.vmax ?? 1));

	// Two canvases: a WebGL2 one (the fast path — HD video, float heatmaps) and a
	// 2D one (fallback for the rare float-RGB / gray+alpha frames, or when WebGL2
	// is unavailable). A canvas can hold only one context type, so we pick per
	// frame and show the matching canvas.
	let glCanvas: HTMLCanvasElement | null = $state(null);
	let canvas2d: HTMLCanvasElement | null = $state(null);
	let useGl = $state(false);
	let renderer: GLImageRenderer | null = null;
	let cssW = 300;
	let cssH = 150;
	let ro: ResizeObserver | null = null;

	onMount(() => {
		if (glCanvas) {
			renderer = GLImageRenderer.tryCreate(glCanvas);
			ro = new ResizeObserver((entries) => {
				for (const e of entries) {
					cssW = e.contentRect.width || cssW;
					cssH = e.contentRect.height || cssH;
				}
			});
			ro.observe(glCanvas);
		}
	});
	onDestroy(() => {
		ro?.disconnect();
		renderer?.dispose();
	});

	// Reused 2D ImageData; reallocated only when the frame size changes.
	let img: ImageData | null = null;

	let lut = makeLUT('gray');
	let lutName = 'gray';
	function lutFor(name: string): Uint8Array {
		if (name !== lutName) {
			lut = makeLUT(name);
			lutName = name;
		}
		return lut;
	}

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	/** Gray channel range — scanned from the data when auto, else the manual
	 * [vmin, vmax]. */
	function grayRange(
		src: ArrayLike<number | bigint>,
		n: number,
		stride: number,
		dtype: string
	): [number, number] {
		if (!autoRange) {
			const [lo, hi] = manualGrayDomain(view, vmin, vmax, dtype);
			return [lo, hi > lo ? hi : lo + 1e-9];
		}
		let lo = Infinity;
		let hi = -Infinity;
		for (let i = 0; i < n; i++) {
			const v = Number(src[i * stride]);
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
		if (!Number.isFinite(lo) || hi <= lo) return [lo, lo + 1];
		return [lo, hi];
	}

	function shapeOf(arr: ArrayData): [number, number, number] | null {
		const shape = arr.shape;
		if (shape.length === 2) return [shape[0], shape[1], 1];
		if (shape.length === 3) {
			const c = shape[2];
			if (![1, 2, 3, 4].includes(c)) return null;
			return [shape[0], shape[1], c];
		}
		return null;
	}

	function isFloatDtype(dtype: string): boolean {
		return dtype.startsWith('<f') || dtype.startsWith('|f') || dtype.startsWith('=f');
	}

	function render(arr: ArrayData): void {
		const dims = shapeOf(arr);
		if (!dims) return;
		const [h, w, c] = dims;
		const isFloat = isFloatDtype(arr.dtype);

		if (renderer && glSupports(c, arr.dtype)) {
			useGl = true;
			let lo = 0;
			let hi = 1;
			if (c === 1) [lo, hi] = grayRange(arr.values, w * h, 1, arr.dtype);
			renderer.render(arr.values, w, h, c, isFloat, {
				colormap,
				lut: lutFor(colormap),
				lo,
				hi,
				cssW,
				cssH
			});
			return;
		}
		useGl = false;
		paint2d(arr, h, w, c, isFloat);
	}

	function paint2d(arr: ArrayData, h: number, w: number, c: number, isFloat: boolean): void {
		if (!canvas2d) return;
		const ctx = canvas2d.getContext('2d');
		if (!ctx) return;
		if (canvas2d.width !== w || canvas2d.height !== h) {
			canvas2d.width = w;
			canvas2d.height = h;
			img = null;
		}
		if (!img || img.width !== w || img.height !== h) img = ctx.createImageData(w, h);

		const dst = img.data;
		const src = arr.values;
		const isU8 = arr.dtype === '|u1' || arr.dtype === '<u1' || arr.dtype === '=u1';
		const scale: (v: number) => number = isU8
			? (v) => v
			: isFloat
				? (v) => Math.max(0, Math.min(255, Math.round(v * 255)))
				: (v) => Math.max(0, Math.min(255, Math.round(v)));
		const n = w * h;

		if (c === 1 || c === 2) {
			const stride = c;
			const L = lutFor(colormap);
			const [lo, hi] = grayRange(src, n, stride, arr.dtype);
			const span = hi - lo || 1;
			for (let i = 0; i < n; i++) {
				const t = (Number(src[i * stride]) - lo) / span;
				const idx = (Math.max(0, Math.min(1, t)) * 255) | 0;
				const li = idx * 3;
				const o = i * 4;
				dst[o] = L[li];
				dst[o + 1] = L[li + 1];
				dst[o + 2] = L[li + 2];
				dst[o + 3] = c === 2 ? scale(Number(src[i * 2 + 1])) : 255;
			}
		} else if (c === 3) {
			for (let i = 0; i < n; i++) {
				const j = i * 3;
				const o = i * 4;
				dst[o] = scale(Number(src[j]));
				dst[o + 1] = scale(Number(src[j + 1]));
				dst[o + 2] = scale(Number(src[j + 2]));
				dst[o + 3] = 255;
			}
		} else {
			for (let i = 0; i < n; i++) {
				const o = i * 4;
				dst[o] = scale(Number(src[o]));
				dst[o + 1] = scale(Number(src[o + 1]));
				dst[o + 2] = scale(Number(src[o + 2]));
				dst[o + 3] = scale(Number(src[o + 3]));
			}
		}
		ctx.putImageData(img, 0, 0);
	}

	$effect(() => {
		// Repaint on a new frame or any colormap / range change.
		void [colormap, autoRange, vmin, vmax, cssW, cssH];
		if (frame) render(asArray(frame.data));
	});
</script>

<canvas bind:this={glCanvas} class:hidden={!useGl}></canvas>
<canvas bind:this={canvas2d} class:hidden={useGl}></canvas>

<style>
	canvas {
		width: 100%;
		height: 100%;
		image-rendering: pixelated;
		background: #000;
		object-fit: contain;
	}
	canvas.hidden {
		display: none;
	}
</style>
