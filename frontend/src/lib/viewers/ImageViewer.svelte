<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { makeLUT } from './colormaps';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	// Colormap + value range apply to single-channel (grayscale) frames; true RGB
	// frames are drawn as-is.
	const colormap = $derived(String(settings.colormap ?? 'gray'));
	const autoRange = $derived(settings.auto !== false);
	const vmin = $derived(Number(settings.vmin ?? 0));
	const vmax = $derived(Number(settings.vmax ?? 1));

	let canvas: HTMLCanvasElement | null = $state(null);
	// Reused across frames; reallocated only when the frame size changes, so a
	// steady HD stream doesn't churn a fresh ~8MB ImageData every paint.
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
	function grayRange(src: ArrayLike<number | bigint>, n: number, stride: number): [number, number] {
		if (!autoRange) return [vmin, vmax > vmin ? vmax : vmin + 1e-9];
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

	function paint(arr: ArrayData): void {
		if (!canvas) return;
		const shape = arr.shape;
		if (shape.length < 2) return;
		let h: number, w: number, c: number;
		if (shape.length === 2) {
			[h, w] = shape;
			c = 1;
		} else if (shape.length === 3) {
			[h, w, c] = shape;
			if (![1, 2, 3, 4].includes(c)) return;
		} else {
			return;
		}

		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		if (canvas.width !== w || canvas.height !== h) {
			canvas.width = w;
			canvas.height = h;
			img = null;
		}
		if (!img || img.width !== w || img.height !== h) img = ctx.createImageData(w, h);

		const dst = img.data;
		const src = arr.values;
		const isFloat =
			arr.dtype.startsWith('<f') || arr.dtype.startsWith('|f') || arr.dtype.startsWith('=f');
		const isU8 = arr.dtype === '|u1' || arr.dtype === '<u1' || arr.dtype === '=u1';
		const scale: (v: number) => number = isU8
			? (v) => v
			: isFloat
				? (v) => Math.max(0, Math.min(255, Math.round(v * 255)))
				: (v) => Math.max(0, Math.min(255, Math.round(v)));
		const n = w * h;

		if (c === 1 || c === 2) {
			// Grayscale → colormap, normalized to the chosen value range.
			const stride = c;
			const L = lutFor(colormap);
			const [lo, hi] = grayRange(src, n, stride);
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
		void [colormap, autoRange, vmin, vmax];
		if (frame) paint(asArray(frame.data));
	});
</script>

<canvas bind:this={canvas}></canvas>

<style>
	canvas {
		width: 100%;
		height: 100%;
		image-rendering: pixelated;
		background: #000;
		object-fit: contain;
	}
</style>
