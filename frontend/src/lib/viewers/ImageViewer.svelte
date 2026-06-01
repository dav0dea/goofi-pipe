<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	let canvas: HTMLCanvasElement | null = $state(null);
	// Reused across frames; reallocated only when the frame size changes, so a
	// steady HD stream doesn't churn a fresh ~8MB ImageData every paint.
	let img: ImageData | null = null;

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	function paint(arr: ArrayData): void {
		if (!canvas) return;
		const shape = arr.shape;
		if (shape.length < 2) return;
		// Accept (H, W) grayscale and (H, W, C) with C in {1,2,3,4}.
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
		// Pick the per-pixel scaler once, not inside the loop.
		const scale: (v: number) => number = isU8
			? (v) => v
			: isFloat
				? (v) => Math.max(0, Math.min(255, Math.round(v * 255)))
				: (v) => Math.max(0, Math.min(255, Math.round(v)));
		const n = w * h;

		// One specialized loop per channel count (dst is a Uint8ClampedArray, so
		// any out-of-range write clamps automatically).
		if (c === 1) {
			for (let i = 0; i < n; i++) {
				const v = scale(Number(src[i]));
				const o = i * 4;
				dst[o] = dst[o + 1] = dst[o + 2] = v;
				dst[o + 3] = 255;
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
		} else if (c === 4) {
			for (let i = 0; i < n; i++) {
				const o = i * 4;
				dst[o] = scale(Number(src[o]));
				dst[o + 1] = scale(Number(src[o + 1]));
				dst[o + 2] = scale(Number(src[o + 2]));
				dst[o + 3] = scale(Number(src[o + 3]));
			}
		} else {
			// c === 2: grayscale + alpha
			for (let i = 0; i < n; i++) {
				const j = i * 2;
				const o = i * 4;
				const v = scale(Number(src[j]));
				dst[o] = dst[o + 1] = dst[o + 2] = v;
				dst[o + 3] = scale(Number(src[j + 1]));
			}
		}
		ctx.putImageData(img, 0, 0);
	}

	$effect(() => {
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
