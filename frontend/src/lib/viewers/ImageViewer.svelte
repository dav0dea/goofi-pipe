<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import { onMount } from 'svelte';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	let canvas: HTMLCanvasElement | null = $state(null);
	let lastImgW = 0;
	let lastImgH = 0;

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	function paint(arr: ArrayData): void {
		if (!canvas) return;
		const shape = arr.shape;
		if (shape.length < 2) return;
		// Accept (H, W), (H, W, C). Squeeze leading singletons by stride
		// math — fold the leading dim into the row count when shape.length>3.
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
			lastImgW = w;
			lastImgH = h;
		}

		const out = ctx.createImageData(w, h);
		const src = arr.values as ArrayLike<number>;
		const isFloat = arr.dtype.startsWith('<f') || arr.dtype.startsWith('|f');
		const isU8 = arr.dtype === '|u1' || arr.dtype === '<u1' || arr.dtype === '=u1';
		const totalPx = w * h;
		const dst = out.data;

		function val(i: number): number {
			const v = Number(src[i]);
			if (isU8) return v;
			if (isFloat) return Math.max(0, Math.min(255, Math.round(v * 255)));
			return Math.max(0, Math.min(255, Math.round(v)));
		}

		for (let i = 0; i < totalPx; i++) {
			const o = i * 4;
			if (c === 1) {
				const v = val(i);
				dst[o] = v;
				dst[o + 1] = v;
				dst[o + 2] = v;
				dst[o + 3] = 255;
			} else if (c === 2) {
				const v = val(i * 2);
				const a = isFloat ? val(i * 2 + 1) : src[i * 2 + 1];
				dst[o] = v;
				dst[o + 1] = v;
				dst[o + 2] = v;
				dst[o + 3] = isU8 ? Number(a) : Number(a);
			} else if (c === 3) {
				dst[o] = val(i * 3);
				dst[o + 1] = val(i * 3 + 1);
				dst[o + 2] = val(i * 3 + 2);
				dst[o + 3] = 255;
			} else {
				dst[o] = val(i * 4);
				dst[o + 1] = val(i * 4 + 1);
				dst[o + 2] = val(i * 4 + 2);
				dst[o + 3] = val(i * 4 + 3);
			}
		}
		ctx.putImageData(out, 0, 0);
	}

	$effect(() => {
		if (frame) paint(asArray(frame.data));
	});
	void lastImgW;
	void lastImgH;

	onMount(() => {
		// noop — paint handled by $effect
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
