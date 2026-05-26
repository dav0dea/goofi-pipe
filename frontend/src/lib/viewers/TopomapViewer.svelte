<script lang="ts">
	import type { DataFrame, ArrayData } from '$lib/codec/decode';
	import { EEG_LAYOUT } from './eegLayout';
	import { onMount, onDestroy } from 'svelte';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	let canvas: HTMLCanvasElement | null = $state(null);
	let resizer: ResizeObserver | null = null;
	let imageData: ImageData | null = null;
	let size = $state({ w: 200, h: 200 });

	function asArray(d: DataFrame['data']): ArrayData {
		return d as ArrayData;
	}

	function viridis(t: number): [number, number, number] {
		t = Math.max(0, Math.min(1, t));
		// Quintic approximation of the inferno-ish ramp — close enough for a
		// small EEG topomap, and pure-JS so we don't bundle a colormap lib.
		const r = Math.min(255, Math.max(0, Math.round(255 * Math.sqrt(t))));
		const g = Math.min(255, Math.max(0, Math.round(255 * t * t)));
		const b = Math.min(255, Math.max(0, Math.round(255 * Math.pow(t, 3) * 0.5)));
		return [r, g, b];
	}

	function paint(arr: ArrayData, channels: string[]): void {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const w = canvas.width;
		const h = canvas.height;
		if (!imageData || imageData.width !== w || imageData.height !== h) {
			imageData = ctx.createImageData(w, h);
		}
		const known: { x: number; y: number; v: number }[] = [];
		const vals = arr.values as ArrayLike<number>;
		for (let i = 0; i < channels.length; i++) {
			const name = channels[i];
			const pos = EEG_LAYOUT[name] ?? EEG_LAYOUT[name.toUpperCase()];
			if (!pos) continue;
			known.push({ x: pos[0], y: pos[1], v: Number(vals[i]) });
		}
		if (known.length === 0) {
			ctx.fillStyle = '#1c2029';
			ctx.fillRect(0, 0, w, h);
			ctx.fillStyle = '#9aa3b3';
			ctx.font = '11px var(--font-mono)';
			ctx.textAlign = 'center';
			ctx.fillText('no recognized channels', w / 2, h / 2);
			return;
		}
		let mn = Infinity;
		let mx = -Infinity;
		for (const k of known) {
			if (k.v < mn) mn = k.v;
			if (k.v > mx) mx = k.v;
		}
		const span = mx - mn || 1;

		const data = imageData.data;
		// IDW interpolation onto the canvas. Mask outside the head circle.
		const cx = w / 2;
		const cy = h / 2;
		const radius = Math.min(w, h) * 0.45;
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const off = (y * w + x) * 4;
				const dx = x - cx;
				const dy = y - cy;
				if (Math.sqrt(dx * dx + dy * dy) > radius) {
					data[off + 3] = 0;
					continue;
				}
				const u = x / w;
				const vy = y / h;
				let weight = 0;
				let acc = 0;
				for (const k of known) {
					const ddx = u - k.x;
					const ddy = vy - k.y;
					const dist2 = ddx * ddx + ddy * ddy;
					if (dist2 < 1e-6) {
						acc = k.v;
						weight = 1;
						break;
					}
					const wgt = 1 / dist2;
					acc += wgt * k.v;
					weight += wgt;
				}
				const v = (acc / weight - mn) / span;
				const [r, g, b] = viridis(v);
				data[off] = r;
				data[off + 1] = g;
				data[off + 2] = b;
				data[off + 3] = 255;
			}
		}
		ctx.putImageData(imageData, 0, 0);
		// Head outline
		ctx.strokeStyle = '#c5c8d6';
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.arc(cx, cy, radius, 0, Math.PI * 2);
		ctx.stroke();
		// Nose
		ctx.beginPath();
		ctx.moveTo(cx - 8, cy - radius);
		ctx.lineTo(cx, cy - radius - 10);
		ctx.lineTo(cx + 8, cy - radius);
		ctx.stroke();
		// Channel dots
		ctx.fillStyle = '#0e1014';
		for (const k of known) {
			ctx.beginPath();
			ctx.arc(k.x * w, k.y * h, 2, 0, Math.PI * 2);
			ctx.fill();
		}
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
