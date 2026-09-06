<!-- DrawPad — a canvas you paint on, whose value is a `data:image/png;base64,…` URL. That is a
     STRING like any other, so a drawing crosses the wire, saves into the patch and is read by an
     expression through machinery that was already there; nothing new had to learn about images.

     The stroke is committed on pointer UP, never per move: a data URL runs to tens of kilobytes,
     and one per pointer event would put megabytes a second through the document. -->
<script lang="ts">
	import { Button } from '$lib/ui';
	import { hexOf, hsvOfHex, hsvToRgb, pickedAt } from './drawPad';

	let {
		value,
		onChange,
		disabled = false
	}: { value: string; onChange: (v: string) => void; disabled?: boolean } = $props();

	/** The bitmap's own size, independent of the widget's box: resizing the widget rescales the
	    picture rather than cropping it. */
	const SIZE = 512;

	let canvas = $state<HTMLCanvasElement | null>(null);
	let wheel = $state<HTMLCanvasElement | null>(null);
	let hue = $state(210);
	let sat = $state(0.85);
	let val = $state(1);
	let size = $state(12);
	let feather = $state(0);
	let erasing = $state(false);
	let drawing = false;
	let last: { x: number; y: number } | null = null;
	/** What we last handed to `onChange`, so our own echo does not reload the canvas under the hand
	    that is drawing on it. */
	let mine = '';

	const hex = $derived(hexOf(hue, sat, val));

	function setHex(h: string): void {
		const next = hsvOfHex(h, hue);
		if (!next) return;
		hue = next.h;
		sat = next.s;
		val = next.v;
	}

	/** The hue/saturation disc, painted once: angle is hue, radius is saturation. Brightness is the
	    slider's job, so moving it does not repaint the wheel. */
	function paintWheel(el: HTMLCanvasElement): void {
		const ctx = el.getContext('2d');
		if (!ctx) return;
		const side = el.width;
		const r = side / 2;
		const img = ctx.createImageData(side, side);
		for (let y = 0; y < side; y++) {
			for (let x = 0; x < side; x++) {
				const dx = x - r;
				const dy = y - r;
				const dist = Math.hypot(dx, dy);
				if (dist > r) continue;
				const i = (y * side + x) * 4;
				const h = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
				const [cr, cg, cb] = hsvToRgb(h, Math.min(1, dist / r), 1);
				img.data[i] = cr;
				img.data[i + 1] = cg;
				img.data[i + 2] = cb;
				// A pixel of feather at the rim, so the disc is not a staircase.
				img.data[i + 3] = Math.round(255 * Math.min(1, r - dist));
			}
		}
		ctx.putImageData(img, 0, 0);
	}
	$effect(() => {
		if (wheel) paintWheel(wheel);
	});

	function pickWheel(e: PointerEvent): void {
		if (!wheel || disabled) return;
		const box = wheel.getBoundingClientRect();
		const r = box.width / 2;
		const picked = pickedAt(e.clientX - box.left - r, e.clientY - box.top - r, r);
		hue = picked.h;
		sat = picked.s;
	}

	/** Load `value` in whenever it is someone else's — a patch load, another viewer, an agent. */
	$effect(() => {
		const url = value;
		const el = canvas;
		if (!el || url === mine) return;
		const ctx = el.getContext('2d');
		if (!ctx) return;
		if (!url) {
			ctx.clearRect(0, 0, SIZE, SIZE);
			return;
		}
		const img = new Image();
		img.onload = () => {
			ctx.clearRect(0, 0, SIZE, SIZE);
			ctx.drawImage(img, 0, 0, SIZE, SIZE);
		};
		img.src = url;
	});

	function at(e: PointerEvent): { x: number; y: number } | null {
		if (!canvas) return null;
		const box = canvas.getBoundingClientRect();
		return {
			x: ((e.clientX - box.left) / box.width) * SIZE,
			y: ((e.clientY - box.top) / box.height) * SIZE
		};
	}

	function stroke(from: { x: number; y: number }, to: { x: number; y: number }): void {
		const ctx = canvas?.getContext('2d');
		if (!ctx) return;
		ctx.save();
		ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
		ctx.strokeStyle = hex;
		ctx.lineWidth = size;
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
		// `filter` is what makes a soft brush soft. Where it is unsupported the stroke is simply
		// hard-edged, which is a lesser brush and never a broken one.
		if (feather > 0) ctx.filter = `blur(${feather}px)`;
		ctx.beginPath();
		ctx.moveTo(from.x, from.y);
		ctx.lineTo(to.x, to.y);
		ctx.stroke();
		ctx.restore();
	}

	function commit(): void {
		if (!canvas) return;
		mine = canvas.toDataURL('image/png');
		onChange(mine);
	}

	function down(e: PointerEvent): void {
		if (disabled) return;
		const p = at(e);
		if (!p) return;
		drawing = true;
		last = p;
		(e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
		// A tap is a dot, so the shortest stroke still leaves a mark.
		stroke(p, { x: p.x + 0.01, y: p.y });
		e.preventDefault();
	}
	function move(e: PointerEvent): void {
		if (!drawing || !last) return;
		const p = at(e);
		if (!p) return;
		stroke(last, p);
		last = p;
	}
	function up(): void {
		if (!drawing) return;
		drawing = false;
		last = null;
		commit();
	}
	function clear(): void {
		const ctx = canvas?.getContext('2d');
		if (!ctx || disabled) return;
		ctx.clearRect(0, 0, SIZE, SIZE);
		commit();
	}
</script>

<div class="pad" data-testid="draw-pad">
	<div class="tools">
		<canvas
			bind:this={wheel}
			class="wheel"
			width="44"
			height="44"
			data-testid="draw-wheel"
			title="Hue around, saturation outward"
			onpointerdown={(e) => {
				(e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
				pickWheel(e);
			}}
			onpointermove={(e) => {
				if (e.buttons) pickWheel(e);
			}}
		></canvas>
		<label class="swatch" title="Pick an exact colour" style={`--ink: ${hex}`}>
			<input
				type="color"
				value={hex}
				{disabled}
				data-testid="draw-colour"
				oninput={(e) => setHex((e.currentTarget as HTMLInputElement).value)}
			/>
		</label>
		<label class="dial" title={`Brightness ${Math.round(val * 100)}%`}>
			<span>lum</span>
			<input type="range" min="0" max="1" step="0.01" bind:value={val} {disabled} />
		</label>
		<label class="dial" title={`Brush ${size} px across`}>
			<span>size</span>
			<input
				type="range"
				min="1"
				max="96"
				step="1"
				bind:value={size}
				{disabled}
				data-testid="draw-size"
			/>
		</label>
		<label class="dial" title={`Feather ${feather} px`}>
			<span>soft</span>
			<input
				type="range"
				min="0"
				max="32"
				step="1"
				bind:value={feather}
				{disabled}
				data-testid="draw-feather"
			/>
		</label>
		<Button
			size="sm"
			variant={erasing ? 'primary' : 'ghost'}
			title="Paint transparency instead of colour"
			{disabled}
			data-testid="draw-eraser"
			onclick={() => (erasing = !erasing)}>erase</Button
		>
		<Button
			size="sm"
			variant="ghost"
			title="Clear the drawing"
			{disabled}
			data-testid="draw-clear"
			onclick={clear}>clear</Button
		>
	</div>
	<canvas
		bind:this={canvas}
		class="sheet"
		width={SIZE}
		height={SIZE}
		data-testid="draw-canvas"
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={up}
	></canvas>
</div>

<style>
	.pad {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		width: 100%;
		height: 100%;
		min-height: 0;
	}
	.tools {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		flex-wrap: wrap;
	}
	.wheel {
		width: 22px;
		height: 22px;
		border-radius: 50%;
		cursor: crosshair;
		touch-action: none;
		flex: none;
	}
	.swatch {
		width: 18px;
		height: 18px;
		border-radius: var(--radius-sm);
		background: var(--ink);
		border: 1px solid var(--border);
		overflow: hidden;
		cursor: pointer;
		flex: none;
	}
	.swatch input {
		opacity: 0;
		width: 100%;
		height: 100%;
		cursor: pointer;
	}
	.dial {
		display: flex;
		align-items: center;
		gap: var(--space-1);
		min-width: 0;
		flex: 1 1 44px;
		color: var(--text-dim);
	}
	.dial span {
		font-size: var(--fs-micro);
		letter-spacing: 0.02em;
		flex: none;
	}
	.dial input {
		width: 100%;
		min-width: 32px;
		accent-color: var(--accent);
	}
	.sheet {
		flex: 1 1 auto;
		width: 100%;
		min-height: 0;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-1);
		cursor: crosshair;
		touch-action: none;
	}
</style>
