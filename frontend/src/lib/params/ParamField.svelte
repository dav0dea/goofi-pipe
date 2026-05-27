<script lang="ts">
	import type { ParamDescriptor } from '$lib/api/types';

	type Props = {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
	};
	const { paramName, descriptor, onCommit }: Props = $props();

	let local = $state<unknown>(descriptor.value);
	// Suppress backend echoes while the user is actively editing — without
	// this, every committed slider/text edit would re-set `local` to the
	// echoed value mid-typing.
	let editing = $state(false);

	$effect(() => {
		if (!editing) local = descriptor.value;
	});

	function commit(v: unknown): void {
		local = v;
		onCommit(v);
	}

	// Auto-extend slider bounds when the live value lies outside [vmin, vmax]
	// (goofi3 does the same — without it a value of 5.0 on a [0, 1] slider
	// would render at the right edge and clip on edit).
	const numeric = $derived(
		descriptor.type === 'float' || descriptor.type === 'int' ? descriptor : null
	);
	const lo = $derived(numeric ? Math.min(numeric.vmin, Number(local ?? 0)) : 0);
	const hi = $derived(numeric ? Math.max(numeric.vmax, Number(local ?? 0)) : 1);
	const step = $derived(
		numeric ? (numeric.type === 'int' ? 1 : Math.max((hi - lo) / 200, 1e-6)) : 1
	);

	function fmtBound(v: number): string {
		if (!Number.isFinite(v)) return '';
		if (numeric?.type === 'int') return String(Math.round(v));
		const abs = Math.abs(v);
		if (abs === 0) return '0';
		if (abs >= 10000 || abs < 0.01) return v.toExponential(1);
		if (abs >= 100) return v.toFixed(0);
		if (abs >= 1) return v.toFixed(2);
		return v.toFixed(3);
	}

	// Drag-on-label scrub. Active on numeric params: pointerdown on the
	// label captures the pointer and converts horizontal motion into value
	// changes. Works for mouse, touchpad, and touch (single pointer events
	// are unified by the browser). Shift = fine (0.1×), Alt = coarse (10×).
	let scrub: { x: number; v: number; pointerId: number } | null = null;

	function startScrub(e: PointerEvent): void {
		if (!numeric) return;
		// Prevent text-selection on the label and any default touch panning.
		e.preventDefault();
		const target = e.currentTarget as HTMLElement;
		try {
			target.setPointerCapture(e.pointerId);
		} catch {
			// Older browsers / some touch devices may reject — drag still
			// works via the window-level listeners below as a fallback.
		}
		editing = true;
		scrub = { x: e.clientX, v: Number(local ?? 0), pointerId: e.pointerId };
	}

	function moveScrub(e: PointerEvent): void {
		if (!scrub || !numeric) return;
		const dx = e.clientX - scrub.x;
		// Sensitivity: how much value per pixel. Bounded range → 1 sweep
		// across ~250px covers the declared range; unbounded int → 1 unit
		// per 4px. Both feel "tight enough to be useful, loose enough to
		// hit any value with a long drag."
		const span = Math.max(hi - lo, 1e-9);
		let sens = numeric.type === 'int' ? 1 / 4 : span / 250;
		if (e.shiftKey) sens *= 0.1;
		if (e.altKey) sens *= 10;
		let v = scrub.v + dx * sens;
		if (numeric.type === 'int') v = Math.round(v);
		if (Number.isFinite(v)) commit(v);
	}

	function endScrub(e: PointerEvent): void {
		if (!scrub) return;
		const target = e.currentTarget as HTMLElement;
		try {
			if (target.hasPointerCapture(scrub.pointerId)) {
				target.releasePointerCapture(scrub.pointerId);
			}
		} catch {
			/* ignore */
		}
		scrub = null;
		editing = false;
	}
</script>

<div class="field" title={descriptor.doc ?? ''}>
	<div class="top-row">
		{#if numeric}
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<span
				class="label scrubbable"
				role="slider"
				aria-valuemin={numeric.vmin}
				aria-valuemax={numeric.vmax}
				aria-valuenow={Number(local)}
				aria-label={paramName}
				tabindex="0"
				onpointerdown={startScrub}
				onpointermove={moveScrub}
				onpointerup={endScrub}
				onpointercancel={endScrub}
				data-testid="param-label-scrub"
			>
				{paramName}
			</span>
			<input
				class="num"
				type="number"
				step={descriptor.type === 'int' ? 1 : 'any'}
				value={Number(local)}
				onfocus={() => (editing = true)}
				onblur={() => (editing = false)}
				onkeydown={(e) => {
					if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
				}}
				oninput={(e) => {
					const raw = (e.currentTarget as HTMLInputElement).value;
					if (raw === '' || raw === '-' || raw === '.') return;
					const v = Number(raw);
					if (Number.isFinite(v)) commit(v);
				}}
			/>
		{:else}
			<span class="label">{paramName}</span>
		{/if}
	</div>

	{#if numeric}
		<div class="slider-row">
			<span class="bound" aria-hidden="true">{fmtBound(lo)}</span>
			<input
				type="range"
				min={lo}
				max={hi}
				{step}
				value={Number(local)}
				onpointerdown={() => (editing = true)}
				onpointerup={() => (editing = false)}
				oninput={(e) => {
					const v = Number((e.currentTarget as HTMLInputElement).value);
					if (Number.isFinite(v)) commit(v);
				}}
			/>
			<span class="bound" aria-hidden="true">{fmtBound(hi)}</span>
		</div>
	{:else if descriptor.type === 'bool'}
		<div class="control">
			{#if descriptor.trigger}
				<button class="trigger" onclick={() => commit(true)}>{paramName}</button>
			{:else}
				<label class="switch">
					<input
						type="checkbox"
						checked={Boolean(local)}
						onchange={(e) => commit((e.currentTarget as HTMLInputElement).checked)}
					/>
					<span class="track"></span>
				</label>
				<span class="bool-label">{local ? 'on' : 'off'}</span>
			{/if}
		</div>
	{:else if descriptor.type === 'string'}
		<div class="control">
			{#if descriptor.options && descriptor.options.length > 0}
				<select
					value={String(local ?? '')}
					onchange={(e) => commit((e.currentTarget as HTMLSelectElement).value)}
				>
					{#each descriptor.options as opt}
						<option value={opt}>{opt}</option>
					{/each}
				</select>
			{:else}
				<input
					type="text"
					value={String(local ?? '')}
					onfocus={() => (editing = true)}
					onblur={() => (editing = false)}
					onkeydown={(e) => {
						if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
					}}
					oninput={(e) => commit((e.currentTarget as HTMLInputElement).value)}
				/>
			{/if}
		</div>
	{:else}
		<div class="control">
			<code>{JSON.stringify(local)}</code>
		</div>
	{/if}
</div>

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 11px;
	}
	.top-row {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	.label {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		text-transform: lowercase;
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		letter-spacing: 0.02em;
	}
	.label.scrubbable {
		/* ew-resize signals the drag affordance to mouse users; touch users
		   discover it via the standard "drag the label name" gesture. */
		cursor: ew-resize;
		user-select: none;
		touch-action: none;
		padding: 4px 0;
		margin: -4px 0;
		/* Slight underline-on-hover hints at interactivity without yelling
		   about it. */
		border-bottom: 1px dashed transparent;
		transition: border-color 100ms ease, color 100ms ease;
	}
	.label.scrubbable:hover {
		color: var(--text);
		border-bottom-color: color-mix(in srgb, var(--accent) 40%, transparent);
	}
	.label.scrubbable:focus {
		outline: none;
		color: var(--text);
		border-bottom-color: var(--accent);
	}
	.slider-row {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.bound {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 9px;
		flex-shrink: 0;
		font-variant-numeric: tabular-nums;
		min-width: 12px;
		text-align: center;
	}
	.control {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	input[type='range'] {
		flex-grow: 1;
		min-width: 0;
		accent-color: var(--accent);
		background: transparent;
		padding: 0;
		border: none;
		height: 22px;
		/* Touch: disable double-tap zoom while interacting with a slider. */
		touch-action: pan-y;
	}
	/* Larger thumb for touch (and crisper on mouse). The native default is
	   ~10-12px on most platforms — too small for a finger. */
	input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--accent);
		border: 2px solid var(--bg-elev-1);
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
		cursor: pointer;
	}
	input[type='range']::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--accent);
		border: 2px solid var(--bg-elev-1);
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
		cursor: pointer;
	}
	input[type='range']::-webkit-slider-runnable-track {
		height: 3px;
		background: var(--border);
		border-radius: 2px;
	}
	input[type='range']::-moz-range-track {
		height: 3px;
		background: var(--border);
		border-radius: 2px;
	}
	.num {
		width: 76px;
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	input[type='text'],
	select {
		flex-grow: 1;
		min-width: 0;
	}
	.trigger {
		width: 100%;
		padding: 8px 12px;
		background: color-mix(in srgb, var(--accent) 22%, transparent);
		border-color: var(--accent);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.trigger:hover:not(:disabled) {
		background: var(--accent);
		color: #0a0c10;
	}
	.switch {
		position: relative;
		width: 38px;
		height: 22px;
		display: inline-block;
		flex-shrink: 0;
	}
	.switch input {
		/* Visual is .track; the input itself is invisible but expanded
		   beyond the track on every side for a comfortable touch target. */
		opacity: 0;
		position: absolute;
		inset: -6px;
		cursor: pointer;
		margin: 0;
	}
	.track {
		position: absolute;
		inset: 0;
		background: var(--bg-elev-3);
		border: 1px solid var(--border);
		border-radius: 999px;
		transition: background 80ms ease;
		pointer-events: none;
	}
	.track::before {
		content: '';
		position: absolute;
		left: 2px;
		top: 1px;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--text-dim);
		transition: transform 120ms ease;
	}
	.switch input:checked ~ .track {
		background: color-mix(in srgb, var(--accent) 35%, transparent);
		border-color: var(--accent);
	}
	.switch input:checked ~ .track::before {
		transform: translateX(15px);
		background: var(--accent);
	}
	.bool-label {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 10px;
		letter-spacing: 0.02em;
	}
	code {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-faint);
	}
</style>
