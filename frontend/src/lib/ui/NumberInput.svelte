<!-- NumberInput — a dumb numeric control: `value` in, `onChange` out, committed on blur / Enter,
     with arrow steps and optional drag-to-scrub. `type=text` because `type=number` reports "" for
     an in-progress "5." and would drop the point. -->
<script lang="ts">
	import { untrack, onDestroy } from 'svelte';
	import type { HTMLInputAttributes } from 'svelte/elements';
	import { useLiveValue } from './liveValue.svelte';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		min,
		max,
		step = 1,
		scrub = false,
		class: klass = '',
		...rest
	}: Omit<HTMLInputAttributes, 'value' | 'type' | 'inputmode' | 'min' | 'max' | 'step' | 'oninput'> & {
		value: number;
		onChange: (v: number) => void;
		min?: number;
		max?: number;
		step?: number;
		/** Enable horizontal drag-to-scrub on the input (a click still focuses to type). */
		scrub?: boolean;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	const live = useLiveValue<number>(
		() => value,
		(v) => onChange(v)
	);

	// Re-synced while idle only, so typing "1." is never rewritten under the cursor.
	let text = $state(untrack(() => fmt(value)));
	$effect(() => {
		if (!live.editing) text = fmt(live.value);
	});

	function fmt(n: number): string {
		return Number.isFinite(n) ? String(n) : '';
	}
	function clamp(n: number): number {
		if (min !== undefined) n = Math.max(min, n);
		if (max !== undefined) n = Math.min(max, n);
		return n;
	}
	/** `n` back on the step's decimal grid; only the incremental paths, never a typed value. */
	function snap(n: number): number {
		const decimals = (String(step).split('.')[1] ?? '').length;
		return decimals ? Number(n.toFixed(decimals)) : n;
	}

	// Each press is a complete gesture, so it commits at once, from the buffer and not the prop.
	function stepBy(e: KeyboardEvent, dir: 1 | -1): void {
		e.preventDefault();
		const typed = Number(text.trim());
		const base = text.trim() !== '' && Number.isFinite(typed) ? typed : live.value;
		const v = clamp(snap(base + dir * step));
		text = fmt(v);
		live.commit(v);
	}

	function commitText(): void {
		if (scrubbing) return; // a scrub-induced blur is not a typed commit
		const raw = text.trim();
		const n = Number(raw);
		if (raw !== '' && Number.isFinite(n)) {
			const v = clamp(n);
			text = fmt(v);
			live.commit(v);
		}
		live.end();
	}

	// Native focus is suppressed on press; click-vs-scrub is decided on move.
	let scrubbing = $state(false);
	// Non-null only while a scrub is in flight, so a cancel or an unmount cannot strand the latch.
	let teardownScrub: (() => void) | null = null;
	function onScrubDown(e: PointerEvent): void {
		if (!scrub || e.button !== 0) return;
		e.preventDefault();
		const el = e.currentTarget as HTMLInputElement;
		const startX = e.clientX;
		const startVal = live.value;
		let active = false;
		// Captured so `up` still lands on `el` when the pointer is released outside the window.
		el.setPointerCapture(e.pointerId);
		const move = (ev: PointerEvent) => {
			if (ev.buttons === 0) return; // released off-window
			const dx = ev.clientX - startX;
			if (!active && Math.abs(dx) < 3) return;
			if (!active) {
				active = true;
				scrubbing = true;
				live.begin();
			}
			const v = clamp(snap(startVal + Math.round(dx / 2) * step));
			text = fmt(v);
			live.commit(v);
		};
		const detach = () => {
			el.removeEventListener('pointermove', move);
			el.removeEventListener('pointerup', up);
			el.removeEventListener('pointercancel', cancel);
			teardownScrub = null;
		};
		const up = () => {
			detach();
			if (active) {
				scrubbing = false;
				// Release the latch only once focus has left, or the idle effect clobbers typed text.
				if (document.activeElement !== el) live.end();
			} else {
				el.focus(); // a click, not a drag → focus to type
			}
		};
		// An aborted gesture is not a click, so this never focuses.
		const cancel = () => {
			detach();
			if (active) {
				scrubbing = false;
				live.end();
			}
		};
		teardownScrub = cancel;
		el.addEventListener('pointermove', move);
		el.addEventListener('pointerup', up);
		el.addEventListener('pointercancel', cancel);
	}
	onDestroy(() => teardownScrub?.());
</script>

<input
	{...rest}
	id={fieldId}
	type="text"
	inputmode="decimal"
	class={`ui-number ${scrub ? 'ui-number-scrub' : ''} ${klass}`.trim()}
	value={text}
	onfocus={() => live.begin()}
	onblur={commitText}
	onpointerdown={onScrubDown}
	onkeydown={(e) => {
		if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
		else if (e.key === 'ArrowUp') stepBy(e, 1);
		else if (e.key === 'ArrowDown') stepBy(e, -1);
	}}
	oninput={(e) => (text = (e.currentTarget as HTMLInputElement).value)}
/>

<style>
	.ui-number {
		width: var(--number-width, 6rem);
		text-align: right;
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	/* `pan-y` claims the horizontal drag and leaves a vertical scroll to the panel. */
	.ui-number-scrub {
		cursor: ew-resize;
		touch-action: pan-y;
	}
	.ui-number:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
</style>
