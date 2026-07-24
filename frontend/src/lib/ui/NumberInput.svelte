<!--
  NumberInput — a dumb numeric control (spec §2.2): `value` in, `onChange` out. Commits on blur /
  Enter (never per-keystroke — so a typed value doesn't echo-jump under the cursor), with optional
  drag-to-scrub. It opts into the shared `useLiveValue` latch for echo suppression rather than
  re-implementing it.

  A `type=text` + `inputmode=decimal` input (not `type=number`) so the raw string is always readable
  while typing — `type=number` reports "" for an in-progress "5." and would drop the point. The
  visible string is buffered locally; it is re-synced from the (echo-suppressed) live value only
  while idle, so keystrokes are never rewritten. Tabular figures keep digits from jittering. `class`
  merged, `data-testid` (and any other input attribute) forwarded via `...rest`.
-->
<script lang="ts">
	import { untrack } from 'svelte';
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
		disabled = false,
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

	const fieldId = claimFieldControlId();
	const live = useLiveValue<number>(
		() => value,
		(v) => onChange(v)
	);

	// The visible string buffer. Re-synced from the echo-suppressed live value while idle, left
	// untouched while editing so typing "1." is never rewritten out from under the cursor.
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

	// Commit the buffer on blur / Enter. An empty or unparseable buffer commits nothing; ending the
	// edit lets the idle effect re-sync `text` back to the source.
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

	// --- drag-to-scrub ---
	// Suppress native focus on press; decide click-vs-scrub on move. A click (no drag past the
	// threshold) focuses to type; a drag scrubs the value live and never focuses.
	let scrubbing = $state(false);
	function onScrubDown(e: PointerEvent): void {
		if (!scrub || disabled || e.button !== 0) return;
		e.preventDefault();
		const el = e.currentTarget as HTMLInputElement;
		const startX = e.clientX;
		const startVal = live.value;
		let active = false;
		const move = (ev: PointerEvent) => {
			const dx = ev.clientX - startX;
			if (!active && Math.abs(dx) < 3) return;
			if (!active) {
				active = true;
				scrubbing = true;
				live.begin();
			}
			const v = clamp(startVal + Math.round(dx / 2) * step);
			text = fmt(v);
			live.commit(v);
		};
		const up = () => {
			window.removeEventListener('pointermove', move);
			window.removeEventListener('pointerup', up);
			if (active) {
				scrubbing = false;
				live.end();
			} else {
				el.focus(); // a click, not a drag → focus to type
			}
		};
		window.addEventListener('pointermove', move);
		window.addEventListener('pointerup', up);
	}
</script>

<input
	{...rest}
	id={fieldId}
	type="text"
	inputmode="decimal"
	{disabled}
	class={`ui-number ${scrub ? 'ui-number-scrub' : ''} ${klass}`.trim()}
	value={text}
	onfocus={() => live.begin()}
	onblur={commitText}
	onpointerdown={onScrubDown}
	onkeydown={(e) => {
		if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
	}}
	oninput={(e) => (text = (e.currentTarget as HTMLInputElement).value)}
/>

<style>
	/* Inherits the app-wide input chrome (background/border/radius/padding/focus) + the coarse-pointer
	   --hit floor; this adds only the numeric register: a compact, right-aligned, tabular-figure box. */
	.ui-number {
		width: var(--number-width, 6rem);
		text-align: right;
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	/* Scrub-enabled inputs hint the horizontal drag with the resize cursor. */
	.ui-number-scrub {
		cursor: ew-resize;
	}
	.ui-number:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
</style>
