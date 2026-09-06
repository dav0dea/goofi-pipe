<!-- Segmented — one chrome-height strip of exclusive segments: the source switch a param row
     and a control widget wear. One segment alone is a toggle. -->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';

	export interface Segment {
		id: string;
		label: string;
		title?: string;
		/** The accessible name, where the label is a glyph. */
		name?: string;
		testid?: string;
	}

	let {
		value,
		segments,
		onChange,
		bad = false,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		/** The lit segment, or null for none. */
		value: string | null;
		segments: Segment[];
		onChange: (id: string) => void;
		/** Paint the lit segment in the danger tone. */
		bad?: boolean;
	} = $props();
</script>

<div {...rest} role="group" class={`ui-segmented ${klass}`.trim()}>
	{#each segments as s (s.id)}
		<button
			type="button"
			class="seg"
			class:on={s.id === value}
			class:bad={s.id === value && bad}
			aria-pressed={s.id === value}
			title={s.title}
			aria-label={s.name}
			data-testid={s.testid}
			onclick={() => onChange(s.id)}>{s.label}</button
		>
	{/each}
</div>

<style>
	.ui-segmented {
		display: flex;
		align-items: stretch;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		transition: border-color var(--dur-fast) var(--ease);
	}
	.ui-segmented:has(.seg.on:only-child) {
		border-color: var(--accent);
	}
	.seg {
		flex: 0 0 auto;
		min-width: 1.4rem;
		height: var(--chrome-control-h);
		padding: 0 var(--space-2);
		border: none;
		background: transparent;
		color: var(--text-muted);
		font-family: var(--font-sans);
		font-size: var(--fs-micro);
		font-weight: 600;
		line-height: 1;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.seg + .seg {
		border-left: 1px solid var(--border);
	}
	.seg:hover {
		background: var(--hover-fill);
		color: var(--text);
	}
	.seg.on {
		background: var(--accent-fill);
		color: var(--accent);
	}
	.seg.on.bad {
		background: var(--danger-fill);
		color: var(--danger);
	}
	.seg:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -1px;
	}
	/* Chrome-height by design; the finger floor is taken back here, as `density="chrome"` controls take theirs. */
	@media (hover: none) and (pointer: coarse) {
		.seg {
			min-width: var(--hit);
		}
	}
</style>
