<!--
  ParamField — the descriptor-driven inspector field (spec §2, D-N2). Renders ONE P `<Field>` whose
  control region is chosen by the pure `controlKind(descriptor)` discriminant (N-Task 1): the component
  is a thin type-switch over the P primitives, never re-deriving the mapping inside the render path.
  Identity-blind — the same callback contract the old ParamField had, so the ParamForm assembler
  (N-Task 4) drives it unchanged.

  Echo-suppression is INHERITED, not re-hand-rolled: the P Slider / NumberInput / TextInput embody the
  `useLiveValue` latch, so a live backend echo can't yank a control mid-edit without any latch here.

  The numeric pair (float | int) is a Slider + an UNBOUNDED NumberInput sharing the value. `vmin/vmax`
  are SOFT bounds — the engine does not clamp on set — so they scope ONLY the Slider's track (which
  auto-extends past them when the live value is out of range); the NumberInput takes no min/max, so
  typing 5 into a [0,1] field commits 5, not 1. Both share a REAL numeric `step` (int → 1, float → the
  slider's ~200-stop span over the auto-extended bounds) — never the native `'any'` string, which would
  NaN the NumberInput's `inputmode=decimal` scrub arithmetic.

  The `'expression'` (fx) editor is N-Task 3 — a marked placeholder holds its control region here, and
  `onSetExpression` is accepted now to freeze the contract. `class`/`data-testid` (and any attribute)
  forward to the Field root via `...rest`.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor } from '$lib/api/types';
	import { Field, Slider, NumberInput, Toggle, Trigger, Select, TextInput, IconButton, Spinner } from '$lib/ui';
	import { controlKind } from './controlKind';

	let {
		paramName,
		descriptor,
		onCommit,
		onSetExpression,
		onRefresh,
		refreshing = false,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
		onSetExpression: (
			expression: string | null,
			opts?: { enabled?: boolean; triggers_process?: boolean }
		) => void;
		onRefresh?: () => void;
		refreshing?: boolean;
	} = $props();

	// The one SSOT for the control decision (pure + unit-tested); the template switches on it. `onSetExpression`
	// is part of the frozen contract — the fx editor that consumes it lands in N-Task 3.
	const kind = $derived(controlKind(descriptor));

	// Numeric bounds + step, narrowed to float | int (`num` is truthy exactly when the control is
	// 'numeric', once the fx branch has been excluded). `step` is computed against the SAME
	// auto-extended bounds the Slider uses so it matches the slider's stop spacing and the NumberInput's
	// scrub increment.
	const num = $derived(descriptor.type === 'float' || descriptor.type === 'int' ? descriptor : null);
	const lo = $derived(num ? Math.min(num.vmin, num.value) : 0);
	const hi = $derived(num ? Math.max(num.vmax, num.value) : 1);
	const step = $derived(num ? (num.type === 'int' ? 1 : Math.max((hi - lo) / 200, 1e-6)) : 1);

	// Options live only on a StringParam; empty for every other kind (the select branch is only reached
	// for a string anyway).
	const options = $derived(descriptor.type === 'string' ? (descriptor.options ?? []) : []);
</script>

<Field label={paramName} doc={descriptor.doc ?? undefined} class={klass} {...rest}>
	{#if kind === 'expression'}
		<!-- fx editor — N-Task 3 (this task ships every non-expression control kind). -->
		<span class="fx-placeholder" data-testid="param-expr-placeholder">fx editor — N-Task 3</span>
	{:else if num}
		<!-- SOFT bounds → Slider only (it auto-extends); the NumberInput is UNBOUNDED (no clamp on set). -->
		<Slider value={num.value} onChange={onCommit} min={num.vmin} max={num.vmax} {step} data-testid="param-slider" />
		<NumberInput value={num.value} onChange={onCommit} {step} scrub data-testid="param-number" />
	{:else if kind === 'trigger'}
		<Trigger onclick={() => onCommit(true)} data-testid="param-trigger">{paramName}</Trigger>
	{:else if kind === 'toggle'}
		<Toggle value={Boolean(descriptor.value)} onChange={onCommit} data-testid="param-toggle" />
	{:else if kind === 'select'}
		<Select
			{options}
			value={String(descriptor.value)}
			onChange={onCommit}
			disabled={refreshing}
			data-testid="param-select"
		/>
		{#if onRefresh}
			<IconButton
				size="sm"
				label={refreshing ? 'Re-scanning…' : 'Re-scan for options'}
				disabled={refreshing}
				aria-busy={refreshing}
				data-testid="param-refresh"
				onclick={onRefresh}
			>
				{#if refreshing}
					<Spinner size="sm" />
				{:else}
					⟳
				{/if}
			</IconButton>
		{/if}
	{:else if kind === 'text'}
		<TextInput value={String(descriptor.value)} onChange={onCommit} data-testid="param-text" />
	{:else}
		<code class="unknown" data-testid="param-unknown">{JSON.stringify(descriptor.value)}</code>
	{/if}
</Field>

<style>
	.fx-placeholder {
		color: var(--text-muted);
		font-size: var(--fs-micro);
		font-style: italic;
	}
	/* The read-only fallback for an unrecognised param type — a monospace JSON dump, wrapping so a long
	   value never forces a horizontal scroll. */
	.unknown {
		min-width: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-muted);
		word-break: break-all;
	}
</style>
