<!--
  ParamField — one inspector field, its control region chosen by `controlKind(descriptor)`, with the
  three-way source toggle on every field: constant, expression, reference. `vmin/vmax` are SOFT
  bounds: they scope only the Slider's track, and the NumberInput beside it commits what is typed.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor, ParamMode, SourcePatch } from '$lib/api/types';
	import { Field, Slider, NumberInput, Toggle, Select, TextInput, Button, Icon } from '$lib/ui';
	import { controlKind } from './controlKind';
	import { literalFor } from './paramSeed';
	import ExprEditor from './expr/ExprEditor.svelte';
	import RefPicker from './RefPicker.svelte';

	let {
		paramName,
		descriptor,
		onCommit,
		onSetSource,
		onRefresh,
		onPulse,
		refreshing = false,
		selfName,
		modes = ['constant', 'expression', 'reference'],
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
		onSetSource: (source: SourcePatch) => void;
		onRefresh?: () => void;
		onPulse?: () => void;
		refreshing?: boolean;
		/** The node's display name, handed to the expression editor as `me`. */
		selfName?: string;
		/** The sources this field offers; a global follows a reference and never an expression. */
		modes?: ParamMode[];
	} = $props();

	const kind = $derived(controlKind(descriptor));

	// `step` is computed against the SAME auto-extended bounds the Slider uses; a native `'any'` would
	// NaN the NumberInput's scrub arithmetic.
	const num = $derived(descriptor.type === 'float' || descriptor.type === 'int' ? descriptor : null);
	const lo = $derived(num ? Math.min(num.vmin, num.value) : 0);
	const hi = $derived(num ? Math.max(num.vmax, num.value) : 1);
	const step = $derived(num ? (num.type === 'int' ? 1 : Math.max((hi - lo) / 200, 1e-6)) : 1);

	const options = $derived(descriptor.type === 'string' ? (descriptor.options ?? []) : []);

	const driven = $derived(descriptor.mode !== 'constant');
	// A reference chosen before one is retained shows the picker without a record to show yet.
	let picking = $state(false);
	// The MODE, not the kind: a pulse is a button in every mode, so its kind cannot name its source.
	const showPicker = $derived(descriptor.mode === 'reference' || picking);
	const showSource = $derived(showPicker || descriptor.mode === 'expression');
	// The error and preview belong to a source that IS live: a picker over a retained expression
	// shows neither.
	const shown = $derived(descriptor.mode === 'reference' || (descriptor.mode === 'expression' && !picking));
	$effect(() => {
		if (descriptor.mode === 'reference') picking = false;
	});

	/** The lit segment — `picking` is a reference being chosen, whatever the committed mode says. */
	function active(mode: ParamMode): boolean {
		if (picking) return mode === 'reference';
		return descriptor.mode === mode;
	}

	function choose(mode: ParamMode): void {
		picking = false;
		if (mode === descriptor.mode) return;
		if (mode === 'expression' && !descriptor.expression) {
			onSetSource({ expression: literalFor(descriptor) });
		} else if (mode === 'reference' && !descriptor.reference) {
			picking = true;
		} else {
			onSetSource({ mode });
		}
	}

	function previewText(): string {
		const v = descriptor.value;
		if (v === null || v === undefined) return '—';
		if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '—';
		if (typeof v === 'boolean') return v ? 'true' : 'false';
		if (typeof v === 'string') return v.length > 32 ? v.slice(0, 31) + '…' : v;
		return String(v);
	}
</script>

{#snippet segment(mode: ParamMode, glyph: string, hint: string, testid: string)}
	<button
		type="button"
		class="seg"
		class:on={active(mode)}
		class:bad={active(mode) && !!descriptor.error}
		aria-pressed={active(mode)}
		onclick={() => choose(mode)}
		title={hint}
		data-testid={testid}>{glyph}</button
	>
{/snippet}

<!-- A SIBLING of the label (via Field's adornment slot), so its buttons never steal the label's focus target. -->
{#snippet source()}
	{#if driven}
		<button
			type="button"
			class="seg lone"
			class:on={descriptor.triggers}
			aria-pressed={descriptor.triggers}
			onclick={() => onSetSource({ triggers: !descriptor.triggers })}
			title="When this source's value changes, wake the node's process()"
			data-testid="param-triggers">trig</button
		>
	{/if}
	<div class="modes" role="group" aria-label={`${paramName} source`} data-testid="param-mode">
		{@render segment('constant', '=', 'A constant: the value beside it', 'param-mode-constant')}
		{#if modes.includes('expression')}
			{@render segment('expression', 'fx', 'An expression over nd(), globals and me, at control rate', 'param-mode-expression')}
		{/if}
		{#if modes.includes('reference')}
			{@render segment('reference', 'ref', "A reference to one node's output, at that node's rate", 'param-mode-reference')}
		{/if}
	</div>
{/snippet}

<Field label={paramName} doc={descriptor.doc ?? undefined} adornment={source} class={klass} {...rest}>
	<!-- `display: contents` so the face inherits WITHOUT laying out: Field requires paired controls to
	     be its direct children, and a real box would take them out of the @container column-flip. -->
	<div class="pf-value">
		{#if kind === 'pulse'}
			<!-- The field's label already names it, so the button carries the ACT alone and fills the row. -->
			<Button class="pf-pulse" title="Fire one pulse" onclick={onPulse} data-testid="param-pulse">
				<Icon name="activity" />pulse
			</Button>
		{/if}
		{#if showSource}
			<div class="src-region">
				{#if showPicker}
					<RefPicker
						value={descriptor.reference}
						paramType={descriptor.type}
						onCommit={(reference) => onSetSource({ reference })}
						testid="param-ref"
					/>
				{:else}
					<ExprEditor
						{selfName}
						value={descriptor.expression ?? ''}
						error={descriptor.error}
						onCommit={(expression) => onSetSource({ expression })}
						label={`${paramName} expression`}
						placeholder="nd('oscillator0').out.data.mean()"
						testid="param-expr-input"
					/>
				{/if}
				{#if shown && descriptor.error}
					<div class="src-error" title={descriptor.error} data-testid="param-source-error">
						<span class="prefix"><Icon name="triangle-alert" /></span>
						<span class="msg">{descriptor.error}</span>
					</div>
				{:else if shown && kind !== 'pulse'}
					<div class="src-preview" title={String(descriptor.value)}>
						<span class="prefix" aria-hidden="true">=</span>
						<span class="value">{previewText()}</span>
					</div>
				{/if}
			</div>
		{:else if num}
			<!-- SOFT bounds → Slider only; the NumberInput is UNBOUNDED (the engine does not clamp on set). -->
			<Slider value={num.value} onChange={onCommit} min={num.vmin} max={num.vmax} {step} data-testid="param-slider" />
			<NumberInput value={num.value} onChange={onCommit} {step} scrub data-testid="param-number" />
		{:else if kind === 'toggle'}
			<Toggle value={Boolean(descriptor.value)} onChange={onCommit} data-testid="param-toggle" />
		{:else if kind === 'select'}
			<!-- A non-refreshable dropdown passes no `onRefresh`, so the Select renders no ⟳. -->
			<Select
				{options}
				value={String(descriptor.value)}
				onChange={onCommit}
				onRefresh={descriptor.refreshable ? onRefresh : undefined}
				{refreshing}
				refreshTestid="param-refresh"
				data-testid="param-select"
			/>
		{:else if kind === 'text'}
			<TextInput value={String(descriptor.value)} onChange={onCommit} data-testid="param-text" />
		{:else if kind === 'unknown'}
			<code class="unknown" data-testid="param-unknown">{JSON.stringify(descriptor.value)}</code>
		{/if}
	</div>
</Field>

<style>
	/* Values are data; the label above them is chrome. Box-less, so the controls inside stay Field's
	   own direct children. */
	.pf-value {
		display: contents;
		font-family: var(--font-mono);
		/* Narrower than the primitive's default: a param row seats a slider, a number AND the source
		   switch, and the number is the one of the three with slack to give. */
		--number-width: 5rem;
	}
	/* The source switch is chrome beside a value, not a control of its own, so it wears the strip
	   box the panel headers wear rather than a row of pills. */
	.modes {
		display: flex;
		align-items: stretch;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
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
		transition:
			background var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease),
			border-color var(--dur-fast) var(--ease);
	}
	.seg + .seg {
		border-left: 1px solid var(--border);
	}
	.seg.lone {
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
	.seg:hover {
		background: var(--hover-fill);
		color: var(--text);
	}
	.seg.on {
		background: var(--accent-fill);
		color: var(--accent);
	}
	.seg.lone.on {
		border-color: var(--accent);
	}
	.seg.on.bad {
		background: var(--danger-fill);
		color: var(--danger);
	}
	.seg:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -1px;
	}
	/* The strip is chrome-height by design; the finger floor is taken back here, as `$lib/ui`'s own
	   `density="chrome"` controls take theirs. */
	@media (hover: none) and (pointer: coarse) {
		.seg {
			min-width: var(--hit);
		}
	}
	/* A pulse has no value beside it, so the whole row is the target — and it takes the rung above
	   the fields around it, so a press target never reads as one more box to type in. */
	.pf-value :global(.pf-pulse) {
		flex: 1 1 auto;
		background: var(--surface-3);
		border-color: var(--border-strong);
	}
	.src-region {
		flex: 3 1 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}
	.src-error,
	.src-preview {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		min-width: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		padding: 0 var(--space-1);
	}
	.src-error {
		color: var(--danger);
	}
	.src-error .prefix {
		flex-shrink: 0;
	}
	.src-error .msg {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.src-preview {
		color: var(--text-muted);
	}
	.unknown {
		font-size: var(--fs-micro);
		color: var(--text-muted);
	}
</style>
