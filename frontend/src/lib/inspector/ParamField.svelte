<!--
  ParamField — one inspector field, its control region chosen by `controlKind(descriptor)`, with the
  three-way source toggle on every field: constant, expression, reference. `vmin/vmax` are SOFT
  bounds: they scope only the Slider's track, and the NumberInput beside it commits what is typed.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor, ParamMode, SourcePatch } from '$lib/api/types';
	import { Field, Slider, NumberInput, Toggle, Select, TextInput, Icon, Chip, type BadgeTone } from '$lib/ui';
	import { controlKind } from './controlKind';
	import ExprEditor from './expr/ExprEditor.svelte';
	import RefPicker from './RefPicker.svelte';

	let {
		paramName,
		descriptor,
		onCommit,
		onSetSource,
		onRefresh,
		refreshing = false,
		selfName,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
		onSetSource: (source: SourcePatch) => void;
		onRefresh?: () => void;
		refreshing?: boolean;
		/** The node's display name, handed to the expression editor as `me`. */
		selfName?: string;
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
	const showPicker = $derived(kind === 'reference' || picking);
	// The error and preview belong to a source that IS live: a picker over a retained expression
	// shows neither.
	const shown = $derived(kind === 'reference' || (kind === 'expression' && !picking));
	$effect(() => {
		if (kind === 'reference') picking = false;
	});

	function tone(mode: ParamMode): BadgeTone {
		if (descriptor.mode !== mode) return 'neutral';
		return descriptor.error ? 'danger' : 'accent';
	}

	// The current value as a Python literal — the seed when an expression is first switched on.
	function literalFor(d: ParamDescriptor): string {
		const v = d.value;
		if (typeof v === 'number') return String(v);
		if (typeof v === 'boolean') return v ? 'True' : 'False';
		return JSON.stringify(v);
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

<!-- A SIBLING of the label (via Field's adornment slot), so its buttons never steal the label's focus target. -->
{#snippet source()}
	{#if driven}
		<Chip
			tone={descriptor.triggers ? 'accent' : 'neutral'}
			aria-pressed={descriptor.triggers}
			onclick={() => onSetSource({ triggers: !descriptor.triggers })}
			title="When this source's value changes, wake the node's process()"
			data-testid="param-triggers"
		>
			trig
		</Chip>
	{/if}
	<div class="mode" role="group" aria-label={`${paramName} source`} data-testid="param-mode">
		<Chip
			tone={tone('constant')}
			aria-pressed={descriptor.mode === 'constant' && !picking}
			onclick={() => choose('constant')}
			title="A constant: the value below"
			data-testid="param-mode-constant"
		>
			=
		</Chip>
		<Chip
			tone={tone('expression')}
			aria-pressed={descriptor.mode === 'expression'}
			onclick={() => choose('expression')}
			title="An expression over nd(), globals and me, at control rate"
			data-testid="param-mode-expression"
		>
			fx
		</Chip>
		<Chip
			tone={tone('reference')}
			aria-pressed={descriptor.mode === 'reference' || picking}
			onclick={() => choose('reference')}
			title="A reference to one node's output, at that node's rate"
			data-testid="param-mode-reference"
		>
			ref
		</Chip>
	</div>
{/snippet}

<Field label={paramName} doc={descriptor.doc ?? undefined} adornment={source} class={klass} {...rest}>
	<!-- `display: contents` so the face inherits WITHOUT laying out: Field requires paired controls to
	     be its direct children, and a real box would take them out of the @container column-flip. -->
	<div class="pf-value">
		{#if showPicker || kind === 'expression'}
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
				{:else if shown}
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
		{:else}
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
	}
	.mode {
		display: inline-flex;
		gap: var(--space-1);
	}
	.src-region {
		flex: 1;
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
