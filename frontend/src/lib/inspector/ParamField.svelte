<!--
  ParamField — one inspector field, its control region chosen by `controlKind(descriptor)`, with the fx
  expression binding on every field. `vmin/vmax` are SOFT bounds: they scope only the Slider's track,
  and the NumberInput beside it commits what is typed.
-->
<script lang="ts">
	import { tick } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor } from '$lib/api/types';
	import {
		Field,
		Slider,
		NumberInput,
		Toggle,
		Select,
		TextInput,
		Icon,
		IconButton,
		Chip,
		type BadgeTone
	} from '$lib/ui';
	import { controlKind } from './controlKind';
	import ExprEditor from './expr/ExprEditor.svelte';
	import { ui } from '$lib/stores/ui.svelte';

	let {
		paramName,
		descriptor,
		onCommit,
		onSetExpression,
		onRefresh,
		refreshing = false,
		selfName,
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

	const fxActive = $derived(descriptor.expression_enabled);
	const fxTone = $derived<BadgeTone>(
		descriptor.expression_error ? 'danger' : fxActive ? 'accent' : 'neutral'
	);

	let multilineOpen = $state(false);
	// The expanded editor OWNS its text, so `apply` asks it to commit rather than reading a mirror.
	let applyExpanded = $state<(() => void) | null>(null);
	let fxRegionEl = $state<HTMLDivElement | null>(null);

	// A stable per-field id, so the global standdown below is ref-counted rather than a shared boolean.
	const editorId = $props.id();

	// Leaving expression mode with the editor open must re-run the standdown effect below, so its
	// cleanup fires instead of stranding `multilineOpen` on an unmounted editor.
	$effect(() => {
		if (!fxActive && multilineOpen) multilineOpen = false;
	});

	// While the multi-line editor is open every app-global chord stands down: it owns a full-height code
	// surface, so Ctrl+S is its editor's, not the app's.
	$effect(() => {
		if (multilineOpen) {
			ui().openEditor(editorId);
			return () => ui().closeEditor(editorId);
		}
	});

	function currentFlags(): { enabled?: boolean; triggers_process?: boolean } {
		return {
			enabled: descriptor.expression_enabled,
			triggers_process: descriptor.expression_triggers_process
		};
	}

	// The current value as a Python literal — the seed when fx is first switched on.
	function literalFor(d: ParamDescriptor): string {
		const v = d.value;
		if (typeof v === 'number') return String(v);
		if (typeof v === 'boolean') return v ? 'True' : 'False';
		return JSON.stringify(v);
	}

	function toggleFx(): void {
		if (fxActive) {
			// Stash the source, so a later flip-on does not retype it.
			onSetExpression(descriptor.expression, {
				enabled: false,
				triggers_process: descriptor.expression_triggers_process
			});
		} else {
			const seed = descriptor.expression ?? literalFor(descriptor);
			onSetExpression(seed, {
				enabled: true,
				triggers_process: descriptor.expression_triggers_process
			});
		}
	}

	function toggleTriggersProcess(): void {
		onSetExpression(descriptor.expression, {
			enabled: descriptor.expression_enabled,
			triggers_process: !descriptor.expression_triggers_process
		});
	}

	// The inline commit FORCES enabled; the multi-line apply below PRESERVES the flags.
	function commitExpr(v: string): void {
		onSetExpression(v, { ...currentFlags(), enabled: true });
	}

	function openMultiline(): void {
		multilineOpen = true;
	}
	function applyMultiline(v: string): void {
		onSetExpression(v, currentFlags());
		void cancelMultiline();
	}
	async function applyFromChip(): Promise<void> {
		applyExpanded?.();
		await cancelMultiline();
	}
	async function cancelMultiline(): Promise<void> {
		multilineOpen = false;
		await restoreExpandFocus();
	}
	// After the editor unmounts, focus returns to the ⤢ expand affordance rather than falling to <body>.
	async function restoreExpandFocus(): Promise<void> {
		await tick();
		fxRegionEl?.querySelector<HTMLElement>('[data-testid="param-expr-expand"]')?.focus();
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

<!-- A SIBLING of the label (via Field's adornment slot), so its button never steals the label's focus target. -->
{#snippet fx()}
	{#if fxActive}
		<Chip
			tone={descriptor.expression_triggers_process ? 'accent' : 'neutral'}
			aria-pressed={descriptor.expression_triggers_process}
			onclick={toggleTriggersProcess}
			title="When this expression's value changes, wake the node's process()"
			data-testid="param-expr-triggers-process"
		>
			trig
		</Chip>
	{/if}
	<Chip
		tone={fxTone}
		aria-pressed={fxActive}
		onclick={toggleFx}
		title={fxActive ? 'Disable expression (keeps the source)' : 'Enable expression'}
		data-testid="param-fx-toggle"
	>
		fx
	</Chip>
{/snippet}

<Field label={paramName} doc={descriptor.doc ?? undefined} adornment={fx} class={klass} {...rest}>
	<!-- `display: contents` so the face inherits WITHOUT laying out: Field requires paired controls to
	     be its direct children, and a real box would take them out of the @container column-flip. -->
	<div class="pf-value">
		{#if kind === 'expression'}
			<div class="fx-region" bind:this={fxRegionEl}>
				{#if multilineOpen}
					<ExprEditor
						{selfName}
						multiline
						value={descriptor.expression ?? ''}
						error={descriptor.expression_error}
						onCommit={applyMultiline}
						onCancel={cancelMultiline}
						bindCommit={(c) => (applyExpanded = c)}
						label={`${paramName} expression`}
						testid="param-expr-multiline"
						autofocus
					/>
					<div class="fx-actions">
						<span class="fx-kbd" aria-hidden="true">⌃⏎ apply · esc cancel</span>
						<Chip onclick={cancelMultiline} data-testid="param-expr-collapse">collapse</Chip>
						<Chip tone="accent" onclick={applyFromChip} data-testid="param-expr-apply">apply</Chip>
					</div>
				{:else}
					<div class="fx-inline">
						<ExprEditor
							{selfName}
							value={descriptor.expression ?? ''}
							error={descriptor.expression_error}
							onCommit={commitExpr}
							label={`${paramName} expression`}
							placeholder="nd('oscillator0').out.data.mean()"
							testid="param-expr-input"
						/>
						<IconButton
							size="sm"
							label="Open the multi-line editor"
							onclick={openMultiline}
							data-testid="param-expr-expand"
						>
							<Icon name="maximize-2" />
						</IconButton>
					</div>
				{/if}
				{#if descriptor.expression_error}
					<div class="fx-error" title={descriptor.expression_error} data-testid="param-expr-error">
						<span class="prefix"><Icon name="triangle-alert" /></span>
						<span class="msg">{descriptor.expression_error}</span>
					</div>
				{:else}
					<div class="fx-preview" title={String(descriptor.value)}>
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
	.fx-region {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}
	.fx-inline {
		display: flex;
		align-items: stretch;
		gap: var(--space-2);
		min-width: 0;
	}
	.fx-actions {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		justify-content: flex-end;
	}
	.fx-kbd {
		flex: 1;
		min-width: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.fx-error,
	.fx-preview {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		min-width: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		padding: 0 var(--space-1);
	}
	.fx-error {
		color: var(--danger);
	}
	.fx-error .prefix {
		flex-shrink: 0;
	}
	.fx-error .msg {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.fx-preview {
		color: var(--text-muted);
	}
	.fx-preview .prefix {
		/* The `=` lead-in recedes behind the value; the preview itself is live, not disabled. */
		opacity: 0.6;
	}
	.fx-preview .value {
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.unknown {
		min-width: 0;
		font-size: var(--fs-micro);
		color: var(--text-muted);
		word-break: break-all;
	}
</style>
