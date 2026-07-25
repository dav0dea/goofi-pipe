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

  The always-present fx binding (spec §3, D-N3/D-N4) hangs off the Field's `adornment` snippet — the `fx`
  Chip on every field (accent when active, danger when the expression errors) plus a `trig` Chip while
  active. When `expression_enabled`, the control region becomes the buffered single-line expr `TextInput`
  (its echo-suppression INHERITED from the P control's `useLiveValue` latch, not re-hand-rolled) with an
  expand affordance that grows an IN-PANEL multi-line textarea (no modal — the graph stays visible), and a
  `= preview` / error row below. The three enabled-semantics are preserved exactly: inline commit FORCES
  `enabled`, the multi-line apply PRESERVES the flags, fx-off STASHES the source. `class`/`data-testid`
  (and any attribute) forward to the Field root via `...rest`.
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
		Trigger,
		Select,
		TextInput,
		IconButton,
		Chip,
		type BadgeTone
	} from '$lib/ui';
	import { controlKind } from './controlKind';
	import { ui } from '$lib/stores/ui.svelte';

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

	// --- fx (expression) binding (spec §3, D-N3/D-N4) ---
	const fxActive = $derived(descriptor.expression_enabled);
	// The fx Chip's tone: danger PINS an error (a pointer to the row below), accent = active, else neutral.
	const fxTone = $derived<BadgeTone>(
		descriptor.expression_error ? 'danger' : fxActive ? 'accent' : 'neutral'
	);

	// The in-panel multi-line editor (D-N4: no modal — it grows in place so the graph stays visible).
	let multilineOpen = $state(false);
	let multilineBuf = $state('');
	// A ref to the fx control region so, on collapse, focus returns to the ⤢ expand button (which only
	// exists in the collapsed inline branch) instead of dropping to <body>.
	let fxRegionEl = $state<HTMLDivElement | null>(null);

	// A stable per-field id so the global standdown is REF-COUNTED, not a shared boolean: two fields
	// with an open editor each register their own id, and the standdown lifts only when the last one
	// unregisters (§ inspector fix #3).
	const editorId = $props.id();

	// If the field leaves expression mode while its editor is open — fx toggled off via the adornment
	// Chip, or an external descriptor change swaps this row to a non-expression param — drop the editor.
	// This re-runs the standdown effect below so its cleanup fires and the field stops holding down
	// global undo/redo (§ inspector fix #2A), instead of stranding `multilineOpen` on an unmounted
	// textarea.
	$effect(() => {
		if (!fxActive && multilineOpen) multilineOpen = false;
	});

	// While the multi-line editor is open the global undo/redo stands down (else Ctrl-Z fights the
	// textarea). Registering by `editorId` clears the standdown on collapse AND on unmount (e.g. a node
	// switch remounts this field) via the effect's cleanup — and ref-counts, so collapsing one editor
	// never lifts another's standdown.
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

	// The current value rendered as a Python literal — the seed when fx is first switched on.
	function literalFor(d: ParamDescriptor): string {
		const v = d.value;
		if (typeof v === 'number') return String(v);
		if (typeof v === 'boolean') return v ? 'True' : 'False';
		return JSON.stringify(v);
	}

	function toggleFx(): void {
		if (fxActive) {
			// OFF — stop the engine but STASH the source so a later flip-on doesn't retype.
			onSetExpression(descriptor.expression, {
				enabled: false,
				triggers_process: descriptor.expression_triggers_process
			});
		} else {
			// ON — reuse the stashed source if there is one, else seed the current value as a literal.
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

	// Committing the inline buffer FORCES enabled — editing implies "make this active"; the fx toggle is
	// the way to disable. (The multi-line apply, by contrast, PRESERVES the flags as-is.)
	function commitExpr(v: string): void {
		onSetExpression(v, { ...currentFlags(), enabled: true });
	}

	function openMultiline(): void {
		multilineBuf = descriptor.expression ?? '';
		multilineOpen = true;
	}
	async function applyMultiline(): Promise<void> {
		onSetExpression(multilineBuf, currentFlags());
		multilineOpen = false;
		await restoreExpandFocus();
	}
	async function cancelMultiline(): Promise<void> {
		multilineOpen = false;
		await restoreExpandFocus();
	}
	// Escape/⌃⏎/collapse-click all route through the two functions above; after the textarea unmounts,
	// return focus to the ⤢ expand affordance (mirrors autofocus-on-expand) so it never falls to <body>.
	async function restoreExpandFocus(): Promise<void> {
		await tick();
		fxRegionEl?.querySelector<HTMLElement>('[data-testid="param-expr-expand"]')?.focus();
	}
	function multilineKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			cancelMultiline();
		} else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
			e.preventDefault();
			e.stopPropagation();
			applyMultiline();
		}
	}

	// The live evaluated value shown under the input — strings truncated ~32 chars (the old preview).
	function previewText(): string {
		const v = descriptor.value;
		if (v === null || v === undefined) return '—';
		if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '—';
		if (typeof v === 'boolean') return v ? 'true' : 'false';
		if (typeof v === 'string') return v.length > 32 ? v.slice(0, 31) + '…' : v;
		return String(v);
	}

	// Focus the multi-line editor when it grows in — an action, so there is no a11y `autofocus` warning.
	function autofocus(el: HTMLTextAreaElement): void {
		el.focus();
		const len = el.value.length;
		el.setSelectionRange(len, len);
	}
</script>

<!-- The always-present fx binding: the fx Chip on every field + the trig Chip while active. A SIBLING of
     the label (via Field's adornment slot) so its button never steals the label's focus target. -->
{#snippet fx()}
	{#if fxActive}
		<Chip
			tone={descriptor.expression_triggers_process ? 'accent' : 'neutral'}
			onclick={toggleTriggersProcess}
			title="When this expression's value changes, wake the node's process()"
			data-testid="param-expr-triggers-process"
		>
			trig
		</Chip>
	{/if}
	<Chip
		tone={fxTone}
		onclick={toggleFx}
		title={fxActive ? 'Disable expression (keeps the source)' : 'Enable expression'}
		data-testid="param-fx-toggle"
	>
		fx
	</Chip>
{/snippet}

<Field label={paramName} doc={descriptor.doc ?? undefined} adornment={fx} class={klass} {...rest}>
	{#if kind === 'expression'}
		<!-- The fx editor takes over the control region (spec §3): the buffered expr input (or, expanded,
		     the in-panel multi-line textarea — D-N4, no modal), with a preview / error row below. -->
		<div class="fx-region" bind:this={fxRegionEl}>
			{#if multilineOpen}
				<textarea
					class="fx-multiline"
					aria-label={`${paramName} expression`}
					use:autofocus
					bind:value={multilineBuf}
					spellcheck="false"
					autocapitalize="off"
					{...{ autocorrect: 'off' }}
					onkeydown={multilineKeydown}
					data-testid="param-expr-multiline"
				></textarea>
				<div class="fx-actions">
					<span class="fx-kbd" aria-hidden="true">⌃⏎ apply · esc cancel</span>
					<Chip onclick={cancelMultiline} data-testid="param-expr-collapse">collapse</Chip>
					<Chip tone="accent" onclick={applyMultiline} data-testid="param-expr-apply">apply</Chip>
				</div>
			{:else}
				<div class="fx-inline">
					<TextInput
						value={descriptor.expression ?? ''}
						onChange={commitExpr}
						inputmode="search"
						placeholder="nd('oscillator0').out.data.mean()"
						data-testid="param-expr-input"
					/>
					<IconButton
						size="sm"
						label="Open the multi-line editor"
						onclick={openMultiline}
						data-testid="param-expr-expand"
					>
						⤢
					</IconButton>
				</div>
			{/if}
			{#if descriptor.expression_error}
				<div class="fx-error" title={descriptor.expression_error} data-testid="param-expr-error">
					<span class="prefix" aria-hidden="true">⚠</span>
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
		<!-- SOFT bounds → Slider only (it auto-extends); the NumberInput is UNBOUNDED (no clamp on set). -->
		<Slider value={num.value} onChange={onCommit} min={num.vmin} max={num.vmax} {step} data-testid="param-slider" />
		<NumberInput value={num.value} onChange={onCommit} {step} scrub data-testid="param-number" />
	{:else if kind === 'trigger'}
		<Trigger onclick={() => onCommit(true)} data-testid="param-trigger">{paramName}</Trigger>
	{:else if kind === 'toggle'}
		<Toggle value={Boolean(descriptor.value)} onChange={onCommit} data-testid="param-toggle" />
	{:else if kind === 'select'}
		<!-- Delegate the ⟳ to the P Select (its built-in refresh affordance), gated on
		     `descriptor.refreshable`: a non-refreshable dropdown passes `onRefresh={undefined}` so no ⟳
		     renders (the engine rejects a refresh for a non-refreshable param by contract). -->
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
</Field>

<style>
	/* The fx editor occupies the whole control region as one column-flexed child (the Field row lays its
	   children horizontally; a single full-width child stacks the input over the preview/error). */
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
	/* The in-panel multi-line editor — grows in place (D-N4), resizable within the panel; no floating window. */
	.fx-multiline {
		width: 100%;
		min-height: 7rem;
		resize: vertical;
		padding: var(--space-3);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		line-height: 1.5;
		color: var(--accent);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		tab-size: 4;
	}
	.fx-multiline:focus {
		border-color: var(--accent);
	}
	/* Touch: lift the multi-line editor to 16px so focusing it does not force-zoom iOS (the desktop
	   --fs-micro size above is kept). Mirrors app.css's coarse input/textarea floor. */
	@media (hover: none) and (pointer: coarse) {
		.fx-multiline {
			font-size: 16px;
		}
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
		opacity: 0.6;
	}
	.fx-preview .value {
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
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
