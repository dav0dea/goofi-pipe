<script lang="ts">
	import { untrack } from 'svelte';
	import type { ParamDescriptor } from '$lib/api/types';
	import ExpressionModal from './ExpressionModal.svelte';

	type SetExprOpts = { enabled?: boolean; triggers_process?: boolean; autoeval?: boolean };
	type Props = {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
		onSetExpression: (expression: string | null, opts?: SetExprOpts) => void;
	};
	const { paramName, descriptor, onCommit, onSetExpression }: Props = $props();

	let local = $state<unknown>(untrack(() => descriptor.value));
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

	// --- expression mode ---
	const exprActive = $derived(descriptor.expression_enabled);
	// Buffer for the inline single-line editor. Kept in sync with the
	// backend source via $effect unless the user is actively typing.
	let exprBuf = $state<string>('');
	let exprEditing = $state(false);
	let modalOpen = $state(false);
	$effect(() => {
		if (!exprEditing && !modalOpen) {
			exprBuf = descriptor.expression ?? '';
		}
	});

	function currentFlags(): SetExprOpts {
		return {
			enabled: descriptor.expression_enabled,
			triggers_process: descriptor.expression_triggers_process,
			autoeval: descriptor.expression_autoeval
		};
	}

	function toggleFx(): void {
		if (exprActive) {
			// Turn off — engine stops, source is stashed on the param so
			// the user can flip back on without losing what they wrote.
			onSetExpression(descriptor.expression, {
				enabled: false,
				triggers_process: descriptor.expression_triggers_process,
				autoeval: descriptor.expression_autoeval
			});
		} else {
			// Turn on — use the stashed source if there is one, else seed
			// from the current value as a Python literal.
			const seed = descriptor.expression ?? literalFor(descriptor);
			onSetExpression(seed, {
				enabled: true,
				triggers_process: descriptor.expression_triggers_process,
				autoeval: descriptor.expression_autoeval
			});
			exprBuf = seed;
		}
	}

	function toggleTriggersProcess(): void {
		onSetExpression(descriptor.expression, {
			enabled: descriptor.expression_enabled,
			triggers_process: !descriptor.expression_triggers_process,
			autoeval: descriptor.expression_autoeval
		});
	}

	function toggleAutoeval(): void {
		onSetExpression(descriptor.expression, {
			enabled: descriptor.expression_enabled,
			triggers_process: descriptor.expression_triggers_process,
			autoeval: !descriptor.expression_autoeval
		});
	}

	function literalFor(d: ParamDescriptor): string {
		const v = d.value;
		if (typeof v === 'number') return String(v);
		if (typeof v === 'boolean') return v ? 'True' : 'False';
		if (typeof v === 'string') return JSON.stringify(v);
		return JSON.stringify(v);
	}

	function commitExpr(): void {
		// Committing the buffer enables the expression — editing implies
		// "I want this to be active". If the user wants to disable, they
		// hit the fx toggle.
		onSetExpression(exprBuf, { ...currentFlags(), enabled: true });
	}

	function previewText(): string {
		const v = descriptor.value;
		if (v === null || v === undefined) return '—';
		if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '—';
		if (typeof v === 'boolean') return v ? 'true' : 'false';
		if (typeof v === 'string') {
			return v.length > 32 ? v.slice(0, 31) + '…' : v;
		}
		return String(v);
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

</script>

<div class="field" title={descriptor.doc ?? ''}>
	<div class="top-row">
		<span class="label">{paramName}</span>
		{#if numeric && !exprActive}
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
		{/if}
		{#if exprActive}
			<button
				class="flag-btn"
				class:on={descriptor.expression_autoeval}
				onclick={toggleAutoeval}
				aria-pressed={descriptor.expression_autoeval}
				title="Re-evaluate this expression before every process() tick (use for expressions without slot refs, e.g. time.time())"
				data-testid="param-expr-autoeval"
			>
				auto
			</button>
			<button
				class="flag-btn"
				class:on={descriptor.expression_triggers_process}
				onclick={toggleTriggersProcess}
				aria-pressed={descriptor.expression_triggers_process}
				title="When this expression's value changes, wake the node's process()"
				data-testid="param-expr-triggers-process"
			>
				trig
			</button>
		{/if}
		<button
			class="fx-btn"
			class:on={exprActive}
			onclick={toggleFx}
			title={exprActive ? 'Disable expression (keeps the source)' : 'Enable expression'}
			aria-pressed={exprActive}
			data-testid="param-fx-toggle"
		>
			fx
		</button>
	</div>

	{#if exprActive}
		<div class="expr-row">
			<input
				class="expr-input"
				type="text"
				spellcheck="false"
				autocomplete="off"
				autocorrect="off"
				autocapitalize="off"
				value={exprBuf}
				placeholder="nd('oscillator0').out.data.mean()"
				onfocus={() => (exprEditing = true)}
				onblur={() => {
					exprEditing = false;
					commitExpr();
				}}
				onkeydown={(e) => {
					if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
					else if (e.key === 'Escape') {
						exprBuf = descriptor.expression ?? '';
						(e.currentTarget as HTMLInputElement).blur();
					}
				}}
				oninput={(e) => (exprBuf = (e.currentTarget as HTMLInputElement).value)}
				data-testid="param-expr-input"
			/>
			<button
				class="expand-btn"
				onclick={() => (modalOpen = true)}
				title="Open in multi-line editor"
				aria-label="Open expression editor"
				data-testid="param-expr-expand"
			>
				⤢
			</button>
		</div>
		<div class="expr-preview" title={String(descriptor.value)}>
			<span class="prefix">=</span>
			<span class="value">{previewText()}</span>
		</div>
	{:else if numeric}
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

{#if modalOpen}
	<ExpressionModal
		title={paramName}
		initial={exprBuf}
		preview={previewText()}
		onApply={(src) => {
			exprBuf = src;
			modalOpen = false;
			onSetExpression(src, currentFlags());
		}}
		onCancel={() => {
			modalOpen = false;
		}}
	/>
{/if}

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
		font-size: 12px;
	}
	.top-row {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	/* The param name is a primary scan target — keep it bright and a notch
	   larger than the supporting chrome (bounds, fx, toggles), so the eye lands
	   on "what" before "how much". */
	.label {
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 13px;
		font-weight: 600;
		text-transform: lowercase;
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		letter-spacing: 0.01em;
	}
	.fx-btn {
		font-family: var(--font-mono);
		font-size: 10px;
		font-style: italic;
		letter-spacing: 0.04em;
		min-width: 24px;
		min-height: 22px;
		padding: 2px 6px;
		border: 1px solid color-mix(in srgb, var(--text-faint) 30%, transparent);
		background: transparent;
		color: var(--text-faint);
		border-radius: 3px;
		cursor: pointer;
		transition:
			background 80ms ease,
			color 80ms ease,
			border-color 80ms ease;
		flex-shrink: 0;
	}
	.fx-btn:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.fx-btn.on {
		background: color-mix(in srgb, var(--accent) 20%, transparent);
		color: var(--accent);
		border-color: var(--accent);
		font-style: normal;
	}
	.slider-row {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.expr-row {
		display: flex;
		align-items: stretch;
		gap: 4px;
		min-width: 0;
	}
	.expr-input {
		flex: 1;
		min-width: 0;
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 5px 8px;
		background: color-mix(in srgb, var(--bg) 70%, transparent);
		border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
		border-radius: 3px;
		color: var(--accent);
	}
	.expr-input:focus {
		outline: none;
		border-color: var(--accent);
	}
	.expand-btn {
		min-width: 26px;
		padding: 0 6px;
		background: color-mix(in srgb, var(--bg-elev-3) 60%, transparent);
		border: 1px solid var(--border);
		color: var(--text-dim);
		font-size: 12px;
		line-height: 1;
		border-radius: 3px;
		cursor: pointer;
		transition:
			background 80ms ease,
			color 80ms ease,
			border-color 80ms ease;
	}
	.expand-btn:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.expr-preview {
		display: flex;
		gap: 6px;
		align-items: baseline;
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-faint);
		padding: 0 2px;
		overflow: hidden;
	}
	.expr-preview .prefix {
		opacity: 0.6;
	}
	.expr-preview .value {
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.flag-btn {
		font-family: var(--font-mono);
		font-size: 9px;
		letter-spacing: 0.04em;
		padding: 2px 7px;
		min-height: 22px;
		border: 1px solid color-mix(in srgb, var(--text-faint) 30%, transparent);
		background: transparent;
		color: var(--text-faint);
		border-radius: 999px;
		cursor: pointer;
		flex-shrink: 0;
		transition:
			background 80ms ease,
			color 80ms ease,
			border-color 80ms ease;
	}
	.flag-btn:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.flag-btn.on {
		background: color-mix(in srgb, var(--accent) 20%, transparent);
		color: var(--accent);
		border-color: var(--accent);
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
		touch-action: pan-y;
	}
	/* The edited value reads as prominently as the label: same brightness,
	   slightly larger than chrome, tabular figures so digits don't jitter. */
	.num {
		width: 84px;
		text-align: right;
		font-size: 13px;
		color: var(--text);
		font-variant-numeric: tabular-nums;
	}
	input[type='text'],
	select {
		flex-grow: 1;
		min-width: 0;
		font-size: 13px;
		color: var(--text);
	}
	/* A trigger is an action, not a value — keep it quiet by default and let it
	   warm to the accent only on hover/press, so it sits in the same visual
	   register as the other controls instead of shouting over them. */
	.trigger {
		width: 100%;
		padding: 7px 12px;
		background: var(--bg-elev-3);
		border: 1px solid var(--border-strong);
		color: var(--text);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.trigger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 14%, var(--bg-elev-3));
		border-color: color-mix(in srgb, var(--accent) 55%, var(--border-strong));
	}
	.trigger:active:not(:disabled) {
		background: var(--accent);
		border-color: var(--accent);
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
