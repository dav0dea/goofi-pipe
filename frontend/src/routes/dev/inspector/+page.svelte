<!--
  /dev/inspector — the backend-free ParamField gallery (spec §7, N-Task 2). Mirrors /dev/ui: a static
  showcase, one `<ParamField>` per non-expression control kind against SYNTHETIC fx-OFF descriptors,
  each with a committed-value read-out the committed `tests/e2e` inspector-gallery spec drives. fx (the
  'expression' kind) is N-Task 3 and not exercised here. Each sample carries a distinct data-testid.
-->
<script lang="ts">
	import ParamField from '$lib/inspector/ParamField.svelte';
	import type { ParamDescriptor } from '$lib/api/types';

	// The fx-OFF envelope shared by every synthetic descriptor (Task 2 exercises no expression state).
	const fxOff = {
		doc: null,
		refreshable: false,
		expression: null,
		expression_enabled: false,
		expression_triggers_process: false,
		expression_error: null
	} as const;

	// --- committed state, one per kind; each read-out is bound to it so the e2e observes the commit. ---
	let floatVal = $state(0.3);
	let intVal = $state(5);
	let boolVal = $state(false);
	let triggerCount = $state(0);
	let textVal = $state('hello');
	let optionVal = $state('sine');
	// A stale-but-live device id, absent from the seeded options below — the P Select prepends it.
	let deviceVal = $state('mic-1');
	const unknownVal = { channels: ['Fz', 'Cz'], n: 2 };

	// The refreshable select's live option set + the ⟳ re-scan simulation.
	let deviceOptions = $state<string[]>(['line-in', 'hdmi']);
	let refreshing = $state(false);
	let refreshCount = $state(0);

	function refreshDevices(): void {
		refreshing = true;
		refreshCount += 1;
		// A brief spin, then a fresh option set the seed list did not have (proves the re-scan landed).
		setTimeout(() => {
			deviceOptions = ['line-in', 'hdmi', `scanned-${refreshCount}`];
			refreshing = false;
		}, 80);
	}

	// The fx commit callback the contract requires; the non-expression samples above never enable it.
	const noExpr = (): void => {};

	// --- fx (expression) sample — starts OFF; every expression flag is a mutable read-out so the e2e
	// observes each toggle/commit. onSetExpression writes the flags the way the store would. ---
	let fxValue = $state(0.5);
	let fxExpr = $state<string | null>(null);
	let fxEnabled = $state(false);
	let fxTriggers = $state(false);

	function fxSetExpression(
		expression: string | null,
		opts?: { enabled?: boolean; triggers_process?: boolean }
	): void {
		fxExpr = expression;
		if (opts?.enabled !== undefined) fxEnabled = opts.enabled;
		if (opts?.triggers_process !== undefined) fxTriggers = opts.triggers_process;
	}

	const fxDesc = $derived<ParamDescriptor>({
		...fxOff,
		type: 'float',
		value: fxValue,
		vmin: 0,
		vmax: 1,
		doc: 'gain',
		expression: fxExpr,
		expression_enabled: fxEnabled,
		expression_triggers_process: fxTriggers
	});

	// A fixed errored expression — the fx chip goes danger-toned, the message lands in the error row.
	const fxErrorDesc: ParamDescriptor = {
		...fxOff,
		type: 'float',
		value: 0,
		vmin: 0,
		vmax: 1,
		expression: 'nd("missing").out.data',
		expression_enabled: true,
		expression_error: 'NameError: name "nd" is not defined'
	};

	// Descriptors are derived so a commit flows back into the control (echo) as well as the read-out.
	const floatDesc = $derived<ParamDescriptor>({
		...fxOff,
		type: 'float',
		value: floatVal,
		vmin: 0,
		vmax: 1,
		doc: 'filter cutoff'
	});
	const intDesc = $derived<ParamDescriptor>({ ...fxOff, type: 'int', value: intVal, vmin: 0, vmax: 16 });
	const boolDesc = $derived<ParamDescriptor>({ ...fxOff, type: 'bool', value: boolVal, trigger: false });
	const triggerDesc = $derived<ParamDescriptor>({ ...fxOff, type: 'bool', value: false, trigger: true });
	const textDesc = $derived<ParamDescriptor>({ ...fxOff, type: 'string', value: textVal, options: null });
	const optionsDesc = $derived<ParamDescriptor>({
		...fxOff,
		type: 'string',
		value: optionVal,
		options: ['sine', 'square', 'saw', 'triangle']
	});
	const deviceDesc = $derived<ParamDescriptor>({
		...fxOff,
		type: 'string',
		value: deviceVal,
		options: deviceOptions,
		refreshable: true
	});
	const unknownDesc = $derived<ParamDescriptor>({ ...fxOff, type: 'unknown', value: unknownVal });
</script>

<main class="gallery">
	<h1>Inspector fields</h1>

	<section>
		<h2>numeric — float (with bounds)</h2>
		<div class="form">
			<ParamField
				paramName="cutoff"
				descriptor={floatDesc}
				onCommit={(v) => (floatVal = Number(v))}
				onSetExpression={noExpr}
				data-testid="inspector-float"
			/>
			<span class="readout" data-testid="inspector-float-value">{floatVal}</span>
		</div>
	</section>

	<section>
		<h2>numeric — int</h2>
		<div class="form">
			<ParamField
				paramName="channels"
				descriptor={intDesc}
				onCommit={(v) => (intVal = Number(v))}
				onSetExpression={noExpr}
				data-testid="inspector-int"
			/>
			<span class="readout" data-testid="inspector-int-value">{intVal}</span>
		</div>
	</section>

	<section>
		<h2>toggle — bool</h2>
		<div class="form">
			<ParamField
				paramName="enabled"
				descriptor={boolDesc}
				onCommit={(v) => (boolVal = Boolean(v))}
				onSetExpression={noExpr}
				data-testid="inspector-bool"
			/>
			<span class="readout" data-testid="inspector-bool-value">{boolVal}</span>
		</div>
	</section>

	<section>
		<h2>trigger — bool + trigger</h2>
		<div class="form">
			<ParamField
				paramName="reset"
				descriptor={triggerDesc}
				onCommit={() => (triggerCount += 1)}
				onSetExpression={noExpr}
				data-testid="inspector-trigger"
			/>
			<span class="readout" data-testid="inspector-trigger-value">{triggerCount}</span>
		</div>
	</section>

	<section>
		<h2>text — string</h2>
		<div class="form">
			<ParamField
				paramName="label"
				descriptor={textDesc}
				onCommit={(v) => (textVal = String(v))}
				onSetExpression={noExpr}
				data-testid="inspector-text"
			/>
			<span class="readout" data-testid="inspector-text-value">{textVal}</span>
		</div>
	</section>

	<section>
		<h2>select — string with options</h2>
		<div class="form">
			<ParamField
				paramName="waveform"
				descriptor={optionsDesc}
				onCommit={(v) => (optionVal = String(v))}
				onSetExpression={noExpr}
				data-testid="inspector-options"
			/>
			<span class="readout" data-testid="inspector-options-value">{optionVal}</span>
		</div>
	</section>

	<section>
		<h2>select — refreshable string (stale value + ⟳)</h2>
		<div class="form">
			<ParamField
				paramName="device"
				descriptor={deviceDesc}
				onCommit={(v) => (deviceVal = String(v))}
				onSetExpression={noExpr}
				onRefresh={refreshDevices}
				{refreshing}
				data-testid="inspector-device"
			/>
			<span class="readout" data-testid="inspector-device-value">{deviceVal}</span>
			<span class="readout" data-testid="inspector-device-refreshes">{refreshCount}</span>
		</div>
	</section>

	<section>
		<h2>unknown — read-only</h2>
		<div class="form">
			<ParamField
				paramName="raw"
				descriptor={unknownDesc}
				onCommit={() => {}}
				onSetExpression={noExpr}
				data-testid="inspector-unknown"
			/>
		</div>
	</section>

	<section>
		<h2>fx — expression binding (off → on)</h2>
		<div class="form">
			<ParamField
				paramName="gain"
				descriptor={fxDesc}
				onCommit={(v) => (fxValue = Number(v))}
				onSetExpression={fxSetExpression}
				data-testid="inspector-fx"
			/>
			<span class="readout" data-testid="inspector-fx-expr">{fxExpr ?? 'null'}</span>
			<span class="readout" data-testid="inspector-fx-enabled">{fxEnabled}</span>
			<span class="readout" data-testid="inspector-fx-triggers">{fxTriggers}</span>
		</div>
	</section>

	<section>
		<h2>fx — expression error</h2>
		<div class="form">
			<ParamField
				paramName="broken"
				descriptor={fxErrorDesc}
				onCommit={() => {}}
				onSetExpression={noExpr}
				data-testid="inspector-fx-error"
			/>
		</div>
	</section>
</main>

<style>
	.gallery {
		box-sizing: border-box;
		height: 100vh;
		overflow-y: auto;
		padding: var(--space-8);
		display: flex;
		flex-direction: column;
		gap: var(--space-8);
	}
	h1 {
		margin: 0;
		font-size: var(--fs-title);
		color: var(--text);
	}
	h2 {
		margin: 0 0 var(--space-4);
		font-size: var(--fs-strong);
		color: var(--text-dim);
	}
	/* A panel-width form so the labelled controls size realistically (they fill their Field). */
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
		width: 18rem;
		max-width: 100%;
	}
	/* A committed-value read-out the e2e reads to observe commit timing/values. */
	.readout {
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}
</style>
