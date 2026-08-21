<!-- Globals panel — a key/value table over the patch's globals. System globals are editable in
     value but locked for delete/rename. -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import { isValidGlobalName, type GlobalType, type GlobalView } from '$lib/crdt/graphDoc';
	import {
		Button,
		Icon,
		IconButton,
		MODE_ATTRS,
		NumberInput,
		ScrollArea,
		Select,
		TextInput,
		Toggle
	} from '$lib/ui';

	// Nothing of the panel contract is read, but it must be DECLARED: without it the inferred
	// props type is `{}` and the registry (`Component<PanelProps>`) won't take this component.
	let {}: PanelProps = $props();
	const g = graph();
	const globals = $derived(g.globals);

	let newName = $state('');
	let newType = $state<GlobalType>('float');
	const nameTaken = $derived(globals.some((gv) => gv.name === newName));
	const canAdd = $derived(isValidGlobalName(newName) && !nameTaken);

	function zeroFor(type: GlobalType): number | string | boolean {
		return type === 'bool' ? false : type === 'string' ? '' : 0;
	}

	async function add(): Promise<void> {
		if (!canAdd) return;
		try {
			await g.addGlobal(newName, zeroFor(newType), newType);
			newName = '';
		} catch {
			/* server rejected (invalid name / collision) — keep the field for correction */
		}
	}

	function commitValue(gv: GlobalView, raw: string | number | boolean): void {
		let val: number | string | boolean;
		if (gv.type === 'bool') val = raw === true;
		else if (gv.type === 'string') val = String(raw);
		else {
			const n = Number(raw);
			if (!Number.isFinite(n)) return;
			val = gv.type === 'int' ? Math.round(n) : n;
		}
		void g.setGlobalValue(gv.name, val).catch(() => {
			/* rejected — the input reverts on the next mirror-back render */
		});
	}

	function commitName(gv: GlobalView, raw: string): void {
		const next = raw.trim();
		if (next === gv.name) return;
		void g.renameGlobal(gv.name, next).catch(() => {
			/* rejected — the field reverts to gv.name on the next mirror-back render */
		});
	}

	function numberDisplay(gv: GlobalView): number {
		return typeof gv.value === 'number' ? gv.value : 0;
	}
</script>

<div class="wrap" data-testid="globals-panel">
	<ScrollArea>
		<div class="gp-body">
			<table>
				<thead>
					<tr>
						<th class="c-name">Name</th>
						<th class="c-val">Value</th>
						<th class="c-act" aria-label="Actions"></th>
					</tr>
				</thead>
				<tbody>
					{#each globals as gv (gv.name)}
						<tr data-testid="global-row" data-name={gv.name} data-system={gv.system}>
							<td class="c-name">
								{#if gv.system}
									<span class="sysname" title="System global — value editable, name locked">
										<span class="lock" aria-hidden="true">🔒</span>{gv.name}
									</span>
								{:else}
									<TextInput
										inputmode="search"
										data-testid="global-name"
										value={gv.name}
										autocomplete="off"
										onChange={(v) => commitName(gv, v)}
									/>
								{/if}
							</td>
							<td class="c-val">
								{#if gv.type === 'bool'}
									<Toggle
										data-testid="global-value"
										value={gv.value === true}
										onChange={(v) => commitValue(gv, v)}
									/>
								{:else if gv.type === 'string'}
									<!-- Machine-read: the `text` default's autocorrect would corrupt a good value. -->
									<TextInput
										inputmode="search"
										data-testid="global-value"
										value={String(gv.value)}
										autocomplete="off"
										onChange={(v) => commitValue(gv, v)}
									/>
								{:else}
									<NumberInput
										data-testid="global-value"
										value={numberDisplay(gv)}
										onChange={(v) => commitValue(gv, v)}
									/>
								{/if}
							</td>
							<td class="c-act">
								<span class="type" title="type">{gv.type}</span>
								{#if !gv.system}
									<IconButton
										variant="ghost"
										size="sm"
										data-testid="global-delete"
										title="Delete global"
										label="Delete {gv.name}"
										onclick={() => void g.removeGlobal(gv.name)}><Icon name="x" /></IconButton
									>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>

			<div class="add" data-testid="global-add">
				<input
					{...MODE_ATTRS.search}
					class="name"
					data-testid="global-add-name"
					placeholder="new_global"
					bind:value={newName}
					autocomplete="off"
					onkeydown={(e) => {
						if (e.key === 'Enter') add();
					}}
				/>
				<Select
					style="flex: 0 1 auto"
					data-testid="global-add-type"
					value={newType}
					onChange={(v) => (newType = v as GlobalType)}
					options={['float', 'int', 'bool', 'string']}
				/>
				<Button size="sm" data-testid="global-add-btn" disabled={!canAdd} onclick={add}>Add</Button>
			</div>
			{#if newName && !isValidGlobalName(newName)}
				<div class="hint bad">Not a valid identifier (letters, digits, _; can't start with a digit; not “globals”).</div>
			{:else if nameTaken}
				<div class="hint bad">A global named “{newName}” already exists.</div>
			{/if}
		</div>
	</ScrollArea>
</div>

<style>
	.wrap {
		height: 100%;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
	.gp-body {
		padding: var(--space-3) var(--space-5) var(--space-6);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--fs-small);
	}
	th {
		text-align: left;
		font-weight: 500;
		color: var(--text-muted);
		font-size: var(--fs-micro);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: var(--space-1) var(--space-3) var(--space-3);
		border-bottom: 1px solid var(--border);
	}
	td {
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
		vertical-align: middle;
	}
	/* Stated on the CELLS: the ui inputs carry `font: inherit`, so the cell hands them a face. */
	td.c-name,
	td.c-val {
		font-family: var(--font-mono);
	}
	.c-name {
		width: 45%;
	}
	.c-val {
		width: 40%;
		/* let the bare NumberInput fill the cell instead of its default fixed width */
		--number-width: 100%;
	}
	.c-act {
		width: 15%;
		white-space: nowrap;
		text-align: right;
	}
	.sysname {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		font-family: var(--font-mono);
		color: var(--text);
	}
	.lock {
		font-size: var(--fs-micro);
		filter: grayscale(1);
		/* Not `--disabled-opacity`: the padlock is a quiet affordance, not a disabled control. */
		opacity: 0.7;
	}
	/* The one native input, kept for live per-keystroke validation; the `td` seam cannot reach it. */
	input.name {
		width: 100%;
		box-sizing: border-box;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		padding: var(--space-1) var(--space-3);
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	.type {
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-muted);
		margin-right: var(--space-3);
	}
	.add {
		display: flex;
		gap: var(--space-3);
		align-items: center;
		margin-top: var(--space-6);
	}
	.add .name {
		flex: 1 1 auto;
	}
	.hint {
		margin-top: var(--space-3);
		font-size: var(--fs-micro);
	}
	.hint.bad {
		color: var(--danger);
	}
	/* iOS force-zooms a focused control under 16px, and `input.name` out-specifies app.css's floor. */
	@media (hover: none) and (pointer: coarse) {
		input.name {
			font-size: 16px;
		}
	}
</style>
