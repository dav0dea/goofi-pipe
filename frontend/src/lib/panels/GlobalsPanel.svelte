<!-- Globals panel — a key/value table over the patch's globals, one collapsed section per group.
     System globals are editable in value but locked for delete/rename; an element owned by a
     control panel is read-only here, because that panel owns it. -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import { groupedGlobals, isValidIdentifier, type GlobalType, type GlobalView } from '$lib/crdt/graphDoc';
	import {
		Button,
		Disclosure,
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
	const groups = $derived(groupedGlobals(globals));

	// The group an add row is open in; `''` is a row for a NEW group, and null is none.
	let adding = $state<string | null>(null);
	let newGroup = $state('');
	let newName = $state('');
	let newType = $state<GlobalType>('float');
	const targetGroup = $derived(adding === '' ? newGroup : (adding ?? ''));
	const fullName = $derived(`${targetGroup}.${newName}`);
	const nameTaken = $derived(globals.some((gv) => gv.name === fullName));
	const nameOk = $derived(isValidIdentifier(targetGroup) && isValidIdentifier(newName));
	const canAdd = $derived(adding !== null && nameOk && !nameTaken);

	// A group starts closed: a patch has many, and the panel is a list of them, not of every value.
	let open = $state<Record<string, boolean>>({});

	function zeroFor(type: GlobalType): number | string | boolean {
		return type === 'bool' ? false : type === 'string' ? '' : 0;
	}

	function openAdd(group: string): void {
		adding = group;
		newGroup = '';
		newName = '';
	}

	function focusInput(el: HTMLInputElement): void {
		el.focus();
	}

	function rowKey(e: KeyboardEvent): void {
		if (e.key === 'Enter') void add();
		else if (e.key === 'Escape') adding = null;
	}

	async function add(): Promise<void> {
		if (!canAdd) return;
		const group = targetGroup;
		try {
			await g.addGlobal(fullName, zeroFor(newType), newType);
			open[group] = true;
			adding = null;
		} catch {
			/* server rejected (invalid name / collision) — keep the row for correction */
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
		if (next === gv.element) return;
		void g.renameGlobal(gv.name, `${gv.group}.${next}`).catch(() => {
			/* rejected — the field reverts to gv.name on the next mirror-back render */
		});
	}

	function numberDisplay(gv: GlobalView): number {
		return typeof gv.value === 'number' ? gv.value : 0;
	}

	function commitGroup(from: string, raw: string): void {
		const next = raw.trim();
		if (next === from) return;
		void g.renameGlobalGroup(from, next).catch(() => {
			/* rejected — the header reverts on the next mirror-back render */
		});
	}
</script>

<div class="wrap" data-testid="globals-panel">
	<ScrollArea>
		<div class="gp-body">
			{#each groups as grp (grp.group)}
				<Disclosure
					class="grp"
					data-testid="global-group"
					data-group={grp.group}
					open={open[grp.group] === true}
					onToggle={(v) => (open[grp.group] = v)}
				>
					{#snippet summary()}
						<span class="grp-name">{grp.group}</span>
						<span class="grp-count">{grp.entries.length}</span>
					{/snippet}
					<table>
						<tbody>
							{#each grp.entries as gv (gv.name)}
								<tr
									data-testid="global-row"
									data-name={gv.name}
									data-system={gv.system}
									data-control={gv.control ? gv.control.kind : undefined}
								>
									<td class="c-name">
										{#if gv.system || gv.control}
											<span
												class="sysname"
												title={gv.control
													? 'Control element — the control panel owns it'
													: gv.locked
														? 'Machine global — the value is this machine\u2019s, read-only'
														: 'System global — value editable, name locked'}
											>
												<span class="lock" aria-hidden="true">🔒</span>{gv.element}
											</span>
										{:else}
											<TextInput
												inputmode="search"
												data-testid="global-name"
												value={gv.element}
												autocomplete="off"
												onChange={(v) => commitName(gv, v)}
											/>
										{/if}
									</td>
									<td class="c-val">
										{#if gv.locked || gv.control}
											<span class="ro-value" data-testid="global-value">{String(gv.value)}</span>
										{:else if gv.type === 'bool'}
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
										{#if !gv.system && !gv.control}
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
					<div class="grp-foot">
						{#if grp.entries.every((e) => !e.system)}
							<TextInput
								inputmode="search"
								data-testid="global-group-name"
								title="Rename this group"
								value={grp.group}
								autocomplete="off"
								onChange={(v) => commitGroup(grp.group, v)}
							/>
						{:else}
							<span class="grp-fixed">{grp.group}</span>
						{/if}
						<IconButton
							variant="ghost"
							size="sm"
							data-testid="global-add-in"
							title="Add a global to {grp.group}"
							label="Add a global to {grp.group}"
							onclick={() => openAdd(grp.group)}><Icon name="plus" /></IconButton
						>
					</div>
					{#if adding === grp.group}
						{@render addRow(false)}
					{/if}
				</Disclosure>
			{/each}

			<div class="new-group">
				<IconButton
					variant="ghost"
					size="sm"
					data-testid="global-add-group-btn"
					title="New group"
					label="New group"
					onclick={() => openAdd('')}><Icon name="plus" /></IconButton
				>
				<span class="new-group-hint">group</span>
			</div>
			{#if adding === ''}
				{@render addRow(true)}
			{/if}
		</div>
	</ScrollArea>
</div>

{#snippet addRow(fresh: boolean)}
	<div class="add" data-testid="global-add">
		{#if fresh}
			<input
				{...MODE_ATTRS.search}
				class="name"
				data-testid="global-add-group"
				placeholder="group"
				bind:value={newGroup}
				autocomplete="off"
				use:focusInput
				onkeydown={rowKey}
			/>
		{/if}
		{#if fresh}
			<input
				{...MODE_ATTRS.search}
				class="name"
				data-testid="global-add-name"
				placeholder="element"
				bind:value={newName}
				autocomplete="off"
				onkeydown={rowKey}
			/>
		{:else}
			<input
				{...MODE_ATTRS.search}
				class="name"
				data-testid="global-add-name"
				placeholder="element"
				bind:value={newName}
				autocomplete="off"
				use:focusInput
				onkeydown={rowKey}
			/>
		{/if}
		<Select
			style="flex: 0 1 auto"
			data-testid="global-add-type"
			value={newType}
			onChange={(v) => (newType = v as GlobalType)}
			options={['float', 'int', 'bool', 'string']}
		/>
		<Button size="sm" data-testid="global-add-btn" disabled={!canAdd} onclick={add}>Add</Button>
		<IconButton
			variant="ghost"
			size="sm"
			data-testid="global-add-cancel"
			title="Cancel"
			label="Cancel"
			onclick={() => (adding = null)}><Icon name="x" /></IconButton
		>
	</div>
	{#if newName && !nameOk}
		<div class="hint bad">Every global is `group.element`, each a valid identifier (letters, digits, _; can't start with a digit; not “globals”).</div>
	{:else if nameTaken}
		<div class="hint bad">A global named “{fullName}” already exists.</div>
	{/if}
{/snippet}

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
	.grp-name {
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.grp-count {
		margin-left: var(--space-2);
		color: var(--text-muted);
		font-size: var(--fs-micro);
	}
	.grp-foot {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-3);
	}
	.grp-fixed {
		flex: 1;
		min-width: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text-muted);
	}
	.new-group {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin-top: var(--space-4);
		padding: 0 var(--space-3);
	}
	.new-group-hint {
		font-size: var(--fs-micro);
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.06em;
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
	.ro-value {
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text-muted);
		overflow-wrap: anywhere;
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
		padding: var(--space-2) var(--space-3);
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
