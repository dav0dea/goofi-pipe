<!-- Globals panel — a two-column key/value table over the patch's globals (the CRDT
     `globals` root, doc-authoritative via graph().globals). Each row edits a named
     typed scalar that expressions read as `globals.<name>` and node process/setup read
     from `ctx.globals`.

     System globals (e.g. `default_ufreq`) are editable in value but locked for
     delete/rename (badged 🔒); user globals can be added, renamed, retyped-by-recreation,
     and removed. All edits are command RPCs (add_global / set_global / rename_global /
     remove_global); the manager applies each and mirrors the result back into the doc,
     which this panel reads. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import { graph } from '$lib/stores/graph.svelte';
	import { isValidGlobalName, type GlobalType, type GlobalView } from '$lib/crdt/graphDoc';

	let { active }: PanelProps = $props();
	const g = graph();
	const globals = $derived(g.globals);

	// Add-row draft. A new global is created with a type-appropriate zero value; the user
	// then edits the value inline. Kept minimal so the common case (name + Add) is one field.
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

	// Commit a value edit, parsing the raw widget value into the global's declared type.
	// A non-numeric entry into a number type is rejected locally; a server rejection snaps the
	// input back to the committed value on the next render (the mirrored doc is the source of truth).
	function commitValue(gv: GlobalView, raw: string | boolean): void {
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

	// Commit a rename (user globals only). On rejection (invalid / collision / system) the input is
	// snapped back to the current name — the store is the source of truth.
	function commitName(gv: GlobalView, input: HTMLInputElement): void {
		const next = input.value.trim();
		if (next === gv.name) return;
		void g.renameGlobal(gv.name, next).catch(() => {
			input.value = gv.name;
		});
	}

	function numberDisplay(gv: GlobalView): number {
		return typeof gv.value === 'number' ? gv.value : 0;
	}
</script>

<div class="wrap" class:active data-testid="globals-panel">
	<div class="scroll">
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
								<input
									class="name"
									data-testid="global-name"
									value={gv.name}
									spellcheck="false"
									autocomplete="off"
									onchange={(e) => commitName(gv, e.currentTarget)}
								/>
							{/if}
						</td>
						<td class="c-val">
							{#if gv.type === 'bool'}
								<input
									type="checkbox"
									data-testid="global-value"
									checked={gv.value === true}
									onchange={(e) => commitValue(gv, e.currentTarget.checked)}
								/>
							{:else if gv.type === 'string'}
								<input
									class="val"
									type="text"
									data-testid="global-value"
									value={String(gv.value)}
									spellcheck="false"
									autocomplete="off"
									onchange={(e) => commitValue(gv, e.currentTarget.value)}
								/>
							{:else}
								<input
									class="val"
									type="number"
									data-testid="global-value"
									step={gv.type === 'int' ? '1' : 'any'}
									value={numberDisplay(gv)}
									onchange={(e) => commitValue(gv, e.currentTarget.value)}
								/>
							{/if}
						</td>
						<td class="c-act">
							<span class="type" title="type">{gv.type}</span>
							{#if !gv.system}
								<button
									class="del"
									data-testid="global-delete"
									title="Delete global"
									aria-label="Delete {gv.name}"
									onclick={() => void g.removeGlobal(gv.name)}>✕</button
								>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>

		<div class="add" data-testid="global-add">
			<input
				class="name"
				data-testid="global-add-name"
				placeholder="new_global"
				bind:value={newName}
				spellcheck="false"
				autocomplete="off"
				onkeydown={(e) => {
					if (e.key === 'Enter') add();
				}}
			/>
			<select class="type-sel" data-testid="global-add-type" bind:value={newType}>
				<option value="float">float</option>
				<option value="int">int</option>
				<option value="bool">bool</option>
				<option value="string">string</option>
			</select>
			<button class="addbtn" data-testid="global-add-btn" disabled={!canAdd} onclick={add}>Add</button>
		</div>
		{#if newName && !isValidGlobalName(newName)}
			<div class="hint bad">Not a valid identifier (letters, digits, _; can't start with a digit; not “globals”).</div>
		{:else if nameTaken}
			<div class="hint bad">A global named “{newName}” already exists.</div>
		{/if}
	</div>
</div>

<style>
	.wrap {
		position: relative;
		height: 100%;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}
	/* Active-panel accent, drawn as an overlay around the content so its top edge sits
	   flush under PanelHeader (not clipped behind it, unlike the panel-frame outline).
	   Mirrors NodeLinkedPanel: square at the top, rounded at the panel's bottom corners. */
	.wrap.active::after {
		content: '';
		position: absolute;
		inset: 0;
		pointer-events: none;
		border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
		border-radius: 0 0 var(--radius-sm) var(--radius-sm);
		z-index: 4;
	}
	.scroll {
		flex: 1;
		overflow-y: auto;
		min-height: 0;
		padding: 6px 8px 10px;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
	}
	th {
		text-align: left;
		font-weight: 500;
		color: var(--text-muted);
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 2px 6px 6px;
		border-bottom: 1px solid var(--border);
	}
	td {
		padding: 3px 6px;
		border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
		vertical-align: middle;
	}
	.c-name {
		width: 45%;
	}
	.c-val {
		width: 40%;
	}
	.c-act {
		width: 15%;
		white-space: nowrap;
		text-align: right;
	}
	.sysname {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-family: var(--font-mono);
		color: var(--text);
	}
	.lock {
		font-size: 0.7rem;
		filter: grayscale(1);
		opacity: 0.7;
	}
	input.name,
	input.val {
		width: 100%;
		box-sizing: border-box;
		font-family: var(--font-mono);
		font-size: 0.78rem;
		padding: 2px 5px;
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	input.name:focus,
	input.val:focus {
		outline: none;
		border-color: var(--accent);
	}
	input[type='number'] {
		text-align: right;
	}
	.type {
		font-family: var(--font-mono);
		font-size: 0.66rem;
		color: var(--text-muted);
		margin-right: 6px;
	}
	.del {
		width: 18px;
		height: 18px;
		display: inline-grid;
		place-items: center;
		padding: 0;
		font-size: 0.7rem;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
	}
	.del:hover {
		color: var(--danger);
		background: var(--surface-2);
	}
	.add {
		display: flex;
		gap: 6px;
		align-items: center;
		margin-top: 10px;
	}
	.add .name {
		flex: 1 1 auto;
	}
	.type-sel {
		font-family: var(--font-mono);
		font-size: 0.76rem;
		padding: 2px 4px;
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	.addbtn {
		font-size: 0.76rem;
		padding: 3px 12px;
		background: var(--surface-3);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
	}
	.addbtn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.addbtn:not(:disabled):hover {
		border-color: var(--accent);
		color: var(--accent);
	}
	.hint {
		margin-top: 6px;
		font-size: 0.72rem;
	}
	.hint.bad {
		color: var(--danger);
	}
</style>
