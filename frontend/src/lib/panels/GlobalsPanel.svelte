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

	// The globals are patch-scoped, not panel-scoped, so this panel reads none of the panel
	// contract — but it must still DECLARE it, or its inferred props type is `{}` and the
	// registry (`Component<PanelProps>`) won't take it.
	let {}: PanelProps = $props();
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

	// Commit a rename (user globals only). On rejection (invalid / collision / system) the store keeps
	// the current name, so the control's live-value latch snaps the field back on the next render.
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
									<!-- Machine-read, like the name beside it: expressions resolve `globals.<name>`
									     and nodes read `ctx.globals`, so the `text` default's autocorrect /
									     sentence-capitalisation / spellcheck would corrupt or red-underline a
									     perfectly good value. -->
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
		/* opacity: intentional — the padlock is a quiet affordance beside the name, not a
		   disabled control; --disabled-opacity would read as "this row is inert". */
		opacity: 0.7;
	}
	/* The add-row name field is the one remaining native input (kept native for its live
	   per-keystroke validation + Enter-to-add); the table cells are now ui primitives. */
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
	/* Touch: iOS force-zooms the page when a control under 16px takes focus, and `input.name`
	   (0,1,1) out-specifies app.css's `input` floor (0,0,1) — so this one field defeated it. The
	   threshold is absolute, not a type rung, which is why it is a literal here as it is there. */
	@media (hover: none) and (pointer: coarse) {
		input.name {
			font-size: 16px;
		}
	}
</style>
