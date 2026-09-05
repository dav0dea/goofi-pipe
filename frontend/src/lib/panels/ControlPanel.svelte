<!-- Control panel — knobs, sliders and fields over ONE group of globals. Out of edit mode a drag
     turns a widget; in edit mode the same drag moves it and the handle resizes it. Every change is
     a globals op, so the manager owns the state and this panel owns only the drawing. -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import { asStateObject } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import type { ControlView, GlobalType, GlobalView } from '$lib/crdt/graphDoc';
	import { isValidIdentifier } from '$lib/crdt/graphDoc';
	import {
		Button,
		Icon,
		IconButton,
		Knob,
		NumberInput,
		ScrollArea,
		Select,
		Slider,
		TextInput,
		Toggle
	} from '$lib/ui';
	import { COLUMNS, freeCell, movedBy, resizedBy, type Cell } from './controlLayout';

	interface ControlState {
		group?: string;
		edit?: boolean;
	}

	const KINDS = ['knob', 'slider', 'number', 'field', 'toggle', 'dropdown'] as const;
	/** The value type each widget draws, so the add form asks for one thing, not two. */
	const TYPE_OF: Record<(typeof KINDS)[number], GlobalType> = {
		knob: 'float',
		slider: 'float',
		number: 'float',
		field: 'string',
		toggle: 'bool',
		dropdown: 'string'
	};

	let props: PanelProps = $props();
	const g = graph();

	const st = $derived(asStateObject(props.state) as ControlState);
	const group = $derived(st.group ?? '');
	const edit = $derived(st.edit === true);
	const elements = $derived(g.globals.filter((gv) => gv.group === group && gv.control));

	let newName = $state('');
	let newKind: (typeof KINDS)[number] = $state('knob');
	const canAdd = $derived(
		isValidIdentifier(newName) && !g.globals.some((gv) => gv.name === `${group}.${newName}`)
	);

	// The unit is the panel's own width, so a widget keeps its share of the panel at every size.
	let board: HTMLDivElement | null = $state(null);
	const unit = $derived.by(() => (board ? board.clientWidth / COLUMNS : 48));

	let drag: { name: string; from: Cell; x: number; y: number; resize: boolean } | null = $state(null);

	function cellOf(gv: GlobalView): Cell {
		const c = gv.control as ControlView;
		return { x: c.x, y: c.y, w: c.w, h: c.h };
	}

	function setEdit(on: boolean): void {
		props.setState({ ...st, edit: on }, 'authored', on ? 'Edit control panel' : 'Done editing');
	}

	function commitCell(gv: GlobalView, cell: Cell): void {
		void g
			.setGlobalControl(gv.name, { ...(gv.control as ControlView), ...cell })
			.catch(() => {
				/* rejected — the widget springs back on the next mirror-back render */
			});
	}

	function setControl(gv: GlobalView, patch: Partial<ControlView>): void {
		void g.setGlobalControl(gv.name, { ...(gv.control as ControlView), ...patch }).catch(() => {
			/* rejected — the field reverts on the next mirror-back render */
		});
	}

	function down(e: PointerEvent, gv: GlobalView, resize: boolean): void {
		if (!edit) return;
		drag = { name: gv.name, from: cellOf(gv), x: e.clientX, y: e.clientY, resize };
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		e.preventDefault();
		e.stopPropagation();
	}

	function move(e: PointerEvent): void {
		if (!drag) return;
		const gv = elements.find((el) => el.name === drag!.name);
		if (!gv) return;
		const [dx, dy] = [e.clientX - drag.x, e.clientY - drag.y];
		commitCell(gv, drag.resize ? resizedBy(drag.from, dx, dy, unit) : movedBy(drag.from, dx, dy, unit));
	}

	function up(): void {
		drag = null;
	}

	async function add(): Promise<void> {
		if (!canAdd) return;
		const type = TYPE_OF[newKind];
		const cell = freeCell(elements.map(cellOf), 2, 2, COLUMNS);
		const control: ControlView = { kind: newKind, ...cell };
		if (type === 'float') {
			control.min = 0;
			control.max = 1;
			control.step = 0.01;
		}
		try {
			await g.addGlobal(`${group}.${newName}`, type === 'bool' ? false : type === 'string' ? '' : 0, type, control);
			newName = '';
		} catch {
			/* rejected (invalid name / collision) — keep the field for correction */
		}
	}

	function commitValue(gv: GlobalView, v: number | string | boolean): void {
		void g.setGlobalValue(gv.name, v).catch(() => {
			/* rejected — the widget reverts on the next mirror-back render */
		});
	}

	function num(gv: GlobalView): number {
		return typeof gv.value === 'number' ? gv.value : 0;
	}
</script>

<div class="wrap" data-testid="control-panel" data-group={group} data-edit={edit}>
	<div class="bar">
		{#if edit}
			<div class="grow">
				<TextInput
					inputmode="search"
					data-testid="control-group-name"
					value={group}
					autocomplete="off"
					onChange={(v) => void g.renameGlobalGroup(group, v.trim()).catch(() => {})}
				/>
			</div>
		{:else}
			<span class="title">{group}</span>
		{/if}
		<IconButton
			variant={edit ? 'primary' : 'ghost'}
			size="sm"
			data-testid="control-edit-toggle"
			title={edit ? 'Done editing' : 'Edit this panel'}
			label={edit ? 'Done editing' : 'Edit this panel'}
			onclick={() => setEdit(!edit)}><Icon name="settings" /></IconButton
		>
	</div>

	<ScrollArea>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="board"
			bind:this={board}
			style={`--unit: ${unit}px; --columns: ${COLUMNS}`}
			onpointermove={move}
			onpointerup={up}
			onpointercancel={up}
		>
			{#each elements as gv (gv.name)}
				{@const c = gv.control as ControlView}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="cell"
					data-testid={`control-${group}-${gv.element}`}
					style={`grid-column: ${c.x + 1} / span ${c.w}; grid-row: ${c.y + 1} / span ${c.h}`}
					onpointerdown={(e) => down(e, gv, false)}
				>
					<span class="label">{gv.element}</span>
					<div class="widget">
						{#if c.kind === 'knob'}
							<Knob
								label={gv.element}
								value={num(gv)}
								min={c.min ?? 0}
								max={c.max ?? 1}
								step={c.step ?? 0}
								disabled={edit}
								onChange={(v) => commitValue(gv, v)}
							/>
						{:else if c.kind === 'slider'}
							<Slider
								value={num(gv)}
								min={c.min ?? 0}
								max={c.max ?? 1}
								step={c.step}
								onChange={(v) => commitValue(gv, v)}
							/>
						{:else if c.kind === 'number'}
							<NumberInput value={num(gv)} onChange={(v) => commitValue(gv, v)} />
						{:else if c.kind === 'toggle'}
							<Toggle value={gv.value === true} onChange={(v) => commitValue(gv, v)} />
						{:else if c.kind === 'dropdown'}
							<Select
								value={String(gv.value)}
								options={c.options ?? []}
								onChange={(v) => commitValue(gv, v)}
							/>
						{:else}
							<TextInput
								inputmode="search"
								value={String(gv.value)}
								autocomplete="off"
								onChange={(v) => commitValue(gv, v)}
							/>
						{/if}
					</div>

					{#if edit}
						<div class="edit">
							<TextInput
								inputmode="search"
								data-testid="control-element-name"
								value={gv.element}
								autocomplete="off"
								onChange={(v) =>
									void g.renameGlobal(gv.name, `${group}.${v.trim()}`).catch(() => {})}
							/>
							<Select
								density="chrome"
								value={c.kind}
								options={KINDS.filter((k) => TYPE_OF[k] === gv.type)}
								onChange={(v) => setControl(gv, { kind: v as ControlView['kind'] })}
							/>
							{#if gv.type === 'float' || gv.type === 'int'}
								<NumberInput value={c.min ?? 0} onChange={(v) => setControl(gv, { min: v })} />
								<NumberInput value={c.max ?? 1} onChange={(v) => setControl(gv, { max: v })} />
							{/if}
							<IconButton
								variant="ghost"
								size="sm"
								data-testid="control-delete"
								title="Delete element"
								label="Delete {gv.element}"
								onclick={() => void g.removeGlobal(gv.name)}><Icon name="x" /></IconButton
							>
						</div>
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<span
							class="handle"
							data-testid="control-resize"
							onpointerdown={(e) => down(e, gv, true)}
						></span>
					{/if}
				</div>
			{/each}
		</div>
	</ScrollArea>

	{#if edit}
		<div class="add" data-testid="control-add">
			<TextInput
				inputmode="search"
				data-testid="control-add-name"
				value={newName}
				autocomplete="off"
				onChange={(v) => (newName = v)}
			/>
			<Select
				density="chrome"
				data-testid="control-add-kind"
				value={newKind}
				options={[...KINDS]}
				onChange={(v) => (newKind = v as (typeof KINDS)[number])}
			/>
			<Button size="sm" data-testid="control-add-btn" disabled={!canAdd} onclick={add}>Add</Button>
		</div>
	{/if}
</div>

<style>
	.wrap {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px solid var(--border);
	}
	.title {
		flex: 1;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.grow {
		flex: 1;
		min-width: 0;
	}
	.board {
		display: grid;
		grid-template-columns: repeat(var(--columns), 1fr);
		grid-auto-rows: var(--unit);
		gap: var(--space-1);
		padding: var(--space-3);
		touch-action: none;
	}
	.cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--space-1);
		min-width: 0;
		min-height: 0;
		position: relative;
		border-radius: var(--radius-2);
		background: var(--surface-1);
	}
	[data-edit='true'] .cell {
		outline: 1px dashed var(--border);
		cursor: grab;
	}
	.label {
		font-size: var(--fs-micro);
		color: var(--text-muted);
		font-family: var(--font-mono);
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.widget {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		min-width: 0;
		flex: 1;
	}
	.edit {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: var(--space-1);
		width: 100%;
		min-width: 0;
	}
	.handle {
		position: absolute;
		right: 0;
		bottom: 0;
		width: var(--space-4);
		height: var(--space-4);
		cursor: nwse-resize;
		background: var(--accent);
		border-radius: var(--radius-1) 0 var(--radius-2) 0;
		opacity: 0.7;
	}
	.add {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-3);
		border-top: 1px solid var(--border);
	}
</style>
