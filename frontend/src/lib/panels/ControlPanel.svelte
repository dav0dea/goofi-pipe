<!-- Control panel — knobs, sliders and fields over ONE group of globals. Edit mode is the group's
     config lock, inverted: out of it a drag turns a widget; in it the same drag moves it, the
     corner resizes it, and the inspector slides in with the palette and the picked widget's form.
     Every change is a globals op, so the manager owns the state and this panel owns only the
     drawing and the gesture in flight. -->
<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import type { PanelProps } from 'panelty';
	import { asStateObject } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import type { ControlView, GlobalView, LockView } from '$lib/crdt/graphDoc';
	import { effectiveLock, isValidIdentifier } from '$lib/crdt/graphDoc';
	import type { ParamDescriptor, SourcePatch } from '$lib/api/types';
	import { bindViewer } from '$lib/api/frames';
	import type { ArrayData, DataFrame } from '$lib/codec/decode';
	import { viewSpecForKind } from '$lib/viewers/capacity';
	import ParamField from '$lib/inspector/ParamField.svelte';
	import SidePane from './SidePane.svelte';
	import {
		Chip,
		EmptyState,
		Field,
		Icon,
		IconButton,
		Knob,
		NumberInput,
		ScrollArea,
		Select,
		Slider,
		TextInput,
		Toggle,
		isTextEditingTarget
	} from '$lib/ui';
	import {
		BORN,
		COLUMNS,
		KINDS,
		TYPE_OF,
		cellAt,
		movedBy,
		resizedBy,
		sameCell,
		type Cell,
		type Kind,
		type Units
	} from './controlLayout';

	interface ControlState {
		group?: string;
	}

	type Value = number | string | boolean;

	let props: PanelProps = $props();
	const g = graph();

	const st = $derived(asStateObject(props.state) as ControlState);
	const group = $derived(st.group ?? '');
	const named = $derived(group !== '');
	const groupLock = $derived<LockView>(g.globalGroups[group] ?? { config: false, value: false });
	const edit = $derived(named && !groupLock.config);
	const elements = $derived(g.globals.filter((gv) => gv.group === group && gv.control));

	let board: HTMLDivElement | null = $state(null);
	let picked = $state<string | null>(null);
	let renaming = $state<string | null>(null);
	const pickedView = $derived(elements.find((el) => el.name === picked) ?? null);

	// The two gestures in flight: a widget being moved or resized, and a chip lifted off the palette.
	let drag: { name: string; from: Cell; x: number; y: number; units: Units; resize: boolean; to: Cell | null } | null =
		$state(null);
	let lift: { kind: Kind; x: number; y: number; w: number; h: number; from: { x: number; y: number } } | null = $state(null);
	// A drop's cell outlives the pointer until the document agrees, so it never flashes back.
	let pending: { name: string; to: Cell } | null = $state(null);
	$effect(() => {
		const p = pending;
		if (p && elements.some((el) => el.name === p.name && sameCell(cellOf(el), p.to))) pending = null;
	});

	function cellOf(gv: GlobalView): Cell {
		const c = gv.control as ControlView;
		return { x: c.x, y: c.y, w: c.w, h: c.h };
	}

	function placed(gv: GlobalView): Cell {
		if (drag?.name === gv.name && drag.to) return drag.to;
		if (pending?.name === gv.name) return pending.to;
		return cellOf(gv);
	}

	function unitsOf(el: HTMLElement): Units & { gap: number } {
		const cs = getComputedStyle(el);
		const gap = parseFloat(cs.rowGap) || 0;
		const inner = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
		const x = (inner + gap) / COLUMNS;
		const row = parseFloat(cs.gridAutoRows);
		return { x, y: Number.isFinite(row) ? row + gap : x, gap };
	}

	function onBoard(e: PointerEvent): { x: number; y: number } | null {
		if (!board) return null;
		const r = board.getBoundingClientRect();
		if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return null;
		const cs = getComputedStyle(board);
		return { x: e.clientX - r.left - parseFloat(cs.paddingLeft), y: e.clientY - r.top - parseFloat(cs.paddingTop) };
	}

	function nameGroup(raw: string): void {
		const to = raw.trim();
		if (to === group || !isValidIdentifier(to)) return;
		void g.renameGlobalGroup(group, to).catch(() => {});
	}

	function setEdit(on: boolean): void {
		picked = null;
		renaming = null;
		stopLearning();
		void g.lockGlobalGroup(group, { config: !on }).catch(() => {});
	}

	function setControl(gv: GlobalView, patch: Partial<ControlView>): void {
		void g.editControl(group, gv.element, patch).catch(() => {});
	}

	function commitValue(gv: GlobalView, v: Value): void {
		void g.setGlobalValue(gv.name, v).catch(() => {});
	}

	function num(v: Value): number {
		return typeof v === 'number' ? v : 0;
	}

	function down(e: PointerEvent, gv: GlobalView, resize: boolean): void {
		if (!edit || !board) return;
		picked = gv.name;
		drag = { name: gv.name, from: cellOf(gv), x: e.clientX, y: e.clientY, units: unitsOf(board), resize, to: null };
		const el = e.currentTarget as HTMLElement;
		el.setPointerCapture(e.pointerId);
		(el.closest('.cell') as HTMLElement | null)?.focus();
		e.preventDefault();
		e.stopPropagation();
	}

	function move(e: PointerEvent): void {
		if (!drag) return;
		const [dx, dy] = [e.clientX - drag.x, e.clientY - drag.y];
		drag.to = drag.resize
			? resizedBy(drag.from, dx, dy, drag.units, COLUMNS)
			: movedBy(drag.from, dx, dy, drag.units, COLUMNS);
	}

	// ONE op per gesture, so a drag is one undo step, the way every other frozen drag is.
	function up(): void {
		if (!drag) return;
		const { name, from, to } = drag;
		drag = null;
		const gv = elements.find((el) => el.name === name);
		if (!gv || !to || sameCell(to, from)) return;
		pending = { name, to };
		void g.editControl(group, gv.element, to).catch(() => (pending = null));
	}

	function zap(e: KeyboardEvent, gv: GlobalView): void {
		if (!edit || isTextEditingTarget(e.target) || (e.key !== 'Delete' && e.key !== 'Backspace')) return;
		e.preventDefault();
		void g.removeControl(group, gv.element);
	}

	function liftChip(e: PointerEvent, kind: Kind): void {
		if (!board) return;
		const u = unitsOf(board);
		const born = BORN[kind];
		lift = {
			kind,
			x: e.clientX,
			y: e.clientY,
			w: born.w * u.x - u.gap,
			h: born.h * u.y - u.gap,
			from: { x: e.clientX, y: e.clientY }
		};
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		e.preventDefault();
	}

	function driftChip(e: PointerEvent): void {
		if (!lift) return;
		lift.x = e.clientX;
		lift.y = e.clientY;
	}

	// A tap bears the widget where the manager places it; a drag bears it where it was let go.
	function dropChip(e: PointerEvent): void {
		if (!lift) return;
		const { kind, from } = lift;
		lift = null;
		const dragged = Math.hypot(e.clientX - from.x, e.clientY - from.y) > 4;
		const point = dragged ? onBoard(e) : null;
		if (dragged && !point) return;
		void bear(kind, point);
	}

	async function bear(kind: Kind, point: { x: number; y: number } | null): Promise<void> {
		const born = BORN[kind];
		const cell = point && board ? cellAt(point.x, point.y, born.w, born.h, unitsOf(board), COLUMNS) : undefined;
		try {
			picked = await g.addControl(group, kind, cell);
		} catch {
			/* refused */
		}
	}

	function zero(type: GlobalView['type']): Value {
		return type === 'bool' ? false : type === 'string' ? '' : 0;
	}

	async function rename(gv: GlobalView, raw: string): Promise<void> {
		renaming = null;
		const element = raw.trim();
		if (element === gv.element || !isValidIdentifier(element)) return;
		try {
			await g.editControl(group, gv.element, { name: element });
			if (picked === gv.name) picked = `${group}.${element}`;
		} catch {
			/* refused */
		}
	}

	async function startRename(gv: GlobalView, cell: HTMLElement): Promise<void> {
		renaming = gv.name;
		await tick();
		const input = cell.querySelector<HTMLInputElement>('[data-testid="control-rename"]');
		input?.focus();
		input?.select();
	}

	function optionsOf(raw: string): string[] {
		return raw
			.split(',')
			.map((s) => s.trim())
			.filter((s) => s !== '');
	}

	/** What the palette ghost of `kind` shows: a value in the middle of the range it is born with. */
	function sample(kind: Kind): Value {
		return TYPE_OF[kind] === 'float' ? 0.5 : zero(TYPE_OF[kind]);
	}

	/** The widget's value as the inspector field reads it: a param whose one other source is a
	 * reference, since a global follows a producer and never an expression. */
	function valueDescriptor(pv: GlobalView, pc: ControlView): ParamDescriptor {
		const base = {
			doc: null,
			refreshable: false,
			mode: pv.source ? ('reference' as const) : ('constant' as const),
			expression: null,
			reference: pv.source?.reference ?? null,
			triggers: false,
			error: null
		};
		if (pv.type === 'float' || pv.type === 'int') {
			return { ...base, type: pv.type, value: num(pv.value), vmin: pc.min ?? 0, vmax: pc.max ?? 1 };
		}
		if (pv.type === 'bool') return { ...base, type: 'bool', value: pv.value === true };
		return { ...base, type: 'string', value: String(pv.value), options: pc.kind === 'dropdown' ? (pc.options ?? []) : null };
	}

	function setSource(pv: GlobalView, patch: SourcePatch): void {
		stopLearning();
		if (patch.reference !== undefined) void g.sourceControl(group, pv.element, patch.reference).catch(() => {});
		else if (patch.mode === 'constant') void g.sourceControl(group, pv.element, '').catch(() => {});
	}

	function setIndex(pv: GlobalView, index: number): void {
		const ref = pv.source?.reference;
		if (!ref) return;
		void g.sourceControl(group, pv.element, ref, Math.max(0, Math.round(index))).catch(() => {});
	}

	// MIDI learn: listen to the followed slot, and the first number that moves is the one.
	let learning = $state<(() => void) | null>(null);
	function stopLearning(): void {
		learning?.();
		learning = null;
	}
	function toggleLearn(pv: GlobalView): void {
		if (learning) return stopLearning();
		const ref = pv.source?.reference;
		const [nodeName, slot] = ref?.split('.') ?? [];
		const uid = g.nodes.find((n) => n.name === nodeName)?.uid;
		if (!ref || !slot || !uid) return;
		const element = pv.element;
		let baseline: number[] | null = null;
		learning = bindViewer(uid, slot, `learn:${group}.${element}`, viewSpecForKind('line', 4096, 64), (f: DataFrame) => {
			const values = (f.data as ArrayData).values;
			if (!values || typeof values.length !== 'number') return;
			if (!baseline) {
				baseline = Array.from(values);
				return;
			}
			for (let i = 0; i < values.length; i++) {
				if (Math.abs(values[i] - (baseline[i] ?? values[i])) > 1e-6) {
					stopLearning();
					void g.sourceControl(group, element, ref, i).catch(() => {});
					return;
				}
			}
		});
	}
	$effect(() => {
		if (!edit || !pickedView) stopLearning();
	});
	onDestroy(stopLearning);
</script>

{#snippet widget(c: ControlView, value: Value, label: string, onChange: (v: Value) => void)}
	{#if c.kind === 'knob'}
		<Knob {label} value={num(value)} min={c.min ?? 0} max={c.max ?? 1} step={c.step ?? 0} {onChange} />
	{:else if c.kind === 'slider'}
		<Slider value={num(value)} min={c.min ?? 0} max={c.max ?? 1} step={c.step} {onChange} />
	{:else if c.kind === 'number'}
		<NumberInput value={num(value)} min={c.min} max={c.max} step={c.step ?? 1} scrub {onChange} />
	{:else if c.kind === 'toggle'}
		<Toggle value={value === true} {onChange} />
	{:else if c.kind === 'dropdown'}
		<Select value={String(value)} options={c.options ?? []} {onChange} />
	{:else}
		<TextInput inputmode="search" value={String(value)} autocomplete="off" {onChange} />
	{/if}
{/snippet}

<div class="wrap" data-testid="control-panel" data-group={group} data-edit={edit}>
	<div class="bar">
		<span class="title">{group}</span>
		<IconButton
			variant={edit ? 'primary' : 'ghost'}
			size="sm"
			data-testid="control-edit-toggle"
			title={edit ? 'Done editing' : 'Edit this panel'}
			label={edit ? 'Done editing' : 'Edit this panel'}
			disabled={!named}
			onclick={() => setEdit(!edit)}><Icon name={edit ? 'check' : 'pencil'} /></IconButton
		>
	</div>

	<ScrollArea>
		<div class="sheet">
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="board"
				data-testid="control-board"
				bind:this={board}
				style={`--columns: ${COLUMNS}`}
				onpointermove={move}
				onpointerup={up}
				onpointercancel={up}
			>
				{#if elements.length === 0}
					<div class="fill">
						<EmptyState data-testid="control-empty">
							{#snippet title()}No widgets yet{/snippet}
							{#snippet hint()}{edit ? 'Drag one in from the palette, or tap it.' : 'Switch on edit mode to add some.'}{/snippet}
						</EmptyState>
					</div>
				{/if}
				{#each elements as gv (gv.name)}
					{@const c = gv.control as ControlView}
					{@const at = placed(gv)}
					{@const held = effectiveLock(gv, groupLock)}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
					<div
						class="cell"
						class:picked={edit && picked === gv.name}
						data-testid={`control-${group}-${gv.element}`}
						style={`grid-column: ${at.x + 1} / span ${at.w}; grid-row: ${at.y + 1} / span ${at.h}`}
						tabindex={edit ? 0 : undefined}
						onpointerdown={(e) => down(e, gv, false)}
						onkeydown={(e) => zap(e, gv)}
					>
						<div
							class="widget"
							class:held={held.value || gv.source !== undefined}
							title={gv.source ? `Follows ${gv.source.reference}` : held.value ? 'Value-locked' : undefined}
						>
							{@render widget(c, gv.value, gv.element, (v) => commitValue(gv, v))}
						</div>
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<span
							class="label"
							title="Double-click to rename"
							ondblclick={(e) => startRename(gv, (e.currentTarget as HTMLElement).parentElement as HTMLElement)}
							>{gv.element}</span
						>
						{#if renaming === gv.name}
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<div class="rename" onpointerdown={(e) => e.stopPropagation()}>
								<TextInput
									inputmode="search"
									data-testid="control-rename"
									value={gv.element}
									autocomplete="off"
									onChange={(v) => {
										if (renaming === gv.name) void rename(gv, v);
									}}
									onkeydown={(e) => {
										if (e.key === 'Escape') renaming = null;
									}}
								/>
							</div>
						{/if}
						{#if edit}
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<span
								class="handle"
								data-testid="control-resize"
								title="Resize"
								onpointerdown={(e) => down(e, gv, true)}
							></span>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	</ScrollArea>

	<SidePane open={edit} testid="control-inspector" storage="goofi.controlPane">
		<ScrollArea>
			<div class="pane">
				<Field label="panel" doc="The group its globals live in: globals.panel.element">
					<TextInput
						inputmode="search"
						data-testid="control-group-name"
						value={group}
						autocomplete="off"
						onChange={nameGroup}
					/>
				</Field>
				<div class="palette" data-testid="control-palette">
					{#each KINDS as kind (kind)}
						<Chip
							tone={lift?.kind === kind ? 'accent' : 'neutral'}
							data-testid={`control-palette-${kind}`}
							title="Drag onto the board, or tap to add"
							onpointerdown={(e) => liftChip(e, kind)}
							onpointermove={driftChip}
							onpointerup={dropChip}
							onpointercancel={() => (lift = null)}
							onclick={(e) => {
								if (e.detail === 0) void bear(kind, null);
							}}>{kind}</Chip
						>
					{/each}
				</div>
				{#if pickedView && !pickedView.lock.config}
					{@const pv = pickedView}
					{@const pc = pv.control as ControlView}
					<div class="props" data-testid="control-props">
						<Field label="name">
							<TextInput
								inputmode="search"
								data-testid="control-props-name"
								value={pv.element}
								autocomplete="off"
								onChange={(v) => void rename(pv, v)}
							/>
						</Field>
						<Field label="widget">
							<Select
								data-testid="control-props-kind"
								value={pc.kind}
								options={KINDS.filter((k) => TYPE_OF[k] === pv.type)}
								onChange={(v) => setControl(pv, { kind: v as Kind })}
							/>
						</Field>
						<ParamField
							paramName="value"
							descriptor={valueDescriptor(pv, pc)}
							modes={['constant', 'reference']}
							onCommit={(v) => commitValue(pv, v as Value)}
							onSetSource={(patch) => setSource(pv, patch)}
							data-testid="control-props-value"
						/>
						{#if pv.source}
							<div class="row">
								<Field label="index" doc="Which number of a wide frame the widget reads — a controller's cc holds 128">
									<NumberInput value={pv.source.index ?? 0} min={0} step={1} onChange={(v) => setIndex(pv, v)} />
								</Field>
								<Chip
									tone={learning ? 'accent' : 'neutral'}
									aria-pressed={learning !== null}
									data-testid="control-learn"
									title="Move one control on the source, and the widget follows that one"
									onclick={() => toggleLearn(pv)}>{learning ? 'listening…' : 'learn'}</Chip
								>
							</div>
						{/if}
						{#if pv.type === 'float' || pv.type === 'int'}
							<div class="row">
								<Field label="min">
									<NumberInput value={pc.min ?? 0} onChange={(v) => setControl(pv, { min: v })} />
								</Field>
								<Field label="max">
									<NumberInput value={pc.max ?? 1} onChange={(v) => setControl(pv, { max: v })} />
								</Field>
								<Field label="step">
									<NumberInput value={pc.step ?? 0} min={0} onChange={(v) => setControl(pv, { step: v })} />
								</Field>
							</div>
						{/if}
						{#if pc.kind === 'dropdown'}
							<Field label="options" doc="Comma-separated">
								<TextInput
									inputmode="text"
									data-testid="control-props-options"
									value={(pc.options ?? []).join(', ')}
									onChange={(v) => setControl(pv, { options: optionsOf(v) })}
								/>
							</Field>
						{/if}
						<div class="row end">
							<IconButton
								variant="ghost"
								size="sm"
								data-testid="control-delete"
								title="Delete element (Delete)"
								label="Delete {pv.element}"
								onclick={() => void g.removeControl(group, pv.element)}><Icon name="x" /></IconButton
							>
						</div>
					</div>
				{:else}
					<EmptyState>
						{#snippet hint()}{pickedView ? 'This widget is config-locked.' : 'Tap a widget to edit it.'}{/snippet}
					</EmptyState>
				{/if}
			</div>
		</ScrollArea>
	</SidePane>

	{#if lift}
		<div class="ghost" style={`left: ${lift.x}px; top: ${lift.y}px`} aria-hidden="true">
			<div class="cell born" style={`width: ${lift.w}px; height: ${lift.h}px`}>
				<div class="widget">
					{@render widget(
						{ kind: lift.kind, min: 0, max: 1, step: 0.01, x: 0, y: 0, w: 0, h: 0 },
						sample(lift.kind),
						lift.kind,
						() => {}
					)}
				</div>
				<span class="label">{lift.kind}</span>
			</div>
		</div>
	{/if}
</div>

<style>
	.wrap {
		position: relative;
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
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.pane {
		display: flex;
		flex-direction: column;
		gap: var(--space-5);
		padding: var(--space-4) var(--space-5) var(--space-6);
	}
	.palette {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
		touch-action: none;
	}
	.props {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}
	.row {
		display: flex;
		align-items: flex-end;
		gap: var(--space-4);
	}
	.row.end {
		justify-content: flex-end;
	}
	/* The board is a container, so a grid unit is a share of ITS width and follows every resize. */
	.sheet {
		container-type: inline-size;
		min-height: 100%;
	}
	.board {
		--gap: var(--space-1);
		--pad: var(--space-3);
		--col: calc((100cqw - 2 * var(--pad) - (var(--columns) - 1) * var(--gap)) / var(--columns));
		--label-h: calc(var(--fs-small) + var(--space-3));
		display: grid;
		grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
		/* A row is a square of the column, floored so a two-row widget still fits a tap target. */
		grid-auto-rows: max(var(--col), calc((var(--hit) + var(--label-h) + 2 * var(--space-2)) / 2));
		gap: var(--gap);
		padding: var(--pad);
		min-height: 100%;
	}
	[data-edit='true'] .board {
		touch-action: none;
	}
	.fill {
		grid-column: 1 / -1;
	}
	.cell {
		position: relative;
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		padding: var(--space-2);
		border-radius: var(--radius-md);
		background: var(--surface-1);
		overflow: hidden;
	}
	[data-edit='true'] .cell {
		outline: 1px dashed var(--border-strong);
		cursor: grab;
	}
	.cell.picked {
		outline: var(--focus-width) solid var(--accent);
	}
	.cell:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
	}
	.widget {
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		--number-width: 100%;
	}
	[data-edit='true'] .widget,
	.born .widget {
		pointer-events: none;
	}
	.widget.held {
		pointer-events: none;
		opacity: var(--disabled-opacity);
	}
	.label {
		height: var(--label-h);
		line-height: var(--label-h);
		text-align: center;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		visibility: hidden;
	}
	.cell:hover .label,
	.cell:focus-within .label,
	[data-edit='true'] .label,
	.born .label {
		visibility: visible;
	}
	@media (hover: none) and (pointer: coarse) {
		.label {
			visibility: visible;
		}
	}
	.rename {
		position: absolute;
		left: var(--space-2);
		right: var(--space-2);
		bottom: var(--space-2);
		z-index: 1;
	}
	/* A --hit corner to grab, painted as a small tab so it does not hide the widget. */
	.handle {
		position: absolute;
		right: 0;
		bottom: 0;
		width: var(--hit);
		height: var(--hit);
		cursor: nwse-resize;
		touch-action: none;
	}
	.handle::after {
		content: '';
		position: absolute;
		right: 0;
		bottom: 0;
		width: var(--space-6);
		height: var(--space-6);
		background: var(--accent);
		border-radius: var(--radius-sm) 0 var(--radius-md) 0;
	}
	/* The ghost IS the widget it will bear, at the size it will be born, carried by its centre. */
	.ghost {
		position: fixed;
		z-index: var(--z-drag-ghost);
		transform: translate(-50%, -50%);
		pointer-events: none;
	}
	.born {
		outline: 1px dashed var(--accent);
		opacity: 0.85;
	}
</style>
