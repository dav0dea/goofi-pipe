<!-- Control panel — knobs, sliders and fields over ONE group of globals. Edit mode is the group's
     config lock, inverted: out of it a drag turns a widget; in it the same drag moves it, the
     corner resizes it, and a chip dragged off the palette bears a new one. Every change is a
     globals op, so the manager owns the state and this panel owns only the drawing and the gesture
     in flight. -->
<script lang="ts">
	import { tick } from 'svelte';
	import type { PanelProps } from 'panelty';
	import { asStateObject } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import type { ControlView, GlobalView, LockView } from '$lib/crdt/graphDoc';
	import { effectiveLock, isValidIdentifier } from '$lib/crdt/graphDoc';
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
		freeCell,
		freshName,
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
	const members = $derived(g.globals.filter((gv) => gv.group === group));
	const elements = $derived(members.filter((gv) => gv.control));

	let board: HTMLDivElement | null = $state(null);
	let picked = $state<string | null>(null);
	let renaming = $state<string | null>(null);
	const pickedView = $derived(elements.find((el) => el.name === picked) ?? null);

	// The two gestures in flight: a widget being moved or resized, and a chip lifted off the palette.
	let drag: { name: string; from: Cell; x: number; y: number; units: Units; resize: boolean; to: Cell | null } | null =
		$state(null);
	let lift: { kind: Kind; name: string; x: number; y: number; w: number; h: number; from: { x: number; y: number } } | null =
		$state(null);
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
		if (!named) props.setState({ ...st, group: to }, 'authored', `Name control panel ${to}`);
		else void g.renameGlobalGroup(group, to).catch(() => {});
	}

	function setEdit(on: boolean): void {
		picked = null;
		renaming = null;
		void g.lockGlobalGroup(group, { config: !on }).catch(() => {});
	}

	function setControl(gv: GlobalView, patch: Partial<ControlView>): void {
		void g.setGlobalControl(gv.name, { ...(gv.control as ControlView), ...patch }).catch(() => {});
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
		void g.setGlobalControl(gv.name, { ...(gv.control as ControlView), ...to }).catch(() => (pending = null));
	}

	function zap(e: KeyboardEvent, gv: GlobalView): void {
		if (!edit || isTextEditingTarget(e.target) || (e.key !== 'Delete' && e.key !== 'Backspace')) return;
		e.preventDefault();
		void g.removeGlobal(gv.name);
	}

	function liftChip(e: PointerEvent, kind: Kind): void {
		if (!board) return;
		const u = unitsOf(board);
		const born = BORN[kind];
		lift = {
			kind,
			name: freshName(kind, members.map((gv) => gv.element)),
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

	// A tap bears the widget in the first free cell; a drag bears it where it was let go.
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
		const cell =
			point && board
				? cellAt(point.x, point.y, born.w, born.h, unitsOf(board), COLUMNS)
				: freeCell(elements.map(cellOf), born.w, born.h, COLUMNS);
		const type = TYPE_OF[kind];
		const control: ControlView = { kind, ...cell };
		if (type === 'float') Object.assign(control, { min: 0, max: 1, step: 0.01 });
		const name = `${group}.${freshName(kind, members.map((gv) => gv.element))}`;
		try {
			await g.addGlobal(name, zero(type), type, control);
			picked = name;
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
		const to = `${group}.${element}`;
		try {
			await g.renameGlobal(gv.name, to);
			if (picked === gv.name) picked = to;
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
		{#if edit || !named}
			<div class="grow">
				<TextInput
					inputmode="search"
					data-testid="control-group-name"
					placeholder="name this panel"
					value={group}
					autocomplete="off"
					onChange={nameGroup}
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
			disabled={!named}
			onclick={() => setEdit(!edit)}><Icon name={edit ? 'check' : 'pencil'} /></IconButton
		>
	</div>

	{#if edit}
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
	{/if}

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
				{#if !named}
					<div class="fill">
						<EmptyState data-testid="control-unnamed">
							{#snippet title()}Name this panel{/snippet}
							{#snippet hint()}Its name is the group its globals live in: <code>globals.name.element</code>.{/snippet}
						</EmptyState>
					</div>
				{:else if elements.length === 0}
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
						<div class="widget" class:held={held.value} title={held.value ? 'Value-locked' : undefined}>
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

	{#if edit && pickedView && !pickedView.lock.config}
		{@const pv = pickedView}
		{@const pc = pv.control as ControlView}
		<div class="props" data-testid="control-props">
			<div class="prop">
				<Field label="name">
					<TextInput
						inputmode="search"
						data-testid="control-props-name"
						value={pv.element}
						autocomplete="off"
						onChange={(v) => void rename(pv, v)}
					/>
				</Field>
			</div>
			<div class="prop narrow">
				<Field label="widget">
					<Select
						data-testid="control-props-kind"
						value={pc.kind}
						options={KINDS.filter((k) => TYPE_OF[k] === pv.type)}
						onChange={(v) => setControl(pv, { kind: v as Kind })}
					/>
				</Field>
			</div>
			{#if pv.type === 'float' || pv.type === 'int'}
				<div class="prop narrow">
					<Field label="min">
						<NumberInput value={pc.min ?? 0} onChange={(v) => setControl(pv, { min: v })} />
					</Field>
				</div>
				<div class="prop narrow">
					<Field label="max">
						<NumberInput value={pc.max ?? 1} onChange={(v) => setControl(pv, { max: v })} />
					</Field>
				</div>
				<div class="prop narrow">
					<Field label="step">
						<NumberInput value={pc.step ?? 0} min={0} onChange={(v) => setControl(pv, { step: v })} />
					</Field>
				</div>
			{/if}
			{#if pc.kind === 'dropdown'}
				<div class="prop">
					<Field label="options" doc="Comma-separated">
						<TextInput
							inputmode="text"
							data-testid="control-props-options"
							value={(pc.options ?? []).join(', ')}
							onChange={(v) => setControl(pv, { options: optionsOf(v) })}
						/>
					</Field>
				</div>
			{/if}
			<IconButton
				variant="ghost"
				size="sm"
				data-testid="control-delete"
				title="Delete element (Delete)"
				label="Delete {pv.element}"
				onclick={() => void g.removeGlobal(pv.name)}><Icon name="x" /></IconButton
			>
		</div>
	{/if}

	{#if lift}
		<div class="ghost" style={`left: ${lift.x}px; top: ${lift.y}px`} aria-hidden="true">
			<div class="cell born" style={`width: ${lift.w}px; height: ${lift.h}px`}>
				<div class="widget">
					{@render widget(
						{ kind: lift.kind, min: 0, max: 1, step: 0.01, x: 0, y: 0, w: 0, h: 0 },
						sample(lift.kind),
						lift.name,
						() => {}
					)}
				</div>
				<span class="label">{lift.name}</span>
			</div>
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
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.grow {
		flex: 1;
		min-width: 0;
	}
	.palette {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px dashed var(--border);
		touch-action: none;
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
	.props {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: var(--space-3) var(--space-4);
		padding: var(--space-3);
		border-top: 1px solid var(--border);
	}
	.prop {
		flex: 1 1 8rem;
		min-width: 0;
	}
	.prop.narrow {
		flex: 0 1 5rem;
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
