<!-- Control panel — knobs, sliders and fields over ONE group of globals. Edit mode is the group's
     config lock, inverted: out of it a drag turns a widget; in it the same drag moves it, the
     corner resizes it, a strip above the board holds the name and the palette, and the picked
     widget's form opens beside the widget itself. Every change is a globals op, so the manager
     owns the state and this panel owns only the drawing and the gesture in flight. -->
<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import type { PanelProps } from 'panelty';
	import { asStateObject } from 'panelty';
	import { graph } from '$lib/stores/graph.svelte';
	import type { ControlView, GlobalView, LockView } from '$lib/crdt/graphDoc';
	import { effectiveLock, isValidIdentifier } from '$lib/crdt/graphDoc';
	import { ui } from '$lib/stores/ui.svelte';
	import { bindViewer } from '$lib/api/frames';
	import type { ArrayData, DataFrame } from '$lib/codec/decode';
	import { viewSpecForKind } from '$lib/viewers/capacity';
	import RefPicker from '$lib/inspector/RefPicker.svelte';
	import {
		Chip,
		EmptyState,
		Field,
		Icon,
		IconButton,
		Knob,
		NumberInput,
		Popover,
		ScrollArea,
		Segmented,
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
	const uiStore = ui();

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
	// The form hangs off the picked widget's own cell, and follows it wherever a drag lands it.
	const anchor = $derived.by(() => {
		if (!board || !pickedView) return null;
		const at = placed(pickedView);
		void [at.x, at.y, at.w, at.h];
		return board.querySelector<HTMLElement>(`[data-testid="control-${group}-${pickedView.element}"]`);
	});

	// The two gestures in flight: a widget being moved or resized, and a chip lifted off the palette.
	let drag: { name: string; from: Cell; x: number; y: number; units: Units; resize: boolean; to: Cell | null } | null =
		$state(null);
	// `at` is the ghost's top-left: snapped to the cell it would land on while over the board, else
	// under the pointer's centre.
	let lift: { kind: Kind; at: { x: number; y: number }; snapped: boolean; w: number; h: number; from: { x: number; y: number } } | null =
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

	/** Where the pointer is on the board, in pixels into its grid, or null when it is off the
	 * board's scroll area — the whole area, since the board itself is only as tall as its widgets. */
	function onBoard(e: PointerEvent): { x: number; y: number } | null {
		const zone = board?.closest('.ui-scrollarea') ?? board;
		if (!board || !zone) return null;
		const z = zone.getBoundingClientRect();
		if (e.clientX < z.left || e.clientX > z.right || e.clientY < z.top || e.clientY > z.bottom) return null;
		const r = board.getBoundingClientRect();
		const cs = getComputedStyle(board);
		return { x: e.clientX - r.left - parseFloat(cs.paddingLeft), y: e.clientY - r.top - parseFloat(cs.paddingTop) };
	}

	/** The ghost's top-left for a pointer at `e`: the cell it would land on, in pixels, while over
	 * the board; else centred under the pointer. */
	function ghostAt(e: PointerEvent, kind: Kind, w: number, h: number): { at: { x: number; y: number }; snapped: boolean } {
		const point = onBoard(e);
		if (point && board) {
			const u = unitsOf(board);
			const born = BORN[kind];
			const cell = cellAt(point.x, point.y, born.w, born.h, u, COLUMNS);
			const r = board.getBoundingClientRect();
			const cs = getComputedStyle(board);
			return {
				at: { x: r.left + parseFloat(cs.paddingLeft) + cell.x * u.x, y: r.top + parseFloat(cs.paddingTop) + cell.y * u.y },
				snapped: true
			};
		}
		return { at: { x: e.clientX - w / 2, y: e.clientY - h / 2 }, snapped: false };
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
		const [w, h] = [born.w * u.x - u.gap, born.h * u.y - u.gap];
		lift = { kind, ...ghostAt(e, kind, w, h), w, h, from: { x: e.clientX, y: e.clientY } };
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		e.preventDefault();
	}

	function driftChip(e: PointerEvent): void {
		if (!lift) return;
		const { at, snapped } = ghostAt(e, lift.kind, lift.w, lift.h);
		lift.at = at;
		lift.snapped = snapped;
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

	// A widget is either set by hand or LINKED to one output of a node. `linking` is the link
	// segment lit before a node is chosen, so the picker shows with nothing to show yet.
	let linking = $state(false);
	$effect(() => {
		if (pickedView?.source) linking = false;
	});
	$effect(() => {
		void picked;
		linking = false;
	});

	function setSource(pv: GlobalView, reference: string): void {
		stopLearning();
		linking = false;
		void g.sourceControl(group, pv.element, reference).catch(() => {});
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

	{#if edit}
		<div class="strip">
			<div class="grow">
				<TextInput
					inputmode="search"
					data-testid="control-group-name"
					title="The group its globals live in: globals.name.element"
					value={group}
					autocomplete="off"
					onChange={nameGroup}
				/>
			</div>
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
						data-node-drop={edit ? gv.name : undefined}
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
							<button
								type="button"
								class="zap"
								data-testid="control-delete"
								title="Delete {gv.element}"
								aria-label="Delete {gv.element}"
								onpointerdown={(e) => e.stopPropagation()}
								onclick={() => void g.removeControl(group, gv.element)}><Icon name="x" /></button
							>
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<span
								class="handle"
								data-testid="control-resize"
								title="Resize"
								onpointerdown={(e) => down(e, gv, true)}
							></span>
						{/if}
						{#if edit && uiStore.nodeDrag !== null}
							<div class="node-drop-hint" class:active={uiStore.nodeDragWidget === gv.name} data-testid="node-drop-hint"></div>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	</ScrollArea>

	{#if edit && pickedView}
		{@const pv = pickedView}
		{@const pc = pv.control as ControlView}
		{#key `${pv.name}:${placed(pv).x},${placed(pv).y}`}
			<Popover
				{anchor}
				open={anchor !== null}
				onDismiss={() => (picked = null)}
				flip
				role="dialog"
				aria-label={`${pv.element} settings`}
				style="--popover-min-width: 18rem"
				data-testid="control-props"
			>
				{#if pv.lock.config}
					<EmptyState>
						{#snippet hint()}This widget is config-locked.{/snippet}
					</EmptyState>
				{:else}
					<div class="props">
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
						<Field label="source">
							<Segmented
								value={pv.source || linking ? 'link' : 'value'}
								segments={[
									{ id: 'value', label: 'value', title: 'Set by hand, on the widget itself', testid: 'control-source-value' },
									{
										id: 'link',
										label: 'link',
										title: 'Follow one output of a node — a MIDI controller, say. Drop the node onto the widget, or pick it',
										testid: 'control-source-link'
									}
								]}
								onChange={(id) => (id === 'link' ? (linking = true) : pv.source ? setSource(pv, '') : (linking = false))}
								aria-label="source"
								data-testid="control-source"
							/>
						</Field>
						{#if pv.source || linking}
							<Field label="link" doc="The node output the widget follows; a node dropped onto the widget lands here">
								<RefPicker
									value={pv.source?.reference ?? null}
									paramType={pv.type}
									onCommit={(r) => setSource(pv, r)}
									testid="control-props-link"
								/>
							</Field>
						{/if}
						{#if pv.source}
							<Field label="index" doc="Which number of a wide frame the widget reads — a controller's cc holds 128">
								<NumberInput value={pv.source.index ?? 0} min={0} step={1} onChange={(v) => setIndex(pv, v)} />
								<Chip
									tone={learning ? 'accent' : 'neutral'}
									aria-pressed={learning !== null}
									data-testid="control-learn"
									title="Move one control on the source, and the widget follows that one"
									onclick={() => toggleLearn(pv)}>{learning ? 'listening…' : 'learn'}</Chip
								>
							</Field>
						{/if}
						{#if pv.type === 'float' || pv.type === 'int'}
							<Field label="range" doc="min, max and step">
								<NumberInput value={pc.min ?? 0} title="min" onChange={(v) => setControl(pv, { min: v })} />
								<NumberInput value={pc.max ?? 1} title="max" onChange={(v) => setControl(pv, { max: v })} />
								<NumberInput value={pc.step ?? 0} min={0} title="step" onChange={(v) => setControl(pv, { step: v })} />
							</Field>
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
					</div>
				{/if}
			</Popover>
		{/key}
	{/if}

	{#if lift}
		<div class="ghost" class:snapped={lift.snapped} style={`left: ${lift.at.x}px; top: ${lift.at.y}px`} aria-hidden="true">
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
	.strip {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2) var(--space-4);
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px dashed var(--border);
	}
	.grow {
		flex: 1 1 10rem;
		min-width: 0;
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
		/* Three numbers share one field's row, so each takes its share rather than a fixed width. */
		--number-width: 100%;
	}
	/* A bare red cross over the widget's corner, no box of its own; its press must not start a drag. */
	.zap {
		position: absolute;
		top: 0;
		right: 0;
		z-index: 1;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: var(--hit);
		min-height: var(--hit);
		padding: 0;
		background: transparent;
		border: none;
		color: var(--danger);
		cursor: pointer;
	}
	.zap:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: -2px;
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
	/* The ghost IS the widget it will bear, at the size it will be born, on the cell it will land. */
	.ghost {
		position: fixed;
		z-index: var(--z-drag-ghost);
		pointer-events: none;
	}
	.ghost.snapped .born {
		outline-style: solid;
		opacity: 1;
	}
	.born {
		outline: 1px dashed var(--accent);
		opacity: 0.85;
	}
</style>
