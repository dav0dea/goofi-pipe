<!--
  Node-editor panel — the SvelteFlow graph plus all its interaction logic,
  extracted from the former Editor.svelte monolith into a self-contained panel.

  Each instance owns its own SvelteFlowProvider (independent viewport) but reads
  the shared graph + selection stores, so multiple editor panels stay in sync
  on content and selection while panning/zooming independently.

  Editor-scoped keyboard shortcuts (Delete, Ctrl+C/V/D, Ctrl+A, Tab, F, Escape)
  are gated on this being the active panel; app-global shortcuts (Ctrl+S/O) and
  the unsaved-changes guard live in AppShell.
-->
<script lang="ts">
	import {
		SvelteFlow,
		Controls,
		MiniMap,
		SvelteFlowProvider,
		ViewportPortal,
		type Connection,
		type Edge,
		type Node
	} from '@xyflow/svelte';
	import GoofiNode from '$lib/editor/GoofiNode.svelte';
	import AddNodeMenu from '$lib/editor/AddNodeMenu.svelte';
	import PlacementPreview from '$lib/editor/PlacementPreview.svelte';
	import {
		computeSnapDelta,
		makeBounds,
		DEFAULT_NODE_W,
		DEFAULT_NODE_H,
		type Bounds,
		type Guide
	} from '$lib/editor/snap';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui, type SlotClickSeed } from '$lib/stores/ui.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import type { PanelProps } from '$lib/workspace/registry';
	import type { LinkInfo, NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';
	import { registerEditor, unregisterEditor } from './editorCommands';
	import InspectorOverlay from './InspectorOverlay.svelte';
	import { onMount } from 'svelte';

	let { panelId }: PanelProps = $props();

	const g = graph();
	const uiStore = ui();
	const sel = selection();
	const ws = workspace();

	/** Only the active panel reacts to editor keyboard shortcuts and pending
	 * slot clicks, so multiple editor panels don't all fire at once. */
	const isActive = (): boolean => ws.activePanelId === panelId;

	// This editor's own selection (independent per panel). The inspector
	// overlay reads it; standalone panels follow whichever editor is active.
	const selectedNode = $derived(sel.selectedNode(panelId));

	$effect(() => {
		// When this editor becomes the active panel, mark it the active editor
		// so the standalone Parameters/Metadata/Errors panels follow it.
		if (ws.activePanelId === panelId) sel.setActiveEditor(panelId);
	});

	let rootEl = $state<HTMLDivElement | null>(null);

	let menuOpen = $state(false);
	let menuPos = $state<{ x: number; y: number }>({ x: 120, y: 120 });
	let menuSeed = $state<SlotClickSeed | null>(null);

	// Watch ui.pendingSlotClick — when a node's port is clicked, open the
	// menu near the port and seed it with the dtype filter + auto-link. Only
	// the active panel consumes it (the click sets this panel active first).
	$effect(() => {
		const seed = uiStore.pendingSlotClick;
		if (!seed) return;
		if (!isActive()) return;
		uiStore.consumeSlotClick();
		menuSeed = seed;
		const offsetX = seed.side === 'source' ? 12 : -332;
		menuPos = {
			x: Math.max(8, Math.min(window.innerWidth - 332, seed.clientX + offsetX)),
			y: Math.max(8, Math.min(window.innerHeight - 360, seed.clientY - 24))
		};
		menuOpen = true;
	});

	let snapGuides = $state<Guide[]>([]);
	let pendingPlacement = $state<{
		typeInfo: NodeTypeInfo;
		seed: SlotClickSeed | null;
		initialClient: { x: number; y: number };
	} | null>(null);

	let flowNodes = $state.raw<Node[]>([]);
	let flowEdges = $state.raw<Edge[]>([]);

	$effect(() => {
		const next: Node[] = g.nodes.map((n) => ({
			id: n.name,
			type: 'goofi',
			position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
			data: { node: n },
			selected: sel.nodes(panelId).has(n.name)
		}));
		flowNodes = next;
	});

	$effect(() => {
		const next: Edge[] = g.links.map((l) => {
			const id = edgeId(l);
			return {
				id,
				source: l.node_out,
				sourceHandle: l.slot_out,
				target: l.node_in,
				targetHandle: l.slot_in,
				selected: sel.edges(panelId).has(id),
				animated: false
			};
		});
		flowEdges = next;
	});

	function edgeId(l: LinkInfo): string {
		return `${l.node_out}.${l.slot_out}→${l.node_in}.${l.slot_in}`;
	}

	function onConnect(c: Connection): void {
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
		void g.addLink({
			node_out: c.source,
			node_in: c.target,
			slot_out: c.sourceHandle,
			slot_in: c.targetHandle
		});
	}

	function onEdgeClick(args: { edge: Edge; event: MouseEvent }): void {
		const e = args.event;
		sel.clickEdge(panelId, args.edge.id, e.shiftKey || e.ctrlKey || e.metaKey);
	}

	function findLinkById(id: string): LinkInfo | null {
		for (const l of g.links) if (edgeId(l) === id) return l;
		return null;
	}

	async function deleteEdgeSelection(): Promise<void> {
		const ids = Array.from(sel.edges(panelId));
		for (const id of ids) {
			const link = findLinkById(id);
			if (!link) continue;
			try {
				await g.removeLink(link);
			} catch (err) {
				console.warn('remove edge failed', err);
			}
		}
		sel.clearEdges(panelId);
	}

	function nodeBoundsFromFlow(id: string, x: number, y: number): Bounds {
		const flowNode = flowNodes.find((n) => n.id === id);
		const w = flowNode?.measured?.width ?? DEFAULT_NODE_W;
		const h = flowNode?.measured?.height ?? DEFAULT_NODE_H;
		return makeBounds(x, y, w, h);
	}

	function buildMeasurements(): Map<string, { width: number; height: number }> {
		const m = new Map<string, { width: number; height: number }>();
		for (const n of flowNodes) {
			if (n.measured?.width && n.measured?.height) {
				m.set(n.id, { width: n.measured.width, height: n.measured.height });
			}
		}
		return m;
	}

	function dragSnapDelta(
		current: Map<string, { x: number; y: number }>,
		altKey: boolean
	): { dx: number; dy: number; guides: Guide[] } {
		const draggedBounds: Bounds[] = [];
		for (const [id, pos] of current) draggedBounds.push(nodeBoundsFromFlow(id, pos.x, pos.y));
		const targets: Bounds[] = [];
		for (const n of g.nodes) {
			if (current.has(n.name)) continue;
			targets.push(nodeBoundsFromFlow(n.name, n.pos[0], n.pos[1]));
		}
		return computeSnapDelta(draggedBounds, targets, altKey);
	}

	function onNodeDragStart(_args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		// No bookkeeping needed: snap is computed each frame off the live
		// args.nodes positions reported by SvelteFlow.
	}

	function onNodeDrag(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		const current = new Map<string, { x: number; y: number }>();
		for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
		const alt = (args.event as MouseEvent).altKey === true;
		const { dx, dy, guides } = dragSnapDelta(current, alt);
		snapGuides = guides;
		if (dx === 0 && dy === 0) return;
		const dragged = new Set(args.nodes.map((n) => n.id));
		flowNodes = flowNodes.map((n) => {
			if (!dragged.has(n.id)) return n;
			const c = current.get(n.id);
			if (!c) return n;
			return { ...n, position: { x: c.x + dx, y: c.y + dy } };
		});
	}

	function onNodeDragStop(args: {
		targetNode: Node | null;
		nodes: Node[];
		event: MouseEvent | TouchEvent;
	}): void {
		const current = new Map<string, { x: number; y: number }>();
		for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
		const alt = (args.event as MouseEvent).altKey === true;
		const { dx, dy } = dragSnapDelta(current, alt);
		if (dx !== 0 || dy !== 0) {
			const dragged = new Set(args.nodes.map((n) => n.id));
			flowNodes = flowNodes.map((n) => {
				if (!dragged.has(n.id)) return n;
				const c = current.get(n.id);
				if (!c) return n;
				return { ...n, position: { x: c.x + dx, y: c.y + dy } };
			});
		}
		for (const n of args.nodes) {
			void g.setNodePos(n.id, [Math.round(n.position.x + dx), Math.round(n.position.y + dy)]);
		}
		snapGuides = [];
	}

	let lastPaneClickAt = 0;
	let lastPaneClickPos = { x: 0, y: 0 };
	const DOUBLE_CLICK_MS = 350;

	function onPaneClick(args: { event: MouseEvent }): void {
		const now = performance.now();
		const here = { x: args.event.clientX, y: args.event.clientY };
		const dt = now - lastPaneClickAt;
		const ddx = here.x - lastPaneClickPos.x;
		const ddy = here.y - lastPaneClickPos.y;
		const close = ddx * ddx + ddy * ddy < 30 * 30;
		if (dt < DOUBLE_CLICK_MS && close) {
			// Double-click on empty canvas → open add-node menu at the click.
			menuPos = {
				x: Math.max(8, Math.min(window.innerWidth - 332, here.x - 8)),
				y: Math.max(8, Math.min(window.innerHeight - 360, here.y + 8))
			};
			menuSeed = null;
			menuOpen = true;
			lastPaneClickAt = 0;
			return;
		}
		lastPaneClickAt = now;
		lastPaneClickPos = here;
		menuOpen = false;
		sel.clickPane(panelId, args.event.shiftKey);
	}

	function onNodeClick(args: { node: Node; event: MouseEvent | TouchEvent }): void {
		const mouse = args.event as MouseEvent;
		sel.clickNode(panelId, args.node.id, mouse.shiftKey || mouse.ctrlKey || mouse.metaKey);
	}

	const nodeTypes = { goofi: GoofiNode };

	function onKeydown(e: KeyboardEvent): void {
		if (!isActive()) return;
		const tag = (e.target as HTMLElement | null)?.tagName ?? '';
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

		const meta = e.ctrlKey || e.metaKey;
		if (meta && e.key.toLowerCase() === 'a') {
			e.preventDefault();
			sel.selectNodes(panelId, g.nodes.map((n) => n.name));
		} else if (meta && e.key.toLowerCase() === 'c') {
			void copySelection();
		} else if (meta && e.key.toLowerCase() === 'v') {
			void pasteClipboard();
		} else if (meta && e.key.toLowerCase() === 'd') {
			e.preventDefault();
			void duplicateSelection();
		} else if (e.key === 'Delete' || e.key === 'Backspace') {
			if (sel.nodes(panelId).size === 0 && sel.edges(panelId).size === 0) return;
			e.preventDefault();
			if (sel.nodes(panelId).size > 0) void deleteSelection();
			if (sel.edges(panelId).size > 0) void deleteEdgeSelection();
		} else if (e.key === 'Tab') {
			e.preventDefault();
			openMenuAtCursor();
		} else if (e.key === 'Escape') {
			if (menuOpen) {
				menuOpen = false;
				menuSeed = null;
			} else {
				sel.clear(panelId);
			}
		} else if (e.key.toLowerCase() === 'f') {
			fitView();
		}
	}

	async function deleteSelection(): Promise<void> {
		const names = Array.from(sel.nodes(panelId));
		for (const n of names) {
			try {
				await g.removeNode(n);
			} catch (e) {
				console.warn('remove failed', e);
			}
		}
		sel.clearNodes(panelId);
	}

	async function copySelection(): Promise<void> {
		const selNodes = g.nodes.filter((n) => sel.nodes(panelId).has(n.name));
		if (selNodes.length === 0) return;
		const avg = selNodes.reduce((acc, n) => [acc[0] + n.pos[0], acc[1] + n.pos[1]], [0, 0]);
		const avgX = avg[0] / selNodes.length;
		const avgY = avg[1] / selNodes.length;
		const links = g.links.filter(
			(l) => sel.nodes(panelId).has(l.node_in) && sel.nodes(panelId).has(l.node_out)
		);
		const payload = {
			__goofi_clip__: 1,
			nodes: selNodes.map((n) => ({
				name: n.name,
				type: n.type,
				category: n.category,
				params: serializableParams(n),
				offset: [n.pos[0] - avgX, n.pos[1] - avgY]
			})),
			links
		};
		try {
			await navigator.clipboard.writeText(JSON.stringify(payload));
		} catch (e) {
			console.warn('clipboard write failed', e);
		}
	}

	async function duplicateSelection(): Promise<void> {
		const selNodes = g.nodes.filter((n) => sel.nodes(panelId).has(n.name));
		if (selNodes.length === 0) return;
		const OFFSET = 40;
		const internalLinks = g.links.filter(
			(l) => sel.nodes(panelId).has(l.node_in) && sel.nodes(panelId).has(l.node_out)
		);
		const rename: Record<string, string> = {};
		const newSelection = new Set<string>();
		for (const n of selNodes) {
			try {
				const newName = await g.addNode(n.type, n.category, [n.pos[0] + OFFSET, n.pos[1] + OFFSET]);
				rename[n.name] = newName;
				newSelection.add(newName);
				for (const [group, params] of Object.entries(n.params)) {
					for (const [name, p] of Object.entries(params)) {
						try {
							await g.updateParam(newName, group, name, p.value);
						} catch {
							/* ignore */
						}
					}
				}
			} catch (e) {
				console.warn('duplicate: add_node failed', e);
			}
		}
		for (const l of internalLinks) {
			try {
				await g.addLink({
					node_out: rename[l.node_out] ?? l.node_out,
					node_in: rename[l.node_in] ?? l.node_in,
					slot_out: l.slot_out,
					slot_in: l.slot_in
				});
			} catch {
				/* ignore */
			}
		}
		if (newSelection.size > 0) sel.selectNodes(panelId, newSelection);
	}

	function serializableParams(n: NodeInstanceInfo): Record<string, Record<string, unknown>> {
		const out: Record<string, Record<string, unknown>> = {};
		for (const [group, params] of Object.entries(n.params)) {
			out[group] = {};
			for (const [name, p] of Object.entries(params)) {
				out[group][name] = p.value;
			}
		}
		return out;
	}

	async function pasteClipboard(): Promise<void> {
		let text = '';
		try {
			text = await navigator.clipboard.readText();
		} catch {
			return;
		}
		let payload: {
			__goofi_clip__?: number;
			nodes: {
				name: string;
				type: string;
				category: string;
				params: Record<string, Record<string, unknown>>;
				offset: [number, number];
			}[];
			links: LinkInfo[];
		};
		try {
			payload = JSON.parse(text);
		} catch {
			return;
		}
		if (payload?.__goofi_clip__ !== 1) return;
		const cx = window.innerWidth / 2;
		const cy = window.innerHeight / 2;
		const rename: Record<string, string> = {};
		for (const n of payload.nodes) {
			try {
				const newName = await g.addNode(n.type, n.category, [
					Math.round(cx / 2 + n.offset[0]),
					Math.round(cy / 2 + n.offset[1])
				]);
				rename[n.name] = newName;
				for (const [group, params] of Object.entries(n.params)) {
					for (const [name, value] of Object.entries(params)) {
						try {
							await g.updateParam(newName, group, name, value);
						} catch {
							/* ignore */
						}
					}
				}
			} catch (e) {
				console.warn('paste: add_node failed', e);
			}
		}
		for (const l of payload.links) {
			try {
				await g.addLink({
					node_out: rename[l.node_out] ?? l.node_out,
					node_in: rename[l.node_in] ?? l.node_in,
					slot_out: l.slot_out,
					slot_in: l.slot_in
				});
			} catch {
				/* ignore */
			}
		}
	}

	function openMenuAtCursor(): void {
		menuPos = { x: mouseX, y: mouseY };
		menuSeed = null;
		menuOpen = true;
	}

	/** Open the add-node menu centered over this panel (TopBar "Add node"). */
	function openAddMenuCentered(): void {
		const r = rootEl?.getBoundingClientRect();
		if (r) menuPos = { x: r.left + r.width / 2 - 160, y: r.top + 60 };
		else menuPos = { x: window.innerWidth / 2 - 160, y: 80 };
		menuSeed = null;
		menuOpen = true;
	}

	async function autoLink(
		seed: SlotClickSeed,
		picked: NodeTypeInfo,
		newName: string
	): Promise<void> {
		const candidates =
			seed.side === 'source'
				? Object.entries(picked.input_slots)
				: Object.entries(picked.output_slots);
		const match = candidates.find(([, dt]) => dt === seed.dtype);
		if (!match) return;
		const [matchedSlot] = match;
		const link =
			seed.side === 'source'
				? { node_out: seed.node, slot_out: seed.slot, node_in: newName, slot_in: matchedSlot }
				: { node_out: newName, slot_out: matchedSlot, node_in: seed.node, slot_in: seed.slot };
		try {
			await g.addLink(link);
		} catch (e) {
			console.warn('auto-link failed', e);
		}
	}

	async function commitPlacement(pos: [number, number]): Promise<void> {
		const placement = pendingPlacement;
		if (!placement) return;
		pendingPlacement = null;
		try {
			const newName = await g.addNode(placement.typeInfo.type, placement.typeInfo.category, pos);
			if (placement.seed && newName) await autoLink(placement.seed, placement.typeInfo, newName);
		} catch (e) {
			console.warn('add_node failed', e);
		}
	}

	let mouseX = 0;
	let mouseY = 0;
	function trackMouse(e: MouseEvent): void {
		mouseX = e.clientX;
		mouseY = e.clientY;
	}

	function fitView(): void {
		rootEl?.querySelector<HTMLButtonElement>('.svelte-flow__controls-fitview')?.click();
	}

	onMount(() => {
		registerEditor(panelId, { openAddMenu: openAddMenuCentered, fitView });
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('mousemove', trackMouse);
		return () => {
			unregisterEditor(panelId);
			// NB: do NOT forget this panel's selection here — unmount also fires
			// on a tab switch (the inactive tab's tree is torn down), and the
			// selection must survive switching away and back. It only clears
			// when the user clicks blank space in the focused editor.
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('mousemove', trackMouse);
		};
	});
</script>

<SvelteFlowProvider>
	<!-- `canvas-wrap` is the marker PlacementPreview uses to tell a commit
	     click (inside the canvas) from a cancel click (outside). -->
	<div class="editor-panel canvas-wrap" bind:this={rootEl}>
		<SvelteFlow
			bind:nodes={flowNodes}
			bind:edges={flowEdges}
			{nodeTypes}
			onconnect={onConnect}
			onnodedragstart={onNodeDragStart}
			onnodedrag={onNodeDrag}
			onnodedragstop={onNodeDragStop}
			onpaneclick={onPaneClick}
			onnodeclick={onNodeClick}
			onedgeclick={onEdgeClick}
			ondelete={async ({ nodes, edges }) => {
				for (const n of nodes) await g.removeNode(n.id).catch(() => {});
				for (const e of edges) {
					const so = e.sourceHandle;
					const si = e.targetHandle;
					if (so && si)
						await g
							.removeLink({ node_out: e.source, node_in: e.target, slot_out: so, slot_in: si })
							.catch(() => {});
				}
			}}
			fitView
			fitViewOptions={{ maxZoom: 1, padding: 0.18 }}
			minZoom={0.05}
			maxZoom={4}
			initialViewport={{ x: 0, y: 0, zoom: 0.85 }}
			zoomOnDoubleClick={false}
		>
			<Controls />
			<MiniMap pannable zoomable />
			{#if pendingPlacement}
				<PlacementPreview
					typeInfo={pendingPlacement.typeInfo}
					initialClient={pendingPlacement.initialClient}
					measurements={buildMeasurements()}
					onCommit={(pos) => void commitPlacement(pos)}
					onCancel={() => {
						pendingPlacement = null;
					}}
				/>
			{/if}
			{#if snapGuides.length > 0}
				<ViewportPortal target="front">
					<svg class="snap-guides" data-testid="snap-guides">
						{#each snapGuides as guide, i (i)}
							{#if guide.x !== undefined}
								<line
									x1={guide.x}
									x2={guide.x}
									y1={-5000}
									y2={5000}
									stroke="var(--accent)"
									stroke-width="1"
									stroke-opacity={guide.opacity}
								/>
							{:else if guide.y !== undefined}
								<line
									x1={-5000}
									x2={5000}
									y1={guide.y}
									y2={guide.y}
									stroke="var(--accent)"
									stroke-width="1"
									stroke-opacity={guide.opacity}
								/>
							{/if}
						{/each}
					</svg>
				</ViewportPortal>
			{/if}
		</SvelteFlow>

		{#if menuOpen}
			<div
				class="menu-overlay"
				onclick={() => {
					menuOpen = false;
					menuSeed = null;
				}}
				role="presentation"
			></div>
			<div class="menu-anchor" style="left: {menuPos.x}px; top: {menuPos.y}px">
				<AddNodeMenu
					seed={menuSeed}
					onPick={(typeInfo) => {
						const seed = menuSeed;
						menuOpen = false;
						menuSeed = null;
						pendingPlacement = {
							typeInfo,
							seed,
							initialClient: { x: mouseX, y: mouseY }
						};
					}}
					onClose={() => {
						menuOpen = false;
						menuSeed = null;
					}}
				/>
			</div>
		{/if}

		<!-- Per-editor selection inspector — slides in within this panel. -->
		<InspectorOverlay node={selectedNode} onFocus={(name) => sel.selectNodes(panelId, [name])} />
	</div>
</SvelteFlowProvider>

<style>
	.editor-panel {
		position: relative;
		width: 100%;
		height: 100%;
		min-width: 0;
		min-height: 0;
	}
	/* Nudge SvelteFlow's controls + minimap off the panel corners so the
	   corner grips (drag-split / drag-join) stay reachable. */
	.editor-panel :global(.svelte-flow__controls) {
		bottom: 20px;
		left: 20px;
	}
	.editor-panel :global(.svelte-flow__minimap) {
		bottom: 20px;
		right: 20px;
	}
	.menu-overlay {
		position: fixed;
		inset: 0;
		z-index: calc(var(--z-addmenu) - 1);
	}
	.menu-anchor {
		position: fixed;
		z-index: var(--z-addmenu);
		width: 320px;
	}
	.snap-guides {
		position: absolute;
		left: 0;
		top: 0;
		width: 1px;
		height: 1px;
		overflow: visible;
		pointer-events: none;
	}
</style>
