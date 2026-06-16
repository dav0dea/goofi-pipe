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
		type FitViewOptions,
		type Node
	} from '@xyflow/svelte';
	import GoofiNode from '$lib/editor/GoofiNode.svelte';
	import AddNodeMenu from '$lib/editor/AddNodeMenu.svelte';
	import PlacementPreview from '$lib/editor/PlacementPreview.svelte';
	import FitToGraph from '$lib/editor/FitToGraph.svelte';
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
	import { getPanelType, type PanelProps } from '$lib/workspace/registry';
	import { portal } from '$lib/workspace/portal';
	import { linkKey, type LinkInfo, type NodeTypeInfo } from '$lib/api/control';
	import { serializeClipboard, parseClipboard, clipToSpecs } from '$lib/editor/clipboard';
	import { copyText } from '$lib/clipboard';
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

	// Whether this editor's inspector pane is enabled (per-panel; default on).
	const inspectorOn = $derived(sel.inspectorEnabledFor(panelId));

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
			const id = linkKey(l);
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
		for (const l of g.links) if (linkKey(l) === id) return l;
		return null;
	}

	async function deleteEdgeSelection(): Promise<void> {
		const links = Array.from(sel.edges(panelId), findLinkById).filter(
			(l): l is LinkInfo => l !== null
		);
		await g.removeLinks(links);
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

	// Positions at drag start, so a node reverts (snaps back) when the drag
	// turns into a panel-link rather than an in-editor reposition.
	let dragOrigin = new Map<string, { x: number; y: number }>();
	// Floating chip following the cursor while a drag is a reference (not a
	// coordinate move). null = normal reposition drag.
	let linkGhost = $state<{ x: number; y: number; name: string } | null>(null);

	/** The leaf panel under a screen point (panels tile, so at most one). Found
	 * geometrically off panel rects — NOT elementFromPoint, since the dragged
	 * node sits under the cursor and would mask the panel beneath it. */
	function panelUnder(x: number, y: number): { id: string; type: string } | null {
		for (const el of document.querySelectorAll<HTMLElement>('[data-panel-id]')) {
			const r = el.getBoundingClientRect();
			if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) {
				return { id: el.dataset.panelId ?? '', type: el.dataset.panelType ?? '' };
			}
		}
		return null;
	}

	/** A node-accepting panel under the cursor, other than this editor — i.e. a
	 * valid drop target that turns the drag into a reference link. */
	function linkTargetAt(event: MouseEvent | TouchEvent): { id: string; type: string } | null {
		const m = event as MouseEvent;
		if (typeof m.clientX !== 'number') return null;
		const t = panelUnder(m.clientX, m.clientY);
		return t && t.id !== panelId && getPanelType(t.type)?.acceptsNode === true ? t : null;
	}

	/** Put the dragged nodes back where the drag started (reference mode). */
	function revertDragged(dragged: Set<string>): void {
		flowNodes = flowNodes.map((n) => {
			if (!dragged.has(n.id)) return n;
			const o = dragOrigin.get(n.id);
			return o ? { ...n, position: { x: o.x, y: o.y } } : n;
		});
	}

	function onNodeDragStart(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		dragOrigin = new Map();
		for (const n of args.nodes) dragOrigin.set(n.id, { x: n.position.x, y: n.position.y });
		// Flag the drag so node-accepting panels show their drop outline.
		uiStore.nodeDrag = args.nodes[0]?.id ?? null;
	}

	function onNodeDrag(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		const dragged = new Set(args.nodes.map((n) => n.id));
		const target = linkTargetAt(args.event);
		if (target) {
			// Reference drag: snap the node back to its origin and let a ghost
			// follow the cursor instead — it's a reference, not a coordinate move.
			uiStore.nodeDragTarget = target.id;
			const m = args.event as MouseEvent;
			linkGhost = { x: m.clientX, y: m.clientY, name: args.nodes[0]?.id ?? '' };
			snapGuides = [];
			revertDragged(dragged);
			return;
		}
		// Normal reposition drag with snapping.
		uiStore.nodeDragTarget = null;
		linkGhost = null;
		const current = new Map<string, { x: number; y: number }>();
		for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
		const alt = (args.event as MouseEvent).altKey === true;
		const { dx, dy, guides } = dragSnapDelta(current, alt);
		snapGuides = guides;
		if (dx === 0 && dy === 0) return;
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
		const dragged = new Set(args.nodes.map((n) => n.id));
		const target = linkTargetAt(args.event);
		if (target) {
			// Dropped on a node-accepting panel → link the node there and leave
			// it where it started (the reference, not the node, moved).
			revertDragged(dragged);
			const name = args.nodes[0]?.id;
			if (name) ws.linkNodeToPanel(target.id, name);
		} else {
			const current = new Map<string, { x: number; y: number }>();
			for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
			const alt = (args.event as MouseEvent).altKey === true;
			const { dx, dy } = dragSnapDelta(current, alt);
			if (dx !== 0 || dy !== 0) {
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
		}
		uiStore.nodeDrag = null;
		uiStore.nodeDragTarget = null;
		linkGhost = null;
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

	/** Framing for every programmatic fit — the Controls/“F” button (via the
	 * `fitViewOptions` prop) and the on-load fit in <FitToGraph>. */
	const FIT_OPTIONS = { maxZoom: 1, padding: 0.18 } satisfies FitViewOptions;

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
		await g.removeNodes(Array.from(sel.nodes(panelId)));
		sel.clearNodes(panelId);
	}

	async function copySelection(): Promise<void> {
		const names = sel.nodes(panelId);
		const selNodes = g.nodes.filter((n) => names.has(n.name));
		if (selNodes.length === 0) return;
		const links = g.links.filter((l) => names.has(l.node_in) && names.has(l.node_out));
		if (!(await copyText(JSON.stringify(serializeClipboard(selNodes, links))))) {
			console.warn('clipboard write failed');
		}
	}

	async function duplicateSelection(): Promise<void> {
		const rename = await g.cloneNodes(sel.nodes(panelId), [40, 40]);
		const created = Object.values(rename);
		if (created.length > 0) sel.selectNodes(panelId, created);
	}


	async function pasteClipboard(): Promise<void> {
		let text = '';
		try {
			text = await navigator.clipboard.readText();
		} catch {
			return;
		}
		const clip = parseClipboard(text);
		if (!clip) return;
		// Anchor the paste near the viewport centre (preserves the prior
		// placement math: a quarter of the window, plus each node's offset).
		const at: [number, number] = [window.innerWidth / 4, window.innerHeight / 4];
		const rename = await g.instantiateNodes(clipToSpecs(clip, at), clip.links);
		const created = Object.values(rename);
		if (created.length > 0) sel.selectNodes(panelId, created);
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
		// Inputs take a single source: replace whatever cable was already feeding
		// this input. Outputs fan out, so the source side never disconnects.
		if (seed.side === 'target') {
			const existing = g.links.filter((l) => l.node_in === seed.node && l.slot_in === seed.slot);
			for (const l of existing) await g.removeLink(l).catch(() => {});
		}
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
			// Auto-select the freshly-placed node so its parameters open in the
			// inspector immediately — matching duplicate / paste / agent placement.
			// Safe before node_added lands: flowNodes derives `selected` from this
			// set, so the node renders selected the moment it appears.
			if (newName) sel.selectNodes(panelId, [newName]);
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

	/** Select a node in this editor (and make it the active selection the
	 * standalone panels follow). The shared handle the TopBar, the error panel,
	 * and the agent surface use to focus a node. */
	function focusNode(name: string): void {
		sel.selectNodes(panelId, [name]);
	}

	onMount(() => {
		registerEditor(panelId, { openAddMenu: openAddMenuCentered, fitView, focusNode });
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
			// If a node drag was mid-flight when this editor unmounts, clear the
			// shared drag flags so other panels don't keep showing drop outlines.
			if (uiStore.nodeDrag !== null) {
				uiStore.nodeDrag = null;
				uiStore.nodeDragTarget = null;
			}
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
			fitViewOptions={FIT_OPTIONS}
			minZoom={0.05}
			maxZoom={4}
			initialViewport={{ x: 0, y: 0, zoom: 0.85 }}
			zoomOnDoubleClick={false}
			autoPanOnNodeDrag={false}
		>
			<Controls />
			<MiniMap pannable zoomable />
			<FitToGraph options={FIT_OPTIONS} />
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

		<!-- Always-visible affordance to toggle this editor's inspector. While the
		     inspector is open (a node selected) the pane covers this icon — that's
		     fine; deselecting brings it back into reach. -->
		<button
			class="inspector-toggle"
			class:on={inspectorOn}
			title={inspectorOn ? 'Hide the inspector' : 'Show the inspector when a node is selected'}
			aria-label="Toggle inspector"
			aria-pressed={inspectorOn}
			data-testid="inspector-toggle"
			onclick={() => sel.toggleInspectorFor(panelId)}
		>
			◧
		</button>

		<!-- Per-editor selection inspector — slides in within this panel. -->
		<InspectorOverlay node={selectedNode} enabled={inspectorOn} />
	</div>
</SvelteFlowProvider>

<!-- Reference chip that follows the cursor while a node is dragged onto a
     node-accepting panel — the node itself stays put in the editor. Portaled
     to <body> so it floats above every panel. -->
{#if linkGhost}
	<div class="link-ghost" use:portal style="left: {linkGhost.x}px; top: {linkGhost.y}px">
		<span class="lg-icon">🔗</span>{linkGhost.name}
	</div>
{/if}

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
	/* Per-editor inspector affordance, parked top-right. Subtle until hovered so
	   it doesn't compete with the canvas; only shown while the inspector is off. */
	.inspector-toggle {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 5;
		width: 26px;
		height: 26px;
		display: grid;
		place-items: center;
		padding: 0;
		font-size: 13px;
		background: color-mix(in srgb, var(--bg-elev-1) 80%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-faint);
		opacity: 0.5;
		cursor: pointer;
		transition:
			opacity 100ms ease,
			color 100ms ease,
			border-color 100ms ease;
	}
	.inspector-toggle:hover {
		opacity: 1;
		color: var(--text);
		border-color: var(--accent);
	}
	/* Subtly indicate the inspector is enabled (it's just hidden because nothing
	   is selected) vs. disabled. */
	.inspector-toggle.on {
		opacity: 0.9;
		color: var(--text);
		border-color: var(--accent);
	}
	.link-ghost {
		position: fixed;
		/* Offset off the cursor so it reads as carried, not pinned. */
		transform: translate(14px, 12px);
		z-index: var(--z-tab-drag);
		pointer-events: none;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 9px;
		max-width: 220px;
		background: var(--bg-elev-2);
		border: 1px solid var(--accent);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		font-family: var(--font-mono);
		font-size: 0.78rem;
		color: var(--text);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.lg-icon {
		font-size: 0.8rem;
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
