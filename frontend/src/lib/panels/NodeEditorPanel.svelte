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
		SvelteFlowProvider,
		ViewportPortal,
		type Connection,
		type Edge,
		type FitViewOptions,
		type Node
	} from '@xyflow/svelte';
	import GoofiNode from '$lib/editor/GoofiNode.svelte';
	import BoundaryNode from '$lib/editor/BoundaryNode.svelte';
	import AddNodeMenu from '$lib/editor/AddNodeMenu.svelte';
	import PlacementPreview from '$lib/editor/PlacementPreview.svelte';
	import FitToGraph from '$lib/editor/FitToGraph.svelte';
	import FlowApi from '$lib/editor/FlowApi.svelte';
	import SubpatchZoomExit from '$lib/editor/SubpatchZoomExit.svelte';
	import SnapGuides from '$lib/editor/SnapGuides.svelte';
	import {
		computeSnapDelta,
		makeBounds,
		DEFAULT_NODE_W,
		DEFAULT_NODE_H,
		type Bounds,
		type Guide
	} from '$lib/editor/snap';
	import { graph } from '$lib/stores/graph.svelte';
	import { history } from '$lib/stores/history.svelte';
	import { ui, type SlotClickSeed } from '$lib/stores/ui.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { getPanelType, type PanelProps } from '$lib/workspace/registry';
	import { portal } from '$lib/workspace/portal';
	import {
		linkKey,
		BOUNDARY_TYPES,
		boundarySpec,
		type InstanceInfo,
		type LinkInfo,
		type NodeInstanceInfo,
		type NodeTypeInfo,
		type SubPatchPort
	} from '$lib/api/control';
	import {
		ROOT_ID,
		buildMemberIndex,
		childrenOfScope,
		drawEndpoint as sceneDrawEndpoint,
		boundaryNodeId,
		isBoundaryNodeId,
		parseBoundaryNodeId
	} from '$lib/editor/subpatchScene';
	import { nodeSurfaceSize, BOUNDARY } from '$lib/editor/nodeMetrics';
	import { serializeClipboard, parseClipboard, clipToSpecs } from '$lib/editor/clipboard';
	import { copyText } from '$lib/clipboard';
	import { registerEditor, unregisterEditor } from './editorCommands';
	import InspectorOverlay from './InspectorOverlay.svelte';
	import { asStateObject } from '$lib/workspace/panelState';
	import { onMount, untrack } from 'svelte';

	let { panelId, state: panelState, setState }: PanelProps = $props();

	// The sub-patch this editor is currently inside, as a path persisted in the
	// panel's layout state so save/reload (and same-session reconnect) recover the
	// exact view: '/' is the root patch, '/subpatch0' / '/subpatch0/subpatch1'
	// descend into nested sub-patches.
	function pathToArray(p: unknown): string[] {
		return typeof p === 'string' ? p.split('/').filter(Boolean) : [];
	}
	function arrayToPath(a: string[]): string {
		return '/' + a.join('/');
	}
	function samePath(a: string[], b: string[]): boolean {
		return a.length === b.length && a.every((v, i) => v === b[i]);
	}

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

	// --- sub-patch navigation (enter-to-edit) -------------------------------
	// The stack of instance ids the editor has descended into; empty = top level.
	// (A stack so nesting is a natural extension; today one level deep is reached.)
	// Seeded from the persisted layout state so a saved/reloaded patch restores
	// which sub-patch this editor was inside.
	let enteredPath = $state<string[]>(untrack(() => pathToArray(asStateObject(panelState).subpatchPath)));
	const entered = $derived(enteredPath.length ? enteredPath[enteredPath.length - 1] : null);

	/** Write the current path back into the panel's layout state (round-trips
	 * through set_layout / the saved .gfi). Untracked so callers in effects don't
	 * pick up `state` as a dependency. */
	function persistEnteredPath(): void {
		untrack(() => {
			const path = arrayToPath(enteredPath);
			if (asStateObject(panelState).subpatchPath === path) return;
			setState({ ...asStateObject(panelState), subpatchPath: path });
		});
	}

	// Follow an EXTERNAL path change (a patch load that reuses this panel id, so
	// the component isn't remounted). Our own writes leave the two equal, so this
	// is a no-op for them; an invalid restored id is trimmed by the climb-out
	// effect below.
	$effect(() => {
		const persisted = pathToArray(asStateObject(panelState).subpatchPath);
		if (!samePath(persisted, untrack(() => enteredPath))) enteredPath = persisted;
	});

	// Boundary-pill id format lives in the pure, tested scene module (single source of
	// truth). `boundaryId`/`isBoundaryId` keep their short names at the call sites.
	const boundaryId = boundaryNodeId;
	const isBoundaryId = isBoundaryNodeId;

	/** Index every entity (node OR nested instance) by uid -> {instId, local, ...}.
	 * The single source for parent-scope + local lookups across the recursive tree. */
	const memberIndex = $derived(buildMemberIndex(g.instances));

	function enterInstance(instId: string): void {
		if (!g.instances[instId]) return;
		if (enteredPath[enteredPath.length - 1] === instId) return; // already inside it
		sel.clear(panelId);
		enteredPath = [...enteredPath, instId];
		persistEnteredPath();
		setTimeout(fitView, 60); // frame the inside once it has rendered
	}

	/** Pop the breadcrumb back to `depth` levels (0 = top of the patch). */
	function exitToDepth(depth: number): void {
		sel.clear(panelId);
		enteredPath = enteredPath.slice(0, depth);
		persistEnteredPath();
		setTimeout(fitView, 60);
	}

	// If an entered instance is dissolved/removed elsewhere, climb back out of it.
	$effect(() => {
		if (enteredPath.length === 0) return;
		let depth = enteredPath.length;
		while (depth > 0 && !g.instances[enteredPath[depth - 1]]) depth--;
		if (depth !== enteredPath.length) {
			enteredPath = enteredPath.slice(0, depth);
			persistEnteredPath();
		}
	});

	/** Resolve a link endpoint to what's actually drawn in the entered scope: walk up
	 * the nesting tree to the nearest visible boundary port (tree-aware). Null when the
	 * slot is not exposed up the chain, or the endpoint is outside the entered subtree. */
	function drawEndpoint(
		node: string,
		slot: string,
		dir: 'in' | 'out'
	): { node: string; handle: string } | null {
		// Root ≡ a scope: the scene algebra always takes a real scope id — ROOT_ID at the
		// root patch (childrenOfScope(ROOT_ID) renders its members). `entered` stays null at
		// root for the "are we inside a sub-patch?" decisions (boundary wiring, member adds).
		return sceneDrawEndpoint(node, slot, dir, entered ?? ROOT_ID, g.instances, memberIndex);
	}

	// Render the DIRECT CHILDREN of the entered scope (ROOT_ID = the root patch): real
	// nodes AND nested instances (collapsed, double-click-enterable group nodes via the
	// SAME GoofiNode component); plus, inside an entered instance, its In/Out boundary pills.
	$effect(() => {
		const scope = entered ?? ROOT_ID;
		const next: Node[] = [];
		const kids = childrenOfScope(scope, g.instances, g.nodes.map((n) => n.uid), memberIndex);
		const childUids = [...kids.nodeUids, ...kids.instUids];
		const pts = childUids
			.map((u) => g.nodeById(u)?.pos)
			.filter((q): q is [number, number] => !!q);
		const minX = pts.length ? Math.min(...pts.map((q) => q[0])) : 0;
		const maxX = pts.length ? Math.max(...pts.map((q) => q[0])) : 0;
		const minY = pts.length ? Math.min(...pts.map((q) => q[1])) : 0;
		for (const uid of childUids) {
			const n = g.nodeById(uid);
			if (!n) continue;
			next.push({
				id: uid,
				type: 'goofi',
				position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
				data: { node: n, label: n.name },
				selected: sel.nodes(panelId).has(uid)
			});
		}
		// Inside an entered instance, also render its In/Out boundary pills (incl.
		// unwired). Stored pos, with a beside-the-children fallback for legacy entries.
		const inst = entered ? g.instances[entered] : null;
		if (inst && entered) {
			const ins = Object.entries(inst.interface).filter(([, p]) => p.dir === 'in');
			const outs = Object.entries(inst.interface).filter(([, p]) => p.dir === 'out');
			const place = (entries: [string, SubPatchPort][], dir: 'in' | 'out', fallbackX: number) =>
				entries.forEach(([name, port], i) => {
					next.push({
						id: boundaryId(entered, name),
						type: 'boundary',
						position: port.pos
							? { x: port.pos[0], y: port.pos[1] }
							: { x: fallbackX, y: minY + i * 96 },
						data: { name, dir, dtype: port.dtype ?? 'ARRAY', wired: port.inner_node !== null },
						selected: sel.nodes(panelId).has(boundaryId(entered, name))
					});
				});
			place(ins, 'in', minX - 280);
			place(outs, 'out', maxX + 320);
		}
		flowNodes = next;
	});

	// Every link, rerouted to the nearest VISIBLE boundary in the entered scope (null =
	// root). Skip when an endpoint is not exposed up the chain, or both ends resolve to
	// the same drawn node (internal to one collapsed sub-patch -> hidden). Inside an
	// entered instance, also draw its boundary pill edges (In pill -> member input,
	// member output -> Out pill); WIRED boundaries only, deletable to unwire.
	$effect(() => {
		const next: Edge[] = [];
		for (const l of g.links) {
			const src = drawEndpoint(l.node_out, l.slot_out, 'out');
			const dst = drawEndpoint(l.node_in, l.slot_in, 'in');
			if (!src || !dst) continue;
			if (src.node === dst.node && l.node_out !== l.node_in) continue; // internal to one collapsed child -> hidden (but keep a real self-loop)
			const id = linkKey(l);
			next.push({
				id,
				source: src.node,
				sourceHandle: src.handle,
				target: dst.node,
				targetHandle: dst.handle,
				selected: sel.edges(panelId).has(id),
				animated: false
			});
		}
		const inst = entered ? g.instances[entered] : null;
		if (inst && entered) {
			for (const [name, port] of Object.entries(inst.interface)) {
				if (port.inner_node == null) continue;
				const disp = g.memberUid(entered, port.inner_node);
				if (!disp) continue;
				const bId = boundaryId(entered, name);
				if (port.dir === 'in') {
					const id = `${bId}->${disp}.${port.inner_slot}`;
					next.push({
						id,
						source: bId,
						sourceHandle: 'out',
						target: disp,
						targetHandle: port.inner_slot,
						selected: sel.edges(panelId).has(id),
						animated: false
					});
				} else {
					const id = `${disp}.${port.inner_slot}->${bId}`;
					next.push({
						id,
						source: disp,
						sourceHandle: port.inner_slot,
						target: bId,
						targetHandle: 'in',
						selected: sel.edges(panelId).has(id),
						animated: false
					});
				}
			}
		}
		flowEdges = next;
	});

	/** Real-node ids selected in THIS editor — the operands for group / copy /
	 * duplicate. Unions the app selection store (click / shift-click / Ctrl+A)
	 * with Svelte Flow's live `selected` flags (marquee / box select) — the store
	 * alone misses marquee selections. Sub-patch group nodes are excluded on
	 * purpose: a sub-patch can't round-trip through the generic node clone/
	 * clipboard path (it carries definition + membership state), so duplicating a
	 * sub-patch goes through the inspector's explicit "Duplicate as shared".
	 * (Delete is the exception — it reads the raw selection so it can remove a
	 * sub-patch too, via remove_instance.) */
	function selectedNodeNames(): string[] {
		const ids = new Set<string>(sel.nodes(panelId));
		for (const n of flowNodes) if (n.selected) ids.add(n.id);
		return [...ids].filter((id) => !(id in g.instances) && !isBoundaryId(id));
	}

	async function groupSelection(): Promise<void> {
		const names = selectedNodeNames();
		if (names.length === 0) return;
		// Place the collapsed group node at the centroid of its members.
		const pts = names.map((n) => g.nodeById(n)?.pos).filter((p): p is [number, number] => !!p);
		const pos: [number, number] = pts.length
			? [
					Math.round(pts.reduce((a, p) => a + p[0], 0) / pts.length),
					Math.round(pts.reduce((a, p) => a + p[1], 0) / pts.length)
				]
			: [0, 0];
		try {
			const instId = await g.groupNodes(names, pos);
			sel.selectNodes(panelId, [instId]);
		} catch (e) {
			console.warn('group failed', e);
		}
	}

	/** The boundary (interface) key of a pill id that belongs to the ENTERED scope, or
	 * null. Scope-checked in the codec so a cross-instance id can't mis-slice. */
	function parseBoundary(id: string): string | null {
		return parseBoundaryNodeId(id, entered);
	}

	function onConnect(c: Connection): void {
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
		// Inside the entered view, an edge touching an In/Out pill wires that
		// boundary to the member slot on the other end (defines the sub-patch port).
		const srcB = parseBoundary(c.source);
		const dstB = parseBoundary(c.target);
		if (srcB || dstB) {
			if (!entered) return;
			const bnd = srcB ?? dstB!;
			const memberId = srcB ? c.target : c.source;
			const memberSlot = srcB ? c.targetHandle : c.sourceHandle;
			const local = memberIndex.get(memberId)?.local;
			if (!local) return; // the other end must be a member of this sub-patch
			void g.wireBoundary(entered, bnd, local, memberSlot).catch((e) =>
				console.warn('wire boundary failed', e)
			);
			return;
		}
		// Otherwise a normal link. A top-level wire to a collapsed sub-patch port
		// (target/source is an instance id, handle is the boundary id) is sent
		// as-is: the bridge splices it to the inner member's flat link.
		void g.addLink({
			node_out: c.source,
			node_in: c.target,
			slot_out: c.sourceHandle,
			slot_in: c.targetHandle
		});
	}

	/** Drag an existing edge's endpoint to a new slot — re-target in place instead
	 * of delete-then-redraw (backlog #25). Removing the old link + adding the new
	 * one is wrapped in one history entry so a single undo reverts the move.
	 * Boundary edges are left to the explicit wire/unwire flow. */
	function onReconnect(oldEdge: Edge, c: Connection): void {
		if (
			parseBoundary(oldEdge.source) ||
			parseBoundary(oldEdge.target) ||
			parseBoundary(c.source ?? '') ||
			parseBoundary(c.target ?? '')
		)
			return;
		const oldSo = oldEdge.sourceHandle;
		const oldSi = oldEdge.targetHandle;
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle || !oldSo || !oldSi) return;
		void history().transaction('Reconnect link', async () => {
			await g.removeLink({ node_out: oldEdge.source, node_in: oldEdge.target, slot_out: oldSo, slot_in: oldSi });
			await g.addLink({
				node_out: c.source as string,
				node_in: c.target as string,
				slot_out: c.sourceHandle as string,
				slot_in: c.targetHandle as string
			});
		});
	}

	function onEdgeClick(args: { edge: Edge; event: MouseEvent }): void {
		const e = args.event;
		sel.clickEdge(panelId, args.edge.id, e.shiftKey || e.ctrlKey || e.metaKey);
	}

	/** A node's snap footprint when Svelte Flow hasn't measured it yet. Computed
	 * per node KIND so every first-class citizen snaps correctly: a sub-patch group
	 * node from its wired-boundary slot layout (it's a `goofi` node like any other),
	 * an In/Out boundary pill from the pill size, and a fresh real node from its
	 * slots — instead of one fixed size that's right only for a typical mid node and
	 * ~100px too tall for a short sub-patch. */
	function nodeFallbackSize(flowNode: Node | undefined): { width: number; height: number } {
		if (flowNode?.type === 'boundary') return { width: BOUNDARY.width, height: BOUNDARY.height };
		const node = flowNode?.data?.node as NodeInstanceInfo | undefined;
		if (node) {
			const inputs = Object.keys(node.input_slots ?? {});
			const outputs = Object.keys(node.output_slots ?? {});
			return nodeSurfaceSize(
				inputs.length,
				outputs.map((s) => uiStore.isSlotExpanded(node.uid, s))
			);
		}
		return { width: DEFAULT_NODE_W, height: DEFAULT_NODE_H };
	}

	function nodeBoundsFromFlow(id: string, x: number, y: number): Bounds {
		const flowNode = flowNodes.find((n) => n.id === id);
		// Prefer the real DOM measurement; fall back to the kind-accurate size so a
		// just-appeared (or transiently unmeasured) node still snaps to its true box.
		let w = flowNode?.measured?.width;
		let h = flowNode?.measured?.height;
		if (w == null || h == null) {
			const fb = nodeFallbackSize(flowNode);
			w ??= fb.width;
			h ??= fb.height;
		}
		return makeBounds(x, y, w, h);
	}

	/** Snap-target bounds (flow coords) for every node on screen in THIS editor,
	 * excluding `exclude`. The single retrieval shared by the node drag AND the
	 * placement preview, so sub-patch instances and boundary pills are first-class
	 * snap targets in both paths and the two can't diverge. Only what's actually
	 * rendered here (flowNodes) — never hidden members of a collapsed sub-patch, nor
	 * nodes outside the entered sub-patch (which g.nodes would wrongly include). */
	function snapTargetBounds(exclude: Set<string>): Bounds[] {
		const targets: Bounds[] = [];
		for (const n of flowNodes) {
			if (exclude.has(n.id)) continue;
			targets.push(nodeBoundsFromFlow(n.id, n.position.x, n.position.y));
		}
		return targets;
	}

	function dragSnapDelta(
		current: Map<string, { x: number; y: number }>,
		altKey: boolean
	): { dx: number; dy: number; guides: Guide[] } {
		const draggedBounds: Bounds[] = [];
		for (const [id, pos] of current) draggedBounds.push(nodeBoundsFromFlow(id, pos.x, pos.y));
		return computeSnapDelta(draggedBounds, snapTargetBounds(new Set(current.keys())), altKey);
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
			// `id` is the uid; the floating chip shows the display name.
			linkGhost = { x: m.clientX, y: m.clientY, name: g.nodeById(args.nodes[0]?.id ?? '')?.name ?? '' };
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
		// A boundary pill isn't a real node — it can't be linked into a panel; it
		// only repositions (mirrored across shared siblings via set_boundary_pos).
		const draggingBoundary = args.nodes.some((n) => isBoundaryId(n.id));
		const target = draggingBoundary ? null : linkTargetAt(args.event);
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
			// One transaction so moving N selected nodes is a single Ctrl+Z (each
			// set*Pos records synchronously, so they fold into one history entry).
			const label = args.nodes.length > 1 ? `Move ${args.nodes.length} nodes` : 'Move node';
			void history().transaction(label, async () => {
				for (const n of args.nodes) {
					const pos: [number, number] = [Math.round(n.position.x + dx), Math.round(n.position.y + dy)];
					const bnd = parseBoundary(n.id);
					if (bnd && entered) {
						void g.setBoundaryPos(entered, bnd, pos).catch(() => {});
					} else {
						void g.setNodePos(n.id, pos);
					}
				}
			});
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

	function sameMembers(set: Set<string>, ids: string[]): boolean {
		if (set.size !== ids.length) return false;
		for (const id of ids) if (!set.has(id)) return false;
		return true;
	}

	// True only between a box/marquee drag's start and end. Gates onSelectionEnd so
	// it can't resurrect a just-cleared selection if Flow happens to fire its end
	// event for a plain pane click (where `selected` flags may still be stale).
	let boxSelecting = false;

	/** Mirror a finished marquee/box-drag selection into the store, which is then
	 * the single source of truth — so a flowNodes rebuild reads `selected` straight
	 * from it and a marquee selection survives a graph change (report B8).
	 *
	 * Keyed on `onselectionstart`/`onselectionend` (a real box gesture), NOT
	 * `onselectionchange`: a store-driven selection (Ctrl+A, click, paste) replaces
	 * every flowNodes object, which makes Flow emit transient echo events — empty
	 * AND partial — mid-rebuild. Honoring those shrank the store the rebuild had just
	 * populated (the bug that left Ctrl+A→group absorbing only one member). Store-
	 * driven changes never start a box gesture, so there are no echoes to filter;
	 * deselection stays owned by onPaneClick → clickPane, Escape, and node-click. We
	 * read the authoritative `selected` flags Flow set on the bound flowNodes/
	 * flowEdges rather than a payload, since the gesture is now complete. */
	function onSelectionEnd(): void {
		if (!boxSelecting) return;
		boxSelecting = false;
		const nodeIds = flowNodes.filter((n) => n.selected).map((n) => n.id);
		const edgeIds = flowEdges.filter((e) => e.selected).map((e) => e.id);
		if (sameMembers(sel.nodes(panelId), nodeIds) && sameMembers(sel.edges(panelId), edgeIds)) return;
		sel.setSelection(panelId, nodeIds, edgeIds);
	}

	// Double-click a sub-patch group node → enter it. We can't use SvelteFlow's
	// `onnodeclick` (it suppresses the 2nd click of a double-click), nor the native
	// `dblclick` event (the 1st click selects the node, which rebuilds flowNodes and
	// detaches the node element, so dblclick/elementFromPoint resolve to nothing on
	// the 2nd click). So we detect it ourselves: record the group node hit by the
	// 1st click, then a 2nd click at the same spot within the threshold enters it.
	const DBL_PX = 6; // a real double-click barely moves the pointer
	let lastClickInst = '';
	let lastClickAt = 0;
	let lastClickX = 0;
	let lastClickY = 0;
	function onCanvasClick(event: MouseEvent): void {
		const now = performance.now();
		if (
			lastClickInst &&
			now - lastClickAt < DOUBLE_CLICK_MS &&
			Math.abs(event.clientX - lastClickX) < DBL_PX &&
			Math.abs(event.clientY - lastClickY) < DBL_PX
		) {
			const inst = lastClickInst;
			lastClickInst = '';
			enterInstance(inst);
			return;
		}
		const id =
			(event.target as HTMLElement | null)?.closest('.svelte-flow__node')?.getAttribute('data-id') ??
			'';
		lastClickAt = now;
		lastClickX = event.clientX;
		lastClickY = event.clientY;
		lastClickInst = id && id in g.instances ? id : '';
	}

	const nodeTypes = { goofi: GoofiNode, boundary: BoundaryNode };

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
			// Select what's actually on screen: the entered sub-patch's members, or
			// every top-level node PLUS the collapsed sub-patch group nodes (which
			// are virtual nodes — they select like any node).
			const kids = childrenOfScope(
				entered ?? ROOT_ID,
				g.instances,
				g.nodes.map((n) => n.uid),
				memberIndex
			);
			const names = [...kids.nodeUids, ...kids.instUids];
			sel.selectNodes(panelId, names);
		} else if (meta && e.key.toLowerCase() === 'c') {
			void copySelection();
		} else if (meta && e.key.toLowerCase() === 'v') {
			void pasteClipboard();
		} else if (meta && e.key.toLowerCase() === 'd') {
			e.preventDefault();
			void duplicateSelection();
		} else if (meta && e.key.toLowerCase() === 'g') {
			e.preventDefault();
			void groupSelection();
		} else if (e.key === 'Tab') {
			e.preventDefault();
			openMenuAtCursor();
		} else if (e.key === 'Escape') {
			if (menuOpen) {
				menuOpen = false;
				menuSeed = null;
			} else if (sel.nodes(panelId).size || sel.edges(panelId).size) {
				sel.clear(panelId);
			} else if (enteredPath.length) {
				exitToDepth(enteredPath.length - 1); // step one level up
			}
		} else if (e.key.toLowerCase() === 'f') {
			fitView();
		}
	}

	async function copySelection(): Promise<void> {
		const names = new Set(selectedNodeNames());
		const selNodes = g.nodes.filter((n) => names.has(n.uid));
		if (selNodes.length === 0) return;
		const links = g.links.filter((l) => names.has(l.node_in) && names.has(l.node_out));
		if (!(await copyText(JSON.stringify(serializeClipboard(selNodes, links))))) {
			console.warn('clipboard write failed');
		}
	}

	async function duplicateSelection(): Promise<void> {
		// One transaction so duplicating N nodes (+ their internal links) is a single Ctrl+Z.
		const rename = await history().transaction('Duplicate nodes', () =>
			g.cloneNodes(selectedNodeNames(), [40, 40], entered ?? undefined)
		);
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
		// Anchor the paste at the visible viewport centre, in FLOW space — the
		// old screen-space anchor (window/4) landed pasted nodes off-screen once
		// the editor was panned or zoomed (report B9). clipToSpecs adds each
		// node's relative offset on top of this anchor.
		const rect = rootEl?.getBoundingClientRect();
		let at: [number, number] = [window.innerWidth / 4, window.innerHeight / 4];
		if (rect && screenToFlow) {
			const c = screenToFlow({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
			at = [c.x, c.y];
		}
		// One transaction so pasting N nodes (+ their internal links) is a single Ctrl+Z.
		const rename = await history().transaction('Paste nodes', () =>
			g.instantiateNodes(clipToSpecs(clip, at), clip.links, entered ?? undefined)
		);
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
		// An In/Out boundary pseudo-type adds a virtual boundary to the entered
		// sub-patch rather than spawning a real node.
		const bspec = boundarySpec(placement.typeInfo.type);
		if (bspec && placement.typeInfo.category === 'boundary') {
			if (!entered) return;
			// One undo step for "add boundary (+ wire it to the clicked slot)".
			await history().transaction(`Add ${placement.typeInfo.type}`, async () => {
				try {
					const bndId = await g.addBoundary(entered, bspec.dir, bspec.dtype, pos);
					// Seeded from a member slot click → wire the new boundary straight to
					// that slot, so In/Out behave like any other auto-connected node. The
					// seed's node is the member uid; wire_boundary wants its local template
					// key (`inst.members` maps uid -> local).
					if (bndId && placement.seed) {
						const local = memberIndex.get(placement.seed.node)?.local;
						if (local) await g.wireBoundary(entered, bndId, local, placement.seed.slot);
					}
				} catch (e) {
					console.warn('add boundary failed', e);
				}
			});
			return;
		}
		// One undo step for "add node (+ auto-wire to the clicked slot)".
		const label = placement.seed
			? `Add ${placement.typeInfo.type} + connect`
			: `Add ${placement.typeInfo.type}`;
		await history().transaction(label, async () => {
			try {
				// Inside a sub-patch, the node becomes a member of the entered instance.
				const newName = await g.addNode(
					placement.typeInfo.type,
					placement.typeInfo.category,
					pos,
					entered ?? undefined
				);
				// Auto-select the freshly-placed node so its parameters open in the
				// inspector immediately — matching duplicate / paste / agent placement.
				// Safe before node_added lands: flowNodes derives `selected` from this
				// set, so the node renders selected the moment it appears.
				if (newName) sel.selectNodes(panelId, [newName]);
				if (placement.seed && newName) await autoLink(placement.seed, placement.typeInfo, newName);
			} catch (e) {
				console.warn('add_node failed', e);
			}
		});
	}

	let mouseX = 0;
	let mouseY = 0;
	function trackMouse(e: MouseEvent): void {
		mouseX = e.clientX;
		mouseY = e.clientY;
	}

	// Bound from <FlowApi> inside <SvelteFlow>; converts a client point to flow
	// space so paste anchors in the visible viewport (report B9).
	let screenToFlow = $state<((p: { x: number; y: number }) => { x: number; y: number }) | undefined>(
		undefined
	);

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
		rootEl?.addEventListener('click', onCanvasClick);
		return () => {
			unregisterEditor(panelId);
			rootEl?.removeEventListener('click', onCanvasClick);
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
		{#if enteredPath.length > 0}
			<!-- Sub-patch breadcrumb: where in the patch hierarchy this editor is. -->
			<nav class="breadcrumb" data-testid="subpatch-breadcrumb" aria-label="Sub-patch path">
				<button class="crumb" onclick={() => exitToDepth(0)} title="Back to the top-level patch"
					>Patch</button
				>
				{#each enteredPath as inst, i (inst)}
					{@const label = g.instances[inst]?.name ?? inst}
					<span class="sep">›</span>
					<button
						class="crumb"
						class:current={i === enteredPath.length - 1}
						onclick={() => exitToDepth(i + 1)}
						title="Go to {label}">{label}</button
					>
				{/each}
			</nav>
		{/if}
		<SvelteFlow
			bind:nodes={flowNodes}
			bind:edges={flowEdges}
			{nodeTypes}
			deleteKey={['Delete', 'Backspace']}
			onconnect={onConnect}
			onreconnect={onReconnect}
			onnodedragstart={onNodeDragStart}
			onnodedrag={onNodeDrag}
			onnodedragstop={onNodeDragStop}
			onpaneclick={onPaneClick}
			onnodeclick={onNodeClick}
			onselectionstart={() => (boxSelecting = true)}
			onselectionend={onSelectionEnd}
			onedgeclick={onEdgeClick}
			ondelete={async ({ nodes, edges }) => {
				// The single delete path (deleteKey wires Delete+Backspace here; the
				// custom keydown handler no longer deletes). Covers app-store AND
				// marquee selection — SvelteFlow filters by each element's `selected`.
				for (const n of nodes) {
					const bnd = parseBoundary(n.id);
					if (bnd && entered) await g.removeBoundary(entered, bnd).catch(() => {});
					else await g.removeNode(n.id).catch(() => {});
				}
				for (const e of edges) {
					// Deleting an In→member / member→Out edge unwires the boundary (the
					// pill survives); a normal flat link is removed.
					const bnd = parseBoundary(e.source) ?? parseBoundary(e.target);
					if (bnd && entered) {
						await g.wireBoundary(entered, bnd, null, null).catch(() => {});
						continue;
					}
					const so = e.sourceHandle;
					const si = e.targetHandle;
					if (so && si)
						await g
							.removeLink({ node_out: e.source, node_in: e.target, slot_out: so, slot_in: si })
							.catch(() => {});
				}
				sel.clear(panelId);
			}}
			fitViewOptions={FIT_OPTIONS}
			minZoom={0.05}
			maxZoom={4}
			initialViewport={{ x: 0, y: 0, zoom: 0.85 }}
			zoomOnDoubleClick={false}
			autoPanOnNodeDrag={false}
		>
			<Controls />
			<FitToGraph options={FIT_OPTIONS} />
			<FlowApi bind:screenToFlowPosition={screenToFlow} />
			<SubpatchZoomExit {entered} onExit={() => exitToDepth(enteredPath.length - 1)} />
			{#if pendingPlacement}
				<PlacementPreview
					typeInfo={pendingPlacement.typeInfo}
					initialClient={pendingPlacement.initialClient}
					targets={snapTargetBounds(new Set())}
					onCommit={(pos) => void commitPlacement(pos)}
					onCancel={() => {
						pendingPlacement = null;
					}}
				/>
			{/if}
			{#if snapGuides.length > 0}
				<ViewportPortal target="front">
					<SnapGuides guides={snapGuides} testid="snap-guides" />
				</ViewportPortal>
			{/if}
		</SvelteFlow>

		{#if flowNodes.length === 0 && !pendingPlacement && !menuOpen}
			<!-- First-run / empty-canvas hint. pointer-events:none so double-click
			     (open add-node menu) and panning still reach the canvas underneath. -->
			<div class="empty-hint" data-testid="empty-hint">
				<div class="eh-title">{entered ? 'This sub-patch is empty' : 'Empty patch'}</div>
				<div class="eh-body">
					Double-click the canvas or press <kbd>+</kbd> to add a node.
					{#if !entered}<br />Load an example from the <strong>Examples ▾</strong> menu to get started.{/if}
				</div>
			</div>
		{/if}

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
					extraTypes={entered ? BOUNDARY_TYPES : []}
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
	/* Nudge SvelteFlow's controls off the panel corner so the corner grips
	   (drag-split / drag-join) stay reachable. */
	.editor-panel :global(.svelte-flow__controls) {
		bottom: 20px;
		left: 20px;
	}
	/* First-run hint over an empty canvas. Non-interactive so it never eats the
	   double-click that opens the add-node menu underneath it. */
	.empty-hint {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 8px;
		text-align: center;
		pointer-events: none;
		color: var(--text-faint);
		z-index: 1;
	}
	.eh-title {
		font-size: 15px;
		font-weight: 600;
		color: var(--text-dim);
	}
	.eh-body {
		font-size: 12px;
		line-height: 1.6;
		max-width: 320px;
	}
	.eh-body kbd {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 1px 5px;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-elev-1);
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
	.breadcrumb {
		position: absolute;
		top: 10px;
		left: 10px;
		z-index: 6;
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 4px 8px;
		max-width: calc(100% - 90px);
		overflow: hidden;
		background: color-mix(in srgb, var(--bg-elev-1) 88%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-1);
		font-family: var(--font-mono);
		font-size: 11px;
	}
	.breadcrumb .crumb {
		background: transparent;
		border: none;
		padding: 1px 4px;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 160px;
	}
	.breadcrumb .crumb:hover {
		color: var(--text);
		background: color-mix(in srgb, var(--accent) 16%, transparent);
	}
	.breadcrumb .crumb.current {
		color: var(--text);
		font-weight: 600;
		cursor: default;
	}
	.breadcrumb .sep {
		color: var(--text-faint);
	}
</style>
