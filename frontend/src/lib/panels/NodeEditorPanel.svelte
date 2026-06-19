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
	import BoundaryNode from '$lib/editor/BoundaryNode.svelte';
	import AddNodeMenu from '$lib/editor/AddNodeMenu.svelte';
	import PlacementPreview from '$lib/editor/PlacementPreview.svelte';
	import FitToGraph from '$lib/editor/FitToGraph.svelte';
	import SubpatchZoomExit from '$lib/editor/SubpatchZoomExit.svelte';
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
	import {
		linkKey,
		BOUNDARY_TYPES,
		boundarySpec,
		type InstanceInfo,
		type LinkInfo,
		type NodeTypeInfo,
		type SubPatchPort
	} from '$lib/api/control';
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

	// --- sub-patch navigation (enter-to-edit) -------------------------------
	// The stack of instance ids the editor has descended into; empty = top level.
	// (A stack so nesting is a natural extension; today one level deep is reached.)
	let enteredPath = $state<string[]>([]);
	const entered = $derived(enteredPath.length ? enteredPath[enteredPath.length - 1] : null);

	const BND_PREFIX = '__bnd__';
	const boundaryId = (instId: string, name: string): string => `${BND_PREFIX}${instId}::${name}`;
	const isBoundaryId = (id: string): boolean => id.startsWith(BND_PREFIX);

	/** The display (graph) name of a member given its local name within `inst`. */
	function memberDisplay(inst: InstanceInfo, local: string): string | null {
		for (const [disp, loc] of Object.entries(inst.members)) if (loc === local) return disp;
		return null;
	}

	function enterInstance(instId: string): void {
		if (!g.instances[instId]) return;
		if (enteredPath[enteredPath.length - 1] === instId) return; // already inside it
		sel.clear(panelId);
		enteredPath = [...enteredPath, instId];
		setTimeout(fitView, 60); // frame the inside once it has rendered
	}

	/** Pop the breadcrumb back to `depth` levels (0 = top of the patch). */
	function exitToDepth(depth: number): void {
		sel.clear(panelId);
		enteredPath = enteredPath.slice(0, depth);
		setTimeout(fitView, 60);
	}

	// If an entered instance is dissolved/removed elsewhere, climb back out of it.
	$effect(() => {
		if (enteredPath.length === 0) return;
		let depth = enteredPath.length;
		while (depth > 0 && !g.instances[enteredPath[depth - 1]]) depth--;
		if (depth !== enteredPath.length) enteredPath = enteredPath.slice(0, depth);
	});

	/** Display name -> the instance that hides it (collapsed sub-patch member). */
	const memberInstance = $derived.by(() => {
		const m = new Map<string, string>();
		for (const [instId, inst] of Object.entries(g.instances)) {
			for (const disp of Object.keys(inst.members)) m.set(disp, instId);
		}
		return m;
	});

	/** Resolve a link endpoint to what's actually drawn: a hidden member is
	 * rerouted to its instance's boundary handle; a normal node is unchanged.
	 * Returns null when the member slot isn't exposed (purely internal). */
	function drawEndpoint(
		node: string,
		slot: string,
		dir: 'in' | 'out'
	): { node: string; handle: string } | null {
		const instId = memberInstance.get(node);
		if (instId === undefined) return { node, handle: slot };
		const inst = g.instances[instId];
		const local = inst.members[node];
		for (const [bname, port] of Object.entries(inst.interface)) {
			if (port.dir === dir && port.inner_node === local && port.inner_slot === slot) {
				return { node: instId, handle: bname };
			}
		}
		return null;
	}

	$effect(() => {
		const inst = entered ? g.instances[entered] : null;
		const next: Node[] = [];
		if (inst && entered) {
			// Inside a sub-patch: render its members (with local labels) plus the
			// In/Out boundary nodes derived from its interface, laid out on either
			// side of the members' bounding box.
			const disps = Object.keys(inst.members);
			const pts = disps
				.map((d) => g.nodeByName(d)?.pos)
				.filter((p): p is [number, number] => !!p);
			const minX = pts.length ? Math.min(...pts.map((p) => p[0])) : 0;
			const maxX = pts.length ? Math.max(...pts.map((p) => p[0])) : 0;
			const minY = pts.length ? Math.min(...pts.map((p) => p[1])) : 0;
			for (const disp of disps) {
				const n = g.nodeByName(disp);
				if (!n) continue;
				next.push({
					id: disp,
					type: 'goofi',
					position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
					data: { node: n, label: inst.members[disp] },
					selected: sel.nodes(panelId).has(disp)
				});
			}
			// In/Out boundary pills (incl. unwired). Use the stored pos; fall back to
			// a beside-the-members layout for legacy entries with no pos.
			const ins = Object.entries(inst.interface).filter(([, p]) => p.dir === 'in');
			const outs = Object.entries(inst.interface).filter(([, p]) => p.dir === 'out');
			const place = (
				entries: [string, SubPatchPort][],
				dir: 'in' | 'out',
				fallbackX: number
			) =>
				entries.forEach(([name, port], i) => {
					next.push({
						id: boundaryId(entered, name),
						type: 'boundary',
						position: port.pos
							? { x: port.pos[0], y: port.pos[1] }
							: { x: fallbackX, y: minY + i * 96 },
						data: { name, dir, dtype: g.boundaryDtype(inst, port), wired: port.inner_node !== null },
						selected: sel.nodes(panelId).has(boundaryId(entered, name))
					});
				});
			place(ins, 'in', minX - 280);
			place(outs, 'out', maxX + 320);
			flowNodes = next;
			return;
		}
		for (const n of g.nodes) {
			if (memberInstance.has(n.name)) continue; // hidden member of a collapsed sub-patch
			next.push({
				id: n.name,
				type: 'goofi',
				position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
				data: { node: n },
				selected: sel.nodes(panelId).has(n.name)
			});
		}
		for (const [instId, inst2] of Object.entries(g.instances)) {
			// A collapsed sub-patch is rendered by the SAME node component as a real
			// node (its wired boundaries are slots, output slots show viewers); the
			// onEnter/onExpand handlers drive the header's sub-patch controls.
			next.push({
				id: instId,
				type: 'goofi',
				position: { x: inst2.pos?.[0] ?? 0, y: inst2.pos?.[1] ?? 0 },
				data: { node: g.nodeByName(instId), onEnter: enterInstance, onExpand: expandInstance },
				selected: sel.nodes(panelId).has(instId)
			});
		}
		flowNodes = next;
	});

	$effect(() => {
		const inst = entered ? g.instances[entered] : null;
		const next: Edge[] = [];
		if (inst && entered) {
			// Internal member↔member links, drawn directly.
			for (const l of g.links) {
				if (!(l.node_out in inst.members) || !(l.node_in in inst.members)) continue;
				const id = linkKey(l);
				next.push({
					id,
					source: l.node_out,
					sourceHandle: l.slot_out,
					target: l.node_in,
					targetHandle: l.slot_in,
					selected: sel.edges(panelId).has(id),
					animated: false
				});
			}
			// Boundary edges: In node → member input, member output → Out node.
			// Only WIRED boundaries draw an edge; deletable so the user can unwire
			// by deleting it (ondelete routes a boundary edge to wireBoundary(null)).
			for (const [name, port] of Object.entries(inst.interface)) {
				if (port.inner_node == null) continue;
				const disp = memberDisplay(inst, port.inner_node);
				if (!disp) continue;
				const bId = boundaryId(entered, name);
				if (port.dir === 'in') {
					const id = `${bId}→${disp}.${port.inner_slot}`;
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
					const id = `${disp}.${port.inner_slot}→${bId}`;
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
			flowEdges = next;
			return;
		}
		for (const l of g.links) {
			const oi = memberInstance.get(l.node_out);
			const ii = memberInstance.get(l.node_in);
			if (oi !== undefined && oi === ii) continue; // internal to one sub-patch — hidden
			const src = drawEndpoint(l.node_out, l.slot_out, 'out');
			const dst = drawEndpoint(l.node_in, l.slot_in, 'in');
			if (!src || !dst) continue; // an endpoint isn't an exposed boundary
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
		const pts = names.map((n) => g.nodeByName(n)?.pos).filter((p): p is [number, number] => !!p);
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

	function expandInstance(instId: string): void {
		void g.expandInstance(instId).catch((e) => console.warn('expand failed', e));
	}

	/** The boundary id (interface key) of a boundary flow-node id, or null. */
	function parseBoundary(id: string): string | null {
		if (!entered || !isBoundaryId(id)) return null;
		return id.slice(`${BND_PREFIX}${entered}::`.length);
	}

	function onConnect(c: Connection): void {
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
		// Inside the entered view, an edge touching an In/Out pill wires that
		// boundary to the member slot on the other end (defines the sub-patch port).
		const srcB = parseBoundary(c.source);
		const dstB = parseBoundary(c.target);
		if (srcB || dstB) {
			if (!entered) return;
			const inst = g.instances[entered];
			const bnd = srcB ?? dstB!;
			const memberId = srcB ? c.target : c.source;
			const memberSlot = srcB ? c.targetHandle : c.sourceHandle;
			const local = inst?.members[memberId];
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

	function onEdgeClick(args: { edge: Edge; event: MouseEvent }): void {
		const e = args.event;
		sel.clickEdge(panelId, args.edge.id, e.shiftKey || e.ctrlKey || e.metaKey);
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
		// Snap only to what's actually on screen in THIS editor (flowNodes) — never
		// to hidden members of a collapsed sub-patch or nodes outside the entered
		// sub-patch (which g.nodes would include).
		const targets: Bounds[] = [];
		for (const n of flowNodes) {
			if (current.has(n.id)) continue;
			targets.push(nodeBoundsFromFlow(n.id, n.position.x, n.position.y));
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
			for (const n of args.nodes) {
				const pos: [number, number] = [Math.round(n.position.x + dx), Math.round(n.position.y + dy)];
				const bnd = parseBoundary(n.id);
				if (bnd && entered) {
					void g.setBoundaryPos(entered, bnd, pos).catch(() => {});
				} else {
					void g.setNodePos(n.id, pos);
				}
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
			const names = entered
				? Object.keys(g.instances[entered]?.members ?? {})
				: [
						...g.nodes.filter((n) => !memberInstance.has(n.name)).map((n) => n.name),
						...Object.keys(g.instances)
					];
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
		const selNodes = g.nodes.filter((n) => names.has(n.name));
		if (selNodes.length === 0) return;
		const links = g.links.filter((l) => names.has(l.node_in) && names.has(l.node_out));
		if (!(await copyText(JSON.stringify(serializeClipboard(selNodes, links))))) {
			console.warn('clipboard write failed');
		}
	}

	async function duplicateSelection(): Promise<void> {
		const rename = await g.cloneNodes(selectedNodeNames(), [40, 40]);
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
		// An In/Out boundary pseudo-type adds a virtual boundary to the entered
		// sub-patch rather than spawning a real node.
		const bspec = boundarySpec(placement.typeInfo.type);
		if (bspec && placement.typeInfo.category === 'boundary') {
			if (!entered) return;
			try {
				const bndId = await g.addBoundary(entered, bspec.dir, bspec.dtype, pos);
				// Seeded from a member slot click → wire the new boundary straight to
				// that slot, so In/Out behave like any other auto-connected node. The
				// seed's node is the member display name; wire_boundary wants its local.
				if (bndId && placement.seed) {
					const local = g.instances[entered]?.members[placement.seed.node];
					if (local) await g.wireBoundary(entered, bndId, local, placement.seed.slot);
				}
			} catch (e) {
				console.warn('add boundary failed', e);
			}
			return;
		}
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
					<span class="sep">›</span>
					<button
						class="crumb"
						class:current={i === enteredPath.length - 1}
						onclick={() => exitToDepth(i + 1)}
						title="Go to {inst}">{inst}</button
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
			onnodedragstart={onNodeDragStart}
			onnodedrag={onNodeDrag}
			onnodedragstop={onNodeDragStop}
			onpaneclick={onPaneClick}
			onnodeclick={onNodeClick}
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
			<MiniMap pannable zoomable />
			<FitToGraph options={FIT_OPTIONS} />
			<SubpatchZoomExit {entered} onExit={() => exitToDepth(enteredPath.length - 1)} />
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
