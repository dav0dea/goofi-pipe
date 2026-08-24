<!-- Node-editor panel — the SvelteFlow graph and its interaction logic. Each instance owns its own
     viewport but reads the shared graph and selection stores. Editor-scoped keyboard shortcuts are
     gated on this being the active panel; app-global ones live in AppShell. -->
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
	import AddNodeMenu from '$lib/editor/AddNodeMenu.svelte';
	import PlacementPreview from '$lib/editor/PlacementPreview.svelte';
	import FitToGraph from '$lib/editor/FitToGraph.svelte';
	import { camera } from '$lib/editor/camera';
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
	import { notify } from '$lib/stores/notify.svelte';
	import { ui, slotKey, type SlotClickSeed } from '$lib/stores/ui.svelte';
	import { selection } from '$lib/stores/selection.svelte';
	import { workspace } from 'panelty';
	import { getPanelType, type PanelProps } from 'panelty';
	import { portal } from 'panelty';
	import {
		linkKey,
		type NodeInstanceInfo,
		type NodeTypeInfo
	} from '$lib/api/control';
	import { ROOT_ID, childrenOfScope, drawEndpoint as sceneDrawEndpoint } from '$lib/editor/subpatchScene';
	import { nodeSurfaceSize, inputUnits } from '$lib/editor/nodeMetrics';
	import { isSlotExpanded } from '$lib/viewers/inlineView';
	import {
		inputAnchors,
		nearSlots,
		sameKeys,
		SLOT_PROXIMITY_PX,
		type SlotAnchor
	} from '$lib/editor/slotProximity';
	import { createLongPress } from 'panelty';
	import { createDoubleTapZoom, zoomStep, type FlowViewport } from '$lib/editor/doubleTapZoom';
	import { eventPoint } from '$lib/editor/eventPoint';
	import { serializeClipboard, parseClipboard, fragmentCentre } from '$lib/editor/clipboard';
	import { copyText } from '$lib/clipboard';
	import { registerEditor, unregisterEditor } from './editorCommands';
	import InspectorOverlay from './InspectorOverlay.svelte';
	import { arrayToPath, asStateObject, pathToArray } from 'panelty';
	import { Button, IconButton, EmptyState, isTextEditingTarget } from '$lib/ui';
	import { clampToViewport, overlayViewport } from 'panelty';
	import { onMount, tick, untrack } from 'svelte';

	let { panelId, state: panelState, setState }: PanelProps = $props();

	function samePath(a: string[], b: string[]): boolean {
		return a.length === b.length && a.every((v, i) => v === b[i]);
	}

	const g = graph();
	const uiStore = ui();
	const sel = selection();
	const ws = workspace();

	/** Only the active panel reacts to keyboard shortcuts and pending slot clicks. */
	const isActive = (): boolean => ws.activePanelId === panelId;

	const selectedNode = $derived(sel.selectedNode(panelId));

	// The standing ◧ preference MINUS a live ✕ dismissal, which holds only until the selection
	// next changes.
	const inspectorOn = $derived(sel.inspectorVisibleFor(panelId));

	$effect(() => {
		if (ws.activePanelId === panelId) sel.setActiveEditor(panelId);
	});

	let rootEl = $state<HTMLDivElement | null>(null);

	/** Which edge of the menu's own box the requested open point names. */
	type MenuAlign = 'start' | 'center' | 'end';

	let menuOpen = $state(false);
	// The REQUESTED spawn point and the RENDERED one are separate, so the placement effect never
	// writes its own dependency.
	let menuAt = $state<{ x: number; y: number; align: MenuAlign }>({ x: 120, y: 120, align: 'start' });
	let menuPos = $state<{ x: number; y: number }>({ x: 120, y: 120 });
	let menuSeed = $state<SlotClickSeed | null>(null);
	let menuEl = $state<HTMLDivElement | null>(null);
	// A long press opens the menu with the finger still down, so the touchend arrives as a compat
	// click. At the WINDOW in capture, because which layer that click lands on is not knowable.
	let swallowMenuClick = $state(false);
	$effect(() => {
		if (!swallowMenuClick) return;
		const eat = (e: MouseEvent): void => {
			e.stopPropagation();
			swallowMenuClick = false;
		};
		const opts = { capture: true, once: true } as const;
		window.addEventListener('click', eat, opts);
		return () => window.removeEventListener('click', eat, opts);
	});

	/** Open the add-node menu at a viewport point — the one placement path for all four entry
	 * points. The point names the menu's left, centre or right edge; the effect below clamps it. */
	function openAddMenu(
		x: number,
		y: number,
		align: MenuAlign = 'start',
		seed: SlotClickSeed | null = null,
		swallowNextDismiss = false
	): void {
		menuAt = { x, y, align };
		menuPos = { x, y }; // corrected below before paint
		menuSeed = seed;
		swallowMenuClick = swallowNextDismiss;
		menuOpen = true;
	}

	/** The coarse-pointer door onto the add-node menu. Armed for `touch` alone: a held mouse button
	 * is the start of a desktop pan. */
	const canvasPress = createLongPress((at) =>
		openAddMenu(at.clientX - 8, at.clientY + 8, 'start', null, true)
	);

	/** Empty canvas only, and not while a ghost is pending: the canvas belongs to that placement. */
	function onCanvasPointerDown(e: PointerEvent): void {
		if (e.pointerType !== 'touch' || pendingPlacement) return;
		if (!onBareCanvas(e.target)) return;
		canvasPress.start(e);
	}

	const onBareCanvas = (target: EventTarget | null): boolean =>
		Boolean((target as HTMLElement | null)?.classList.contains('svelte-flow__pane'));

	// Double-tap-and-drag zoom, beside pinch; the seam is `zoomOnDoubleClick={false}` below.
	const tapZoom = createDoubleTapZoom();
	// Sampled ONCE at the start, so the drag cannot accumulate rounding.
	let zoomFrom: FlowViewport | null = null;
	let zoomAnchor: { x: number; y: number } | null = null;

	/** The pan block. On `touchstart`, not `pointerdown`: SvelteFlow pans by d3-zoom, which binds
	 * `touchstart` on its own pane wrapper, so only a capture listener above it can stop the pan. */
	function onCanvasTouchStart(e: TouchEvent): void {
		// A second finger is a PINCH: hand the whole gesture back rather than compete with it.
		if (pendingPlacement || e.touches.length > 1 || !onBareCanvas(e.target)) {
			tapZoom.cancel();
			return;
		}
		const p = eventPoint(e);
		if (!p || !tapZoom.down(p, e.timeStamp)) return;

		// A double tap held still would otherwise also fire the long-press door on top of the zoom.
		canvasPress.cancel();
		zoomFrom = getViewport?.() ?? null;
		zoomAnchor = screenToFlow?.({ x: p.clientX, y: p.clientY }) ?? null;
		e.stopPropagation();
		e.preventDefault();
	}

	function onCanvasTouchMove(e: TouchEvent): void {
		const p = eventPoint(e);
		if (!p) return;
		const factor = tapZoom.move(p);
		if (factor === null || !zoomFrom || !zoomAnchor) return;
		e.stopPropagation();
		e.preventDefault();
		setViewport?.(zoomStep(zoomFrom, zoomAnchor, factor, { min: MIN_ZOOM, max: MAX_ZOOM }));
	}

	function onCanvasTouchEnd(e: TouchEvent): void {
		const p = eventPoint(e);
		if (p) tapZoom.up(p, e.timeStamp);
		zoomFrom = null;
		zoomAnchor = null;
	}

	// Measure the mounted menu, then re-clamp: a spawn point is a degenerate anchor rect.
	$effect(() => {
		const el = menuEl;
		if (!el) return;
		const at = menuAt;
		const place = (): void => {
			const r = el.getBoundingClientRect();
			const left =
				at.align === 'center' ? at.x - r.width / 2 : at.align === 'end' ? at.x - r.width : at.x;
			// `overlayViewport()`, not `window.innerHeight`: this menu focuses its search on open, so
			// the soft keyboard is on its way up as it lands and the layout viewport does not shrink.
			const p = clampToViewport(
				{ left, top: at.y, right: left, bottom: at.y, width: 0, height: 0 },
				{ width: r.width, height: r.height },
				overlayViewport()
			);
			menuPos = { x: p.left, y: p.top };
		};
		place();
		const vv = window.visualViewport;
		vv?.addEventListener('resize', place);
		return () => vv?.removeEventListener('resize', place);
	});

	// A clicked port opens the menu beside it, seeded with the dtype filter + auto-link.
	$effect(() => {
		const seed = uiStore.pendingSlotClick;
		if (!seed) return;
		if (!isActive()) return;
		uiStore.consumeSlotClick();
		const source = seed.side === 'source';
		openAddMenu(
			seed.clientX + (source ? 12 : -12),
			seed.clientY - 24,
			source ? 'start' : 'end',
			seed
		);
	});

	let snapGuides = $state<Guide[]>([]);
	let pendingPlacement = $state<{
		typeInfo: NodeTypeInfo;
		seed: SlotClickSeed | null;
		initialClient: { x: number; y: number };
	} | null>(null);

	let flowNodes = $state.raw<Node[]>([]);
	let flowEdges = $state.raw<Edge[]>([]);
	// Bumped to force a flowEdges re-derive: SvelteFlow inserts a dropped connection optimistically
	// before `onConnect` runs, and a REJECTED wire produces no doc echo to rebuild from.
	let reconcileTick = $state(0);

	// The stack of instance ids this editor has descended into; empty = top level.
	let enteredPath = $state<string[]>(untrack(() => pathToArray(asStateObject(panelState).subpatchPath)));
	const entered = $derived(enteredPath.length ? enteredPath[enteredPath.length - 1] : null);

	/** Write the current path back into the panel's state. Classified as NAVIGATION: descending into
	 * a sub-patch is looking, not editing, so it must not mark the patch unsaved. */
	function persistEnteredPath(): void {
		untrack(() => {
			const path = arrayToPath(enteredPath);
			if (asStateObject(panelState).subpatchPath === path) return;
			setState({ ...asStateObject(panelState), subpatchPath: path }, 'navigation');
		});
	}

	// Follow an EXTERNAL path change — a patch load that reuses this panel id, so the component is
	// not remounted. Our own writes leave the two equal.
	$effect(() => {
		const persisted = pathToArray(asStateObject(panelState).subpatchPath);
		if (!samePath(persisted, untrack(() => enteredPath))) enteredPath = persisted;
	});

	/** uid → the scope it is drawn in. Membership rides the record, so this is a read, not a walk. */
	const memberIndex = $derived(new Map(g.nodes.map((n) => [n.uid, n.scope])));

	/** Is this uid a sub-patch facade — the one thing this gesture can ENTER? */
	function isScope(uid: string): boolean {
		return !!g.nodeById(uid)?.subpatch;
	}

	function enterInstance(instId: string): void {
		if (!isScope(instId)) return;
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
		while (depth > 0 && !isScope(enteredPath[depth - 1])) depth--;
		if (depth !== enteredPath.length) {
			enteredPath = enteredPath.slice(0, depth);
			persistEnteredPath();
		}
	});

	/** Resolve a link endpoint to what is actually drawn in the entered scope: the facade of the
	 * nearest enclosing scope, or null when the endpoint lies outside the entered subtree. */
	function drawEndpoint(node: string, slot: string): { node: string; handle: string } | null {
		// The scene algebra always takes a real scope id; `entered` stays null at root, which is
		// what the "are we inside a sub-patch?" decisions read.
		return sceneDrawEndpoint(node, slot, entered ?? ROOT_ID, memberIndex);
	}

	// Render the direct children of the entered scope — leaves, nested facades and boundary ports
	// alike, because each is a node record that names this scope.
	$effect(() => {
		const scope = entered ?? ROOT_ID;
		const next: Node[] = [];
		for (const uid of childrenOfScope(scope, memberIndex)) {
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
		flowNodes = next;
	});

	// Every link, rerouted to the nearest VISIBLE boundary port in the entered scope. A port's own
	// inner wire is one of them: it is a link like any other, drawn where its port is drawn.
	$effect(() => {
		reconcileTick; // re-derive on demand to drop an optimistic ghost edge after a rejected wire
		const next: Edge[] = [];
		for (const l of g.links) {
			const src = drawEndpoint(l.node_out, l.slot_out);
			const dst = drawEndpoint(l.node_in, l.slot_in);
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
		flowEdges = next;
	});

	/** Node ids selected in THIS editor — the store's selection plus a live marquee. */
	function selectedUids(): string[] {
		const ids = new Set<string>(sel.nodes(panelId));
		for (const n of flowNodes) if (n.selected) ids.add(n.id);
		return [...ids];
	}

	async function groupSelection(): Promise<void> {
		const names = selectedUids();
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

	function onConnect(c: Connection): void {
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
		// Every cable is one `add_link`, the pill's included: a port is a node to the op vocabulary,
		// and a top-level wire to a collapsed facade is spliced to the inner leaf by the bridge.
		void g
			.addLink({
				node_out: c.source,
				node_in: c.target,
				slot_out: c.sourceHandle,
				slot_in: c.targetHandle
			})
			.catch((e) => {
				notify().failure('Connect', e);
				reconcileTick++;
			});
	}

	/** Drag an existing edge's endpoint to a new slot. Remove + add are one history entry, so a
	 * single undo reverts the move. */
	function onReconnect(oldEdge: Edge, c: Connection): void {
		const oldSo = oldEdge.sourceHandle;
		const oldSi = oldEdge.targetHandle;
		if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle || !oldSo || !oldSi) return;
		void history()
			.transaction('Reconnect link', async () => {
				await g.removeLink({ node_out: oldEdge.source, node_in: oldEdge.target, slot_out: oldSo, slot_in: oldSi });
				await g.addLink({
					node_out: c.source as string,
					node_in: c.target as string,
					slot_out: c.sourceHandle as string,
					slot_in: c.targetHandle as string
				});
			})
			// `transaction` re-throws, so a refused move needs this catch: the rebuild puts every
			// cable back where `g.links` says it is.
			.catch((e) => {
				notify().failure('Reconnect', e);
				reconcileTick++;
			});
	}

	function onEdgeClick(args: { edge: Edge; event: MouseEvent }): void {
		const e = args.event;
		sel.clickEdge(panelId, args.edge.id, e.shiftKey || e.ctrlKey || e.metaKey);
	}

	// Input names, revealed by proximity while a cable is in flight. The anchors are snapshotted
	// ONCE per drag in FLOW space, so a canvas that pans or zooms mid-drag needs no invalidation.
	let cableAnchors: SlotAnchor[] = [];
	let cableNear: ReadonlySet<string> = new Set();

	function publishCableNear(next: ReadonlySet<string>): void {
		if (sameKeys(next, cableNear)) return; // don't invalidate every node for an unchanged set
		cableNear = next;
		uiStore.setCableNear(next);
	}

	function onCableMove(e: PointerEvent): void {
		const toFlow = screenToFlow;
		if (!toFlow || cableAnchors.length === 0) return;
		const zoom = getViewport?.().zoom ?? 1;
		// The radius is a SCREEN distance, so it is converted into flow space, not the anchors out.
		publishCableNear(
			nearSlots(cableAnchors, toFlow({ x: e.clientX, y: e.clientY }), SLOT_PROXIMITY_PX / zoom)
		);
	}

	function onCableStart(): void {
		cableAnchors = inputAnchors(
			flowNodes.flatMap((f) => {
				// Boundary pills carry no name tag; only real nodes have a `.conn-label` to reveal.
				const n = f.type === 'goofi' ? (f.data?.node as NodeInstanceInfo | undefined) : undefined;
				if (!n) return [];
				const multi = new Set(n.input_multi ?? []);
				return [{ uid: f.id, x: f.position.x, y: f.position.y, slots: Object.keys(n.input_slots ?? {}), multi }];
			}),
			slotKey
		);
		publishCableNear(new Set());
		// On `window`, not the panel: a cable dragged past the panel's edge is still in flight.
		window.addEventListener('pointermove', onCableMove);
	}

	function onCableEnd(): void {
		window.removeEventListener('pointermove', onCableMove);
		cableAnchors = [];
		publishCableNear(new Set());
	}

	/** A node's snap footprint when Svelte Flow has not measured it yet. */
	function nodeFallbackSize(flowNode: Node | undefined): { width: number; height: number } {
		const node = flowNode?.data?.node as NodeInstanceInfo | undefined;
		if (node) {
			const inputs = Object.keys(node.input_slots ?? {});
			const outputs = Object.keys(node.output_slots ?? {});
			const multi = new Set(node.input_multi ?? []);
			return nodeSurfaceSize(
				inputUnits(inputs, (s) => multi.has(s)),
				outputs.map((s) => isSlotExpanded(node, s))
			);
		}
		return { width: DEFAULT_NODE_W, height: DEFAULT_NODE_H };
	}

	function nodeBoundsFromFlow(id: string, x: number, y: number): Bounds {
		const flowNode = flowNodes.find((n) => n.id === id);
		let w = flowNode?.measured?.width;
		let h = flowNode?.measured?.height;
		if (w == null || h == null) {
			const fb = nodeFallbackSize(flowNode);
			w ??= fb.width;
			h ??= fb.height;
		}
		return makeBounds(x, y, w, h);
	}

	/** Snap-target bounds for every node on screen in THIS editor, shared by the node drag and the
	 * placement preview. Only `flowNodes` — `g.nodes` would include what this scope does not draw. */
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

	// Positions at drag start, so a node snaps back when the drag turns into a panel-link.
	let dragOrigin = new Map<string, { x: number; y: number }>();
	// The chip that follows the cursor while a drag is a reference; null = a reposition drag.
	let linkGhost = $state<{ x: number; y: number; name: string } | null>(null);

	/** The leaf panel under a screen point. Geometric, not `elementFromPoint`: the dragged node sits
	 * under the cursor and would mask the panel beneath it. */
	function panelUnder(x: number, y: number): { id: string; type: string } | null {
		for (const el of document.querySelectorAll<HTMLElement>('[data-panel-id]')) {
			const r = el.getBoundingClientRect();
			if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) {
				return { id: el.dataset.panelId ?? '', type: el.dataset.panelType ?? '' };
			}
		}
		return null;
	}

	/** A node-accepting panel under the cursor, other than this editor. */
	function linkTargetAt(event: MouseEvent | TouchEvent): { id: string; type: string } | null {
		const p = eventPoint(event);
		if (!p) return null;
		const t = panelUnder(p.clientX, p.clientY);
		return t && t.id !== panelId && getPanelType(t.type)?.acceptsNode === true ? t : null;
	}

	/** Put the dragged nodes back where the drag started. */
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
		uiStore.nodeDrag = args.nodes[0]?.id ?? null;
	}

	function onNodeDrag(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		const dragged = new Set(args.nodes.map((n) => n.id));
		const target = linkTargetAt(args.event);
		if (target) {
			// A reference drag, not a coordinate move: the node snaps back and a ghost follows.
			uiStore.nodeDragTarget = target.id;
			// `eventPoint`, because a TouchEvent carries no `clientX` of its own.
			const p = eventPoint(args.event) ?? { clientX: 0, clientY: 0 };
			linkGhost = { x: p.clientX, y: p.clientY, name: g.nodeById(args.nodes[0]?.id ?? '')?.name ?? '' };
			snapGuides = [];
			revertDragged(dragged);
			return;
		}
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
			// One transaction, so moving N nodes is a single undo. Each set*Pos records AFTER its RPC
			// resolves, so the calls must be AWAITED inside it or the buffer is empty at flush.
			const label = args.nodes.length > 1 ? `Move ${args.nodes.length} nodes` : 'Move node';
			void history().transaction(label, async () => {
				for (const n of args.nodes) {
					const pos: [number, number] = [Math.round(n.position.x + dx), Math.round(n.position.y + dy)];
					await g.setNodePos(n.id, pos);
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
			openAddMenu(here.x - 8, here.y + 8);
			lastPaneClickAt = 0;
			return;
		}
		lastPaneClickAt = now;
		lastPaneClickPos = here;
		menuOpen = false;
		sel.clickPane(panelId, args.event.shiftKey);
		// SvelteFlow calls `unselectNodesAndEdges()` immediately AFTER this callback, whatever the
		// store decided, so wherever the store KEEPS the selection it must be re-derived after it.
		if (sel.nodes(panelId).size || sel.edges(panelId).size) void tick().then(reassertSelection);
	}

	/** Push the store's selection back onto the rendered `selected` flags. */
	function reassertSelection(): void {
		flowNodes = flowNodes.map((n) => ({ ...n, selected: sel.nodes(panelId).has(n.id) }));
		flowEdges = flowEdges.map((e) => ({ ...e, selected: sel.edges(panelId).has(e.id) }));
	}

	function onNodeClick(args: { node: Node; event: MouseEvent | TouchEvent }): void {
		// A click can land between a graph mutation and the flowNodes rebuild, carrying an id that
		// no longer exists.
		if (!flowNodes.some((n) => n.id === args.node.id)) return;
		const mouse = args.event as MouseEvent;
		sel.clickNode(panelId, args.node.id, mouse.shiftKey || mouse.ctrlKey || mouse.metaKey);
	}

	function sameMembers(set: Set<string>, ids: string[]): boolean {
		if (set.size !== ids.length) return false;
		for (const id of ids) if (!set.has(id)) return false;
		return true;
	}

	// True only between a box/marquee drag's start and end, so a plain pane click's end event
	// cannot resurrect a just-cleared selection.
	let boxSelecting = false;

	/** Mirror a finished marquee into the store. Keyed on start/end, never `onselectionchange`: a
	 * store-driven selection replaces every flowNodes object and Flow then emits transient echoes. */
	function onSelectionEnd(): void {
		if (!boxSelecting) return;
		boxSelecting = false;
		const nodeIds = flowNodes.filter((n) => n.selected).map((n) => n.id);
		const edgeIds = flowEdges.filter((e) => e.selected).map((e) => e.id);
		if (sameMembers(sel.nodes(panelId), nodeIds) && sameMembers(sel.edges(panelId), edgeIds)) return;
		sel.setSelection(panelId, nodeIds, edgeIds);
	}

	// Double-click to enter a sub-patch, detected here because `onnodeclick` suppresses the 2nd
	// click. In CAPTURE and CONSUMED: the inspector slides over the node the 2nd click must hit.
	const DBL_PX = 6; // a real double-click barely moves the pointer…
	const DBL_PX_TOUCH = 16; // …but a finger does, and 6px is well under any tap slop
	let lastClickInst = '';
	let lastClickAt = 0;
	let lastClickX = 0;
	let lastClickY = 0;
	/** The node a click landed on, or ''. */
	function nodeUnder(target: EventTarget | null): string {
		return (
			(target as HTMLElement | null)?.closest?.('.svelte-flow__node')?.getAttribute('data-id') ?? ''
		);
	}
	function onCanvasClick(event: MouseEvent): void {
		const now = performance.now();
		const hereNode = nodeUnder(event.target);
		// …of which only a sub-patch instance is something this gesture can ENTER.
		const here = isScope(hereNode) ? hereNode : '';
		// Per gesture, not per device.
		const slop = (event as PointerEvent).pointerType === 'touch' ? DBL_PX_TOUCH : DBL_PX;
		if (
			lastClickInst &&
			now - lastClickAt < DOUBLE_CLICK_MS &&
			Math.abs(event.clientX - lastClickX) < slop &&
			Math.abs(event.clientY - lastClickY) < slop &&
			// A second click resolving to a DIFFERENT NODE is that node's first click. Asked of the
			// NODE, not the instance: '' is reserved for the inspector having slid over it.
			(hereNode === '' || hereNode === lastClickInst)
		) {
			const inst = lastClickInst;
			lastClickInst = '';
			// The gesture owns this click; `preventDefault` also covers a checkbox's activation
			// behaviour, which propagation alone does not stop.
			event.stopPropagation();
			event.preventDefault();
			enterInstance(inst);
			return;
		}
		lastClickAt = now;
		lastClickX = event.clientX;
		lastClickY = event.clientY;
		lastClickInst = here;
	}

	const nodeTypes = { goofi: GoofiNode };

	/** Framing for every programmatic fit. */
	const FIT_OPTIONS = { maxZoom: 1, padding: 0.18 } satisfies FitViewOptions;

	/** How far the canvas may be zoomed; the double-tap zoom clamps itself to the same pair. */
	const MIN_ZOOM = 0.05;
	const MAX_ZOOM = 4;

	/** True when the CANVAS owns the keyboard rather than a control. Deliberately NOT a focusable
	 * DESCENDANT of a node: Tab is how a keyboard reaches the next slot pill inside one. */
	function canvasHasKeys(target: HTMLElement | null): boolean {
		if (!target) return false;
		return (
			target === document.body ||
			target.classList.contains('svelte-flow__pane') ||
			target.classList.contains('svelte-flow__node')
		);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (!isActive()) return;
		const t = e.target as HTMLElement | null;
		// The DOM says a modal owns the keyboard, NOT `ui().modalOpen` — that ref-count is also
		// raised by a merely expanded in-panel textarea.
		if (t?.closest?.('dialog[open]')) return;
		if (isTextEditingTarget(t)) return;

		const meta = e.ctrlKey || e.metaKey;
		if (meta && e.key.toLowerCase() === 'a') {
			e.preventDefault();
			selectAll();
		} else if (meta && e.key.toLowerCase() === 'c') {
			// Stop the browser's own copy: it would put the (empty) DOM selection on the clipboard
			// over the payload. Ctrl+V is deliberately NOT here — the `paste` event is that door.
			e.preventDefault();
			void copySelection();
		} else if (meta && e.key.toLowerCase() === 'x') {
			e.preventDefault();
			void cutSelection();
		} else if (meta && e.key.toLowerCase() === 'd') {
			e.preventDefault();
			void duplicateSelection();
		} else if (meta && e.key.toLowerCase() === 'g') {
			e.preventDefault();
			void groupSelection();
		} else if (e.key === 'Tab' && !e.shiftKey && canvasHasKeys(t)) {
			// Scoped to the bare canvas, or nothing outside it is ever Tab-reachable (WCAG 2.1.2).
			// Shift+Tab is left alone: it is the way back OUT of a canvas nothing has focused yet.
			e.preventDefault();
			openAddMenu(mouseX, mouseY);
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

	/** The single delete path, for SvelteFlow's `ondelete` and the app header's Delete row. Nodes go
	 * as ONE batch so undo restores them all BEFORE their links — and a boundary port is one of
	 * them, because `remove_node` and `remove_link` both answer for a port. */
	async function deleteElements({ nodes, edges }: { nodes: Node[]; edges: Edge[] }): Promise<void> {
		const nodeIds = nodes.map((n) => n.id);
		const deleted = new Set(nodeIds);
		await g.removeNodes(nodeIds).catch(() => {});
		for (const e of edges) {
			// A link touching a batch-deleted node went with it; don't double-record its removal.
			if (deleted.has(e.source) || deleted.has(e.target)) continue;
			const so = e.sourceHandle;
			const si = e.targetHandle;
			if (so && si)
				await g
					.removeLink({ node_out: e.source, node_in: e.target, slot_out: so, slot_in: si })
					.catch(() => {});
		}
		sel.clear(panelId);
	}

	/** The rendered `selected` flags: the store's selection unioned with a live marquee. */
	function selectedElements(): { nodes: Node[]; edges: Edge[] } {
		return { nodes: flowNodes.filter((n) => n.selected), edges: flowEdges.filter((e) => e.selected) };
	}

	function hasSelection(): boolean {
		const { nodes, edges } = selectedElements();
		return nodes.length > 0 || edges.length > 0;
	}

	function deleteSelection(): void {
		if (hasSelection()) void deleteElements(selectedElements());
	}

	/** Select what is on screen: the entered scope's members, group nodes included. */
	function selectAll(): void {
		sel.selectNodes(panelId, childrenOfScope(entered ?? ROOT_ID, memberIndex));
	}

	/** Put the selection on the clipboard, answering what was put there. The manager reads the
	 * SUBTREE, so a sub-patch's members, ports and nested scopes come with it — which is what makes
	 * the payload paste-able into a patch that never held those uids. */
	async function copySelection(): Promise<string[]> {
		const uids = selectedUids();
		if (uids.length === 0) return [];
		if (!(await copyText(JSON.stringify(serializeClipboard(await g.copyNodes(uids)))))) {
			notify().failure('Copy', 'the clipboard refused the payload');
			return [];
		}
		return uids;
	}

	/** Cut: the copy, then the delete, as ONE history entry — so one undo puts the nodes back and
	 * the clipboard still holds them. The delete waits on the copy, or a failed write would take
	 * the nodes with it. */
	async function cutSelection(): Promise<void> {
		const uids = await copySelection();
		if (uids.length === 0) return;
		await history()
			.transaction('Cut nodes', () => g.removeNodes(uids))
			.catch((e) => notify().failure('Cut', e));
		sel.clear(panelId);
	}

	async function duplicateSelection(): Promise<void> {
		const rename = await history().transaction('Duplicate nodes', () =>
			g.cloneNodes(selectedUids(), [40, 40], entered ?? undefined)
		);
		const created = Object.values(rename);
		if (created.length > 0) sel.selectNodes(panelId, created);
	}


	/** Paste what the platform clipboard holds. The `paste` EVENT is the door that always works —
	 * `navigator.clipboard.readText` needs a secure context, and goofi is served over plain http on
	 * a LAN — so a menu item, which has no event to read, asks for it and says so when refused. */
	async function pasteClipboard(): Promise<void> {
		try {
			await pasteText(await navigator.clipboard.readText());
		} catch {
			notify().failure('Paste', 'this browser only pastes with the keyboard here — press Ctrl+V');
		}
	}

	async function pasteText(text: string): Promise<void> {
		const clip = parseClipboard(text);
		if (!clip) return;
		// Anchor the paste at the visible viewport centre, in FLOW space. The fragment carries the
		// positions it was copied at, so what goes to the manager is the SHIFT between the two.
		const rect = rootEl?.getBoundingClientRect();
		let at: [number, number] = [window.innerWidth / 4, window.innerHeight / 4];
		if (rect && screenToFlow) {
			const c = screenToFlow({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
			at = [c.x, c.y];
		}
		const from = fragmentCentre(clip.doc);
		const rename = await history().transaction('Paste nodes', () =>
			g.pasteNodes(clip.doc, [Math.round(at[0] - from[0]), Math.round(at[1] - from[1])], entered ?? undefined)
		);
		const created = Object.values(rename);
		if (created.length > 0) sel.selectNodes(panelId, created);
	}

	/** Open the add-node menu centered over this panel, for callers that name a panel not a point. */
	function openAddMenuCentered(): void {
		const r = rootEl?.getBoundingClientRect();
		if (r) openAddMenu(r.left + r.width / 2, r.top + 60, 'center');
		else openAddMenu(window.innerWidth / 2, 80, 'center');
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
		// Inputs take a single source, so an existing cable is replaced; outputs fan out.
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
		// A boundary type takes this same path: `add_node` with `inst_id` is what makes a PORT of the
		// entered sub-patch, and the catalog gives it the slots `autoLink` matches against.
		const label = placement.seed
			? `Add ${placement.typeInfo.type} + connect`
			: `Add ${placement.typeInfo.type}`;
		await history().transaction(label, async () => {
			try {
				const newName = await g.addNode(
					placement.typeInfo.type,
					placement.typeInfo.category,
					pos,
					entered ?? undefined
				);
				// Safe before `node_added` lands: flowNodes derives `selected` from this set.
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

	// Bound from <FlowApi> inside <SvelteFlow>.
	let screenToFlow = $state<((p: { x: number; y: number }) => { x: number; y: number }) | undefined>(
		undefined
	);
	let getViewport = $state<(() => FlowViewport) | undefined>(undefined);
	let setViewport = $state<((v: FlowViewport) => void) | undefined>(undefined);

	// A layout reshape or a page switch DESTROYS this component, so the camera outliving it is what
	// carries the framing across.
	const cam = camera(untrack(() => panelId));
	let viewport = $state<FlowViewport>(cam.viewport ?? { x: 0, y: 0, zoom: 0.85 });
	$effect(() => {
		cam.viewport = viewport;
	});

	function fitView(): void {
		rootEl?.querySelector<HTMLButtonElement>('.svelte-flow__controls-fitview')?.click();
	}

	/** Select a node in this editor — the shared handle for focusing one from elsewhere. */
	function focusNode(uid: string): void {
		sel.selectNodes(panelId, [uid]);
	}

	/** The platform's own paste, which carries the text with it — so it needs no permission and no
	 * secure context. Guarded exactly as the key handler is: a paste into a text field is that
	 * field's, and a paste into a panel that is not active is not this editor's. */
	function onPaste(e: ClipboardEvent): void {
		if (!isActive()) return;
		const t = e.target as HTMLElement | null;
		if (t?.closest?.('dialog[open]') || isTextEditingTarget(t)) return;
		const text = e.clipboardData?.getData('text') ?? '';
		if (!parseClipboard(text)) return; // not ours: leave it for whatever else is listening
		e.preventDefault();
		void pasteText(text);
	}

	onMount(() => {
		registerEditor(panelId, {
			openAddMenu: openAddMenuCentered,
			fitView,
			focusNode,
			selectAll,
			clearSelection: () => sel.clear(panelId),
			deleteSelection,
			groupSelection: () => void groupSelection(),
			copySelection: () => void copySelection(),
			cutSelection: () => void cutSelection(),
			pasteClipboard: () => void pasteClipboard(),
			duplicateSelection: () => void duplicateSelection(),
			hasSelection
		});
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('paste', onPaste);
		window.addEventListener('mousemove', trackMouse);
		rootEl?.addEventListener('click', onCanvasClick, true);
		rootEl?.addEventListener('pointerdown', onCanvasPointerDown);
		rootEl?.addEventListener('pointermove', canvasPress.move);
		rootEl?.addEventListener('pointerup', canvasPress.cancel);
		rootEl?.addEventListener('pointercancel', canvasPress.cancel);
		// CAPTURE, so these run before d3-zoom's own listeners; `passive: false` so the
		// `preventDefault` above is honoured.
		const touchOpts = { capture: true, passive: false } as const;
		rootEl?.addEventListener('touchstart', onCanvasTouchStart, touchOpts);
		rootEl?.addEventListener('touchmove', onCanvasTouchMove, touchOpts);
		rootEl?.addEventListener('touchend', onCanvasTouchEnd, touchOpts);
		rootEl?.addEventListener('touchcancel', tapZoom.cancel, true);
		return () => {
			unregisterEditor(panelId);
			rootEl?.removeEventListener('click', onCanvasClick, true);
			rootEl?.removeEventListener('pointerdown', onCanvasPointerDown);
			rootEl?.removeEventListener('pointermove', canvasPress.move);
			rootEl?.removeEventListener('pointerup', canvasPress.cancel);
			rootEl?.removeEventListener('pointercancel', canvasPress.cancel);
			rootEl?.removeEventListener('touchstart', onCanvasTouchStart, true);
			rootEl?.removeEventListener('touchmove', onCanvasTouchMove, true);
			rootEl?.removeEventListener('touchend', onCanvasTouchEnd, true);
			rootEl?.removeEventListener('touchcancel', tapZoom.cancel, true);
			canvasPress.cancel(); // a press in flight must not fire into an unmounted editor
			tapZoom.cancel(); // …and neither may a zoom gesture keep writing a torn-down viewport
			onCableEnd(); // …nor may a cable in flight leave name tags lit on a torn-down canvas
			// Do NOT forget this panel's selection here: unmount also fires on a tab switch, and the
			// selection must survive switching away and back.
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('paste', onPaste);
			window.removeEventListener('mousemove', trackMouse);
			// A drag in flight must not leave drop outlines lit on the other panels.
			if (uiStore.nodeDrag !== null) {
				uiStore.nodeDrag = null;
				uiStore.nodeDragTarget = null;
			}
		};
	});
</script>

<SvelteFlowProvider>
	<!-- `canvas-wrap` is the marker PlacementPreview uses to tell a commit click from a cancel. -->
	<div class="editor-panel canvas-wrap" bind:this={rootEl}>
		{#if enteredPath.length > 0}
			<nav class="breadcrumb" data-testid="subpatch-breadcrumb" aria-label="Sub-patch path">
				<Button variant="ghost" size="sm" onclick={() => exitToDepth(0)} title="Back to the top-level patch"
					>Patch</Button
				>
				{#each enteredPath as inst, i (inst)}
					{@const label = g.nodeById(inst)?.name ?? inst}
					<span class="sep">›</span>
					<Button
						variant="ghost"
						size="sm"
						class={i === enteredPath.length - 1 ? 'crumb-current' : ''}
						onclick={() => exitToDepth(i + 1)}
						title="Go to {label}">{label}</Button
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
			onconnectstart={onCableStart}
			onconnectend={onCableEnd}
			onreconnectstart={onCableStart}
			onreconnectend={onCableEnd}
			onnodedragstart={onNodeDragStart}
			onnodedrag={onNodeDrag}
			onnodedragstop={onNodeDragStop}
			onpaneclick={onPaneClick}
			onnodeclick={onNodeClick}
			onselectionstart={() => (boxSelecting = true)}
			onselectionend={onSelectionEnd}
			onedgeclick={onEdgeClick}
			ondelete={deleteElements}
			fitViewOptions={FIT_OPTIONS}
			minZoom={MIN_ZOOM}
			maxZoom={MAX_ZOOM}
			bind:viewport
			zoomOnDoubleClick={false}
			autoPanOnNodeDrag={false}
		>
			<!-- `showLock` off: goofi has no read-only mode, so Flow's lock reads as breakage. -->
			<Controls showLock={false} />
			<FitToGraph {panelId} options={FIT_OPTIONS} />
			<FlowApi bind:screenToFlowPosition={screenToFlow} bind:getViewport bind:setViewport />
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
			<div class="empty-hint" data-testid="empty-hint">
				<EmptyState>
					{#snippet title()}{entered ? 'This sub-patch is empty' : 'Empty patch'}{/snippet}
					{#snippet hint()}Double-click the canvas or press <kbd>Tab</kbd> to add a node.{/snippet}
				</EmptyState>
			</div>
		{/if}

		{#if menuOpen}
			<!-- Fixed and positioned in VIEWPORT coordinates, so both portal to <body>: `.panel-body`
			     is a query container and must never become their containing block. -->
			<div
				class="menu-overlay"
				use:portal
				onclick={() => {
					menuOpen = false;
					menuSeed = null;
				}}
				role="presentation"
			></div>
			<div
				class="menu-anchor"
				bind:this={menuEl}
				use:portal
				data-testid="add-node-menu-anchor"
				style="left: {menuPos.x}px; top: {menuPos.y}px"
			>
				<AddNodeMenu
					seed={menuSeed}
					boundary={entered !== null}
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

		<!-- Absent exactly while the pane is open: the pane covers this corner at every width, and a
		     mounted control under it is invisible but still tabbable. -->
		{#if !(inspectorOn && selectedNode)}
			<IconButton
				class="inspector-toggle"
				label="Toggle inspector"
				title={inspectorOn ? 'Hide the inspector' : 'Show the inspector when a node is selected'}
				aria-pressed={inspectorOn}
				data-testid="inspector-toggle"
				onclick={() =>
					inspectorOn ? sel.toggleInspectorFor(panelId) : sel.showInspectorFor(panelId)}
			>
				◧
			</IconButton>
		{/if}

		<!-- Its ✕ DISMISSES, holding only until the selection changes; the ◧ above is the switch. -->
		<InspectorOverlay
			node={selectedNode}
			enabled={inspectorOn}
			onClose={() => sel.dismissInspectorFor(panelId)}
		/>
	</div>
</SvelteFlowProvider>

<!-- Portaled to <body> so it floats above every panel. -->
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
	/* The inset keeps the 16px corner grips reachable. `margin: 0` is load-bearing: Flow's own
	   `.svelte-flow__panel` sets `margin: 15px`, which STACKS on these offsets. */
	.editor-panel :global(.svelte-flow__controls) {
		margin: 0;
		bottom: var(--space-8);
		left: var(--space-8);
	}
	/* On touch each button is floored to --hit in both axes, so the cluster is a slab and needs a
	   smaller inset — still clear of the grip, whose 16px box is clipped to a triangle. */
	@media (hover: none) and (pointer: coarse) {
		.editor-panel :global(.svelte-flow__controls) {
			bottom: var(--space-6);
			left: var(--space-6);
		}
	}
	/* Non-interactive, so it never eats the double-click that opens the add-node menu under it. */
	.empty-hint {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
		z-index: 1;
	}
	.empty-hint kbd {
		font-size: var(--fs-small);
		padding: 1px var(--space-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-1);
	}
	.editor-panel :global(.inspector-toggle) {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 5;
		/* Not `--disabled-opacity`: a ghosted affordance over the canvas, not a disabled control. */
		opacity: 0.5;
	}
	.editor-panel :global(.inspector-toggle:hover),
	.editor-panel :global(.inspector-toggle[aria-pressed='true']) {
		opacity: 1;
	}
	.link-ghost {
		position: fixed;
		/* Offset off the cursor so it reads as carried, not pinned. */
		transform: translate(14px, 12px);
		z-index: var(--z-tab-drag);
		pointer-events: none;
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-2) var(--space-5);
		max-width: 220px;
		background: var(--surface-2);
		border: 1px solid var(--accent);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.lg-icon {
		font-size: var(--fs-small);
	}
	.menu-overlay {
		position: fixed;
		inset: 0;
		z-index: calc(var(--z-addmenu) - 1);
	}
	.menu-anchor {
		position: fixed;
		z-index: var(--z-addmenu);
		/* The clamp can only SHIFT a surface that fits, so on a narrow phone the width gives first. */
		width: min(320px, calc(100vw - var(--space-8)));
	}
	.breadcrumb {
		position: absolute;
		top: 10px;
		left: 10px;
		z-index: 6;
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-5);
		max-width: calc(100% - 90px);
		overflow: hidden;
		background: color-mix(in srgb, var(--surface-1) 88%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-1);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
	}
	.breadcrumb :global(.crumb-current) {
		font-weight: 600;
		color: var(--text);
	}
	.breadcrumb .sep {
		color: var(--text-muted);
	}
</style>
