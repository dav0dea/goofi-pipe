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
	import { ui, slotKey, type SlotClickSeed } from '$lib/stores/ui.svelte';
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
		type NodeTypeInfo
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
	import { nodeSurfaceSize, inputUnits, BOUNDARY } from '$lib/editor/nodeMetrics';
	import {
		inputAnchors,
		nearSlots,
		sameKeys,
		SLOT_PROXIMITY_PX,
		type SlotAnchor
	} from '$lib/editor/slotProximity';
	import { createLongPress } from '$lib/editor/longPress';
	import { createDoubleTapZoom, zoomStep, type FlowViewport } from '$lib/editor/doubleTapZoom';
	import { eventPoint } from '$lib/editor/eventPoint';
	import { serializeClipboard, parseClipboard, clipToSpecs } from '$lib/editor/clipboard';
	import { copyText } from '$lib/clipboard';
	import { registerEditor, unregisterEditor } from './editorCommands';
	import InspectorOverlay from './InspectorOverlay.svelte';
	import { arrayToPath, asStateObject, pathToArray } from '$lib/workspace/panelState';
	import {
		Button,
		IconButton,
		EmptyState,
		clampToViewport,
		overlayViewport,
		isTextEditingTarget
	} from '$lib/ui';
	import { onMount, tick, untrack } from 'svelte';

	let { panelId, state: panelState, setState }: PanelProps = $props();

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

	// Whether this editor's inspector pane actually shows (per-panel; default on). The visible
	// state is the standing ◧ preference MINUS a live ✕ dismissal — the ✕ closes the pane only
	// until the selection next changes, while the ◧ is the real off-switch.
	const inspectorOn = $derived(sel.inspectorVisibleFor(panelId));

	$effect(() => {
		// When this editor becomes the active panel, mark it the active editor
		// so the standalone Parameters/Metadata/Errors panels follow it.
		if (ws.activePanelId === panelId) sel.setActiveEditor(panelId);
	});

	let rootEl = $state<HTMLDivElement | null>(null);

	/** Which edge of the menu's own box the requested open point names. */
	type MenuAlign = 'start' | 'center' | 'end';

	let menuOpen = $state(false);
	// The REQUESTED spawn point (what a call site asked for) and the RENDERED one (what the measured
	// clamp settled on) are separate, so the placement effect never writes its own dependency.
	let menuAt = $state<{ x: number; y: number; align: MenuAlign }>({ x: 120, y: 120, align: 'start' });
	let menuPos = $state<{ x: number; y: number }>({ x: 120, y: 120 });
	let menuSeed = $state<SlotClickSeed | null>(null);
	let menuEl = $state<HTMLDivElement | null>(null);
	// A long press opens the menu with the finger still down, so the touchend ENDING that same
	// gesture arrives as a compat click — at the RELEASE point, which is wherever the clamp settled
	// the menu. Exactly one such click is swallowed. Every `openAddMenu` sets this, so it can never
	// leak into a later menu.
	//
	// At the WINDOW, in the capture phase, because WHICH layer that click lands on is not knowable:
	// a press low on the screen makes the clamp slide the menu up over the press point, so the
	// release hits a palette row (a node type the user never chose) rather than the catcher
	// underneath. Consuming it before it reaches any target makes the swallow geometry-independent.
	// TitleTip solves the same long-press/compat-click problem the same way. One click and the
	// listener is gone, so a release that produces none can cost at most the next click — it can
	// never leave the menu inert.
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

	/**
	 * Open the add-node menu at a viewport point — the ONE placement path for all four entry points
	 * (a port click, a canvas double-click, Tab, a canvas long press). The point names the menu's
	 * left, centre or right edge; the effect below measures the rendered menu and clamps it on-screen
	 * through the shared `clampToViewport` (D-M2: the point-anchored menus keep their own shell and
	 * share only the clamp SSOT). No call site knows the menu's size.
	 */
	function openAddMenu(
		x: number,
		y: number,
		align: MenuAlign = 'start',
		seed: SlotClickSeed | null = null,
		swallowNextDismiss = false
	): void {
		menuAt = { x, y, align };
		menuPos = { x, y }; // a known spawn point to start from, corrected below before paint
		menuSeed = seed;
		swallowMenuClick = swallowNextDismiss;
		menuOpen = true;
	}

	/**
	 * The coarse-pointer door onto the add-node menu (R spec §3.2b). The other three routes are all
	 * fine-pointer or keyboard — double-click, Tab, a port click — so on touch this is the only way
	 * to put the first node on an empty canvas.
	 *
	 * Armed for `touch` alone: a held mouse button is the start of a desktop pan, and desktop
	 * behaviour is the reference. The recognizer never stops propagation, so the same pointerdown
	 * still reaches SvelteFlow's pan and `Panel`'s capture-phase `setActive`.
	 */
	const canvasPress = createLongPress((at) =>
		openAddMenu(at.clientX - 8, at.clientY + 8, 'start', null, true)
	);

	/** Empty canvas only — the pane is the direct target just where nothing else is drawn, so a
	 * press on a node (which drags) or on the flow controls can never arm the door.
	 *
	 * And not while a ghost is pending: the canvas belongs to the placement until it is put down, so
	 * a finger resting on it mid-drag is positioning a node, not asking for a second menu on top of
	 * the one that produced the ghost. */
	function onCanvasPointerDown(e: PointerEvent): void {
		if (e.pointerType !== 'touch' || pendingPlacement) return;
		if (!onBareCanvas(e.target)) return;
		canvasPress.start(e);
	}

	const onBareCanvas = (target: EventTarget | null): boolean =>
		Boolean((target as HTMLElement | null)?.classList.contains('svelte-flow__pane'));

	// --- double-tap-and-drag to zoom ------------------------------------------------------------
	// The one-handed zoom, ADDED beside pinch: `zoomOnPinch` is still SvelteFlow's own and is not
	// touched, and the seam this uses is `zoomOnDoubleClick={false}` below — a double tap that was
	// already inert. The recognizer and the anchored viewport arithmetic are in
	// `editor/doubleTapZoom.ts`, where a unit test can reach them; what is left here is the seam.
	const tapZoom = createDoubleTapZoom();
	// The viewport the gesture started from, and the FLOW point under the tap it is taken about.
	// Sampled ONCE at the start, so every move is measured against the same origin and the drag
	// cannot accumulate rounding.
	let zoomFrom: FlowViewport | null = null;
	let zoomAnchor: { x: number; y: number } | null = null;

	/**
	 * THE PAN BLOCK, and why it is on `touchstart` rather than on `pointerdown`.
	 *
	 * SvelteFlow pans by d3-zoom, which binds `touchstart` on its own pane wrapper — a pointerdown
	 * blocker cannot reach it, however it is written. This is the same lesson (and the same shape)
	 * as `PlacementPreview.svelte`'s blocker: a CAPTURE listener above the pane, and
	 * `stopPropagation` — that is the half that stops the pan. `preventDefault` says the touch is
	 * consumed, suppressing the compat mouse cascade and the browser's own double-tap default; what
	 * keeps the add-node menu shut is not it but `canvasPress.cancel()` below, which is why that is
	 * the line `touch-zoom.spec.ts` holds still for 600 ms to pin.
	 *
	 * Held back for EXACTLY the touch that starts the gesture — the second tap — and no other. The
	 * first tap, and every touch that is not part of a double tap, reaches d3-zoom untouched, which
	 * is what keeps one-finger panning and tap-to-select alive.
	 */
	function onCanvasTouchStart(e: TouchEvent): void {
		// A second finger is a PINCH: hand the whole gesture back rather than compete with it.
		if (pendingPlacement || e.touches.length > 1 || !onBareCanvas(e.target)) {
			tapZoom.cancel();
			return;
		}
		const p = eventPoint(e);
		if (!p || !tapZoom.down(p, e.timeStamp)) return;

		// A double tap held still would otherwise also fire the long-press door and open the add-node
		// menu on top of the zoom. The press armed on this touch's `pointerdown`, which precedes
		// `touchstart`, so disarming it here is in time.
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

	// ContextMenu's idiom: measure the mounted menu, then re-clamp. A spawn point is the degenerate
	// anchor rect the shared clamp already understands (`left`/`bottom` ARE the point).
	$effect(() => {
		const el = menuEl;
		if (!el) return;
		const at = menuAt;
		const place = (): void => {
			const r = el.getBoundingClientRect();
			const left =
				at.align === 'center' ? at.x - r.width / 2 : at.align === 'end' ? at.x - r.width : at.x;
			// `overlayViewport()`, not `window.innerHeight` — the same measurement Popover and
			// ContextMenu clamp against. It matters more here than anywhere: this menu FOCUSES its
			// search on open, so the soft keyboard is on its way up as the menu lands, and the layout
			// viewport does not shrink to make room for it.
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

	// Watch ui.pendingSlotClick — when a node's port is clicked, open the
	// menu near the port and seed it with the dtype filter + auto-link. Only
	// the active panel consumes it (the click sets this panel active first).
	$effect(() => {
		const seed = uiStore.pendingSlotClick;
		if (!seed) return;
		if (!isActive()) return;
		uiStore.consumeSlotClick();
		// A source port hangs the menu to its right, a target port to its left — 12px of clearance
		// either way, with the alignment (not a hardcoded width) doing the mirroring.
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
	// Bumped to force a flowEdges re-derive from authoritative state. SvelteFlow
	// optimistically inserts a dropped connection into the bound edges BEFORE onConnect
	// runs; a SUCCESSFUL wire's echo rebuilds the edges and discards it, but a REJECTED
	// wire (dtype/duplicate) produces no echo — so we bump this in the failure path to
	// strip the ghost edge instead of leaving a second cable hanging on the pill.
	let reconcileTick = $state(0);

	// --- sub-patch navigation (enter-to-edit) -------------------------------
	// The stack of instance ids the editor has descended into; empty = top level.
	// (A stack so nesting is a natural extension; today one level deep is reached.)
	// Seeded from the persisted layout state so a saved/reloaded patch restores
	// which sub-patch this editor was inside.
	let enteredPath = $state<string[]>(untrack(() => pathToArray(asStateObject(panelState).subpatchPath)));
	const entered = $derived(enteredPath.length ? enteredPath[enteredPath.length - 1] : null);

	/** Write the current path back into the panel's layout state (round-trips
	 * through set_layout / the saved .gfi). Untracked so callers in effects don't
	 * pick up `state` as a dependency. Classified as NAVIGATION (D-R3): descending
	 * into a sub-patch is looking, not editing — it is persisted so a reload lands
	 * back where you were, but it must not mark the patch unsaved. */
	function persistEnteredPath(): void {
		untrack(() => {
			const path = arrayToPath(enteredPath);
			if (asStateObject(panelState).subpatchPath === path) return;
			setState({ ...asStateObject(panelState), subpatchPath: path }, 'navigation');
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
		// unwired), at their server-stored positions.
		const inst = entered ? g.instances[entered] : null;
		if (inst && entered) {
			for (const [bid, port] of Object.entries(inst.interface)) {
				next.push({
					// The flow id keys on the stable boundary id (routing); the pill shows
					// the renameable NAME, and double-clicking it drives rename_boundary.
					id: boundaryId(entered, bid),
					type: 'boundary',
					position: { x: port.pos[0], y: port.pos[1] },
					data: {
						name: port.name ?? bid,
						dir: port.dir,
						dtype: port.dtype ?? 'ARRAY',
						wired: port.inner_node !== null,
						rename: (newName: string) => g.renameBoundary(entered!, bid, newName)
					},
					selected: sel.nodes(panelId).has(boundaryId(entered, bid))
				});
			}
		}
		flowNodes = next;
	});

	// Every link, rerouted to the nearest VISIBLE boundary in the entered scope (null =
	// root). Skip when an endpoint is not exposed up the chain, or both ends resolve to
	// the same drawn node (internal to one collapsed sub-patch -> hidden). Inside an
	// entered instance, also draw its boundary pill edges (In pill -> member input,
	// member output -> Out pill); WIRED boundaries only, deletable to unwire.
	$effect(() => {
		reconcileTick; // re-derive on demand to drop an optimistic ghost edge after a rejected wire
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
			void g.wireBoundary(entered, bnd, local, memberSlot).catch((e) => {
				console.warn('wire boundary failed', e);
				reconcileTick++; // drop the optimistic ghost edge SvelteFlow drew before this RPC
			});
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

	// --- input names, revealed by proximity while a cable is in flight ---------------------------
	//
	// An input slot's name is drawn only on its connector pill, and only a hover asked for it — so
	// touch rested every tag on the canvas open, permanently. The door is proximity now, and it is
	// the SAME door on both modalities: while a cable is being dragged, the inputs the pointer is
	// closing on name themselves. The arithmetic is in `editor/slotProximity.ts`, where a unit test
	// can reach it; what is left here is the seam.
	//
	// Cheap by construction — the anchors are computed from state this panel already holds (node
	// positions + `nodeMetrics`' connector pitch), so nothing measures the DOM on a pointermove. They
	// are snapshotted ONCE per drag in FLOW space and the pointer is converted into that space per
	// move, which is also why a canvas that pans or zooms mid-drag (auto-pan on connect does exactly
	// that) needs no invalidation: the transform is read fresh every time, the anchors never move.
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
		// The radius is a SCREEN distance (a fingertip is a physical size), so it is the radius that
		// is converted into flow space, not the anchors out of it.
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
		// On `window`, not the panel: a cable dragged past the panel's edge is still in flight, and
		// SvelteFlow's own connection listeners sit on the document for the same reason.
		window.addEventListener('pointermove', onCableMove);
	}

	function onCableEnd(): void {
		window.removeEventListener('pointermove', onCableMove);
		cableAnchors = [];
		publishCableNear(new Set());
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
			const multi = new Set(node.input_multi ?? []);
			return nodeSurfaceSize(
				inputUnits(inputs, (s) => multi.has(s)),
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
		const p = eventPoint(event);
		if (!p) return null;
		const t = panelUnder(p.clientX, p.clientY);
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
			// Same extraction as `linkTargetAt`: the ghost has to follow the FINGER, not a `clientX`
			// a TouchEvent does not have (it read `undefined`, so the chip parked at 0,0).
			const p = eventPoint(args.event) ?? { clientX: 0, clientY: 0 };
			// `id` is the uid; the floating chip shows the display name.
			linkGhost = { x: p.clientX, y: p.clientY, name: g.nodeById(args.nodes[0]?.id ?? '')?.name ?? '' };
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
			// One transaction so moving N selected nodes is a single Ctrl+Z. Each set*Pos records
			// AFTER its command RPC resolves (B3), so the calls must be AWAITED inside the transaction
			// — else the buffer is empty at flush and each records its own top-level step.
			const label = args.nodes.length > 1 ? `Move ${args.nodes.length} nodes` : 'Move node';
			void history().transaction(label, async () => {
				for (const n of args.nodes) {
					const pos: [number, number] = [Math.round(n.position.x + dx), Math.round(n.position.y + dy)];
					const bnd = parseBoundary(n.id);
					if (bnd && entered) {
						await g.setBoundaryPos(entered, bnd, pos).catch(() => {});
					} else {
						await g.setNodePos(n.id, pos);
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
			openAddMenu(here.x - 8, here.y + 8);
			lastPaneClickAt = 0;
			return;
		}
		lastPaneClickAt = now;
		lastPaneClickPos = here;
		menuOpen = false;
		sel.clickPane(panelId, args.event.shiftKey);
		// SvelteFlow calls `unselectNodesAndEdges()` immediately AFTER this callback, whatever the
		// store decided — so wherever the store KEEPS the selection (multi-select mode, where a stray
		// tap on canvas must not undo what the mode is building) the canvas and the store would
		// disagree: nothing painted as selected, every row gated on the rendered flags reading as
		// dead, and the store still holding the operands a Delete would take. Re-derive after it.
		if (sel.nodes(panelId).size || sel.edges(panelId).size) void tick().then(reassertSelection);
	}

	/** Push the store's selection back onto the rendered `selected` flags. The store is the source
	 * of truth for everything but a live marquee, which `onSelectionEnd` folds into it. */
	function reassertSelection(): void {
		flowNodes = flowNodes.map((n) => ({ ...n, selected: sel.nodes(panelId).has(n.id) }));
		flowEdges = flowEdges.map((e) => ({ ...e, selected: sel.edges(panelId).has(e.id) }));
	}

	function onNodeClick(args: { node: Node; event: MouseEvent | TouchEvent }): void {
		// A click can land in the window between a graph mutation and the flowNodes rebuild,
		// carrying an id that no longer exists (SvelteFlow then logs "Node … does not exist").
		// Ignore a click whose node is no longer in the current scene.
		if (!flowNodes.some((n) => n.id === args.node.id)) return;
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
	//
	// Registered in the CAPTURE phase, and a recognised second click is CONSUMED. The FIRST click
	// selects the instance, which slides the inspector in over 120ms across all but `--hit` of the
	// editor — including the node the second click has to land on. In the bubble phase that click
	// had already actuated whatever control had arrived under the pointer on its way up: for a
	// sub-patch that is `subpatch-expand-inspector`, which DISSOLVES it. Capturing lets the gesture
	// claim its own second half before a surface that was not there when the gesture started.
	const DBL_PX = 6; // a real double-click barely moves the pointer…
	const DBL_PX_TOUCH = 16; // …but a finger does, and 6px is an order of magnitude under any tap slop
	let lastClickInst = '';
	let lastClickAt = 0;
	let lastClickX = 0;
	let lastClickY = 0;
	/** The node a click landed on, or '' — real identity, not a coordinate. */
	function nodeUnder(target: EventTarget | null): string {
		return (
			(target as HTMLElement | null)?.closest?.('.svelte-flow__node')?.getAttribute('data-id') ?? ''
		);
	}
	function onCanvasClick(event: MouseEvent): void {
		const now = performance.now();
		const hereNode = nodeUnder(event.target);
		// …of which only a sub-patch instance is something this gesture can ENTER.
		const here = hereNode in g.instances ? hereNode : '';
		// Per gesture, not per device (D-R2): the same `pointerType` seam the long-press door reads.
		const slop = (event as PointerEvent).pointerType === 'touch' ? DBL_PX_TOUCH : DBL_PX;
		if (
			lastClickInst &&
			now - lastClickAt < DOUBLE_CLICK_MS &&
			Math.abs(event.clientX - lastClickX) < slop &&
			Math.abs(event.clientY - lastClickY) < slop &&
			// A second click that resolves to a DIFFERENT NODE is that node's first click, not this
			// one's second — the widened touch slop puts two adjacent nodes inside one window. Asked
			// of the NODE, not of the instance: `g.instances` holds only sub-patches, so an ordinary
			// node also answered '' and so took the "nothing under the pointer" path this guard
			// reserves for the inspector having slid over the node mid-gesture. `lastClickInst` is
			// itself a node id whenever it is non-empty, which is the only case this branch runs in.
			(hereNode === '' || hereNode === lastClickInst)
		) {
			const inst = lastClickInst;
			lastClickInst = '';
			// The gesture owns this click: neither the inspector that just slid over the node nor the
			// pane behind it may also act on it. `preventDefault` covers the activation behaviour a
			// checkbox would still run with propagation merely stopped.
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

	const nodeTypes = { goofi: GoofiNode, boundary: BoundaryNode };

	/** Framing for every programmatic fit — the Controls/“F” button (via the
	 * `fitViewOptions` prop) and the on-load fit in <FitToGraph>. */
	const FIT_OPTIONS = { maxZoom: 1, padding: 0.18 } satisfies FitViewOptions;

	/** How far the canvas may be zoomed. Stated once: <SvelteFlow> is told, and the double-tap zoom
	 *  clamps itself to the same pair rather than discovering them by being refused. */
	const MIN_ZOOM = 0.05;
	const MAX_ZOOM = 4;

	/** True when the CANVAS owns the keyboard rather than a control: nothing focused at all, the pane
	 * itself, or a node's own wrapper — SvelteFlow gives that `tabindex="0"`, so a plain click on a
	 * node parks focus there, and "select a node, then Tab to add the next one" is the whole point of
	 * the shortcut. Deliberately NOT a focusable DESCENDANT of a node: R's `.conn` slot pills are
	 * inside one and Tab is how a keyboard reaches the next of them (`.conn-label`'s focus reveal).
	 * A header button, an add-menu row, an inspector control: all the browser's to traverse. */
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
		// A modal <dialog> owns the keyboard while it is up, and the DOM is what says so — NOT
		// `ui().modalOpen`, which is a ref-count that an in-panel fx editor raises for a merely
		// expanded textarea. Reading that flag here would stand the canvas down for an inspector
		// field; asking the top layer answers the narrower question exactly. (The file browser opens
		// with focus on its path field, which the allowlist below covers — but load mode requires a
		// navigation click, and from that <button> Escape used to dismiss the dialog AND run this
		// panel's Escape ladder behind it.)
		if (t?.closest?.('dialog[open]')) return;
		// Not a tag list: since X the inspector's expression editor is a contenteditable, so Ctrl+A in
		// it has to take the expression and not every node on the canvas (`ui/textEditing.ts`).
		if (isTextEditingTarget(t)) return;

		const meta = e.ctrlKey || e.metaKey;
		if (meta && e.key.toLowerCase() === 'a') {
			e.preventDefault();
			selectAll();
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
		} else if (e.key === 'Tab' && !e.shiftKey && canvasHasKeys(t)) {
			// Scoped to the bare canvas (R2-3). Unscoped this was a ONE-WAY TRAP: the menu's #if block
			// below is unkeyed, so re-entering `openAddMenu` with the menu already open neither
			// remounts it nor re-fires the search focus that made the FIRST Tab look fine — every press
			// after that was a bare preventDefault with nothing to refocus, and no chrome outside this
			// canvas was ever Tab-reachable (WCAG 2.1.2). Shift+Tab is left alone on purpose: it is the
			// way back OUT of a canvas that nothing has focused yet.
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

	/** The single delete path — SvelteFlow's `ondelete` (deleteKey wires Delete+Backspace to it;
	 * the custom keydown handler no longer deletes) and the app header's Delete row, which is the
	 * only door a phone has (R-Task 6). Covers app-store AND marquee selection: SvelteFlow filters
	 * by each element's `selected`, and so does `deleteSelection` below.
	 * Real nodes are deleted as ONE batch (removeNodes) so undo restores them all BEFORE their
	 * links; boundary pills are unwired individually. */
	async function deleteElements({ nodes, edges }: { nodes: Node[]; edges: Edge[] }): Promise<void> {
		const nodeIds: string[] = [];
		for (const n of nodes) {
			const bnd = parseBoundary(n.id);
			if (bnd && entered) await g.removeBoundary(entered, bnd).catch(() => {});
			else nodeIds.push(n.id);
		}
		const deleted = new Set(nodeIds);
		await g.removeNodes(nodeIds).catch(() => {});
		for (const e of edges) {
			// Deleting an In→member / member→Out edge unwires the boundary (the
			// pill survives); a normal flat link is removed.
			const bnd = parseBoundary(e.source) ?? parseBoundary(e.target);
			if (bnd && entered) {
				await g.wireBoundary(entered, bnd, null, null).catch(() => {});
				continue;
			}
			// A normal link touching a batch-deleted node was already removed
			// with it (removeNodes), so don't double-record its removal.
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

	/** What the Delete key would delete: the rendered `selected` flags, which is the union of the
	 * store's selection and a live marquee. */
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

	/** Select what's actually on screen: the entered sub-patch's members, or every top-level node
	 * PLUS the collapsed sub-patch group nodes (which are virtual nodes — they select like any
	 * node). ⌘A's action, and the app header's Select all row. */
	function selectAll(): void {
		const kids = childrenOfScope(
			entered ?? ROOT_ID,
			g.instances,
			g.nodes.map((n) => n.uid),
			memberIndex
		);
		sel.selectNodes(panelId, [...kids.nodeUids, ...kids.instUids]);
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

	/** Open the add-node menu centered over this panel — the `window.goofi` façade's entry point,
	 * which names a panel rather than a point. */
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
	// Likewise the viewport itself, which the double-tap zoom reads and writes.
	let getViewport = $state<(() => FlowViewport) | undefined>(undefined);
	let setViewport = $state<((v: FlowViewport) => void) | undefined>(undefined);

	function fitView(): void {
		rootEl?.querySelector<HTMLButtonElement>('.svelte-flow__controls-fitview')?.click();
	}

	/** Select a node in this editor (and make it the active selection the
	 * standalone panels follow). The shared handle the error panel and the agent
	 * surface use to focus a node. */
	function focusNode(uid: string): void {
		// Flow nodes are keyed by uid (id: uid); select by uid, not display name.
		sel.selectNodes(panelId, [uid]);
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
			pasteClipboard: () => void pasteClipboard(),
			duplicateSelection: () => void duplicateSelection(),
			hasSelection
		});
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('mousemove', trackMouse);
		rootEl?.addEventListener('click', onCanvasClick, true);
		rootEl?.addEventListener('pointerdown', onCanvasPointerDown);
		rootEl?.addEventListener('pointermove', canvasPress.move);
		rootEl?.addEventListener('pointerup', canvasPress.cancel);
		rootEl?.addEventListener('pointercancel', canvasPress.cancel);
		// CAPTURE, so these run before d3-zoom's own listeners further in; `passive: false` so the
		// `preventDefault` above is honoured. A touch is delivered to the element it STARTED on for
		// its whole life, so a gesture that began on this canvas keeps reaching these even when the
		// finger wanders off the panel.
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
				<Button variant="ghost" size="sm" onclick={() => exitToDepth(0)} title="Back to the top-level patch"
					>Patch</Button
				>
				{#each enteredPath as inst, i (inst)}
					{@const label = g.instances[inst]?.name ?? inst}
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
			initialViewport={{ x: 0, y: 0, zoom: 0.85 }}
			zoomOnDoubleClick={false}
			autoPanOnNodeDrag={false}
		>
			<!-- Viewport controls only. `showLock` is Flow's Toggle Interactivity button, which flips
			     nodesDraggable/nodesConnectable/elementsSelectable off in one click: goofi has no
			     read-only mode and nothing else in the UI would say the canvas is locked, so it reads
			     as breakage rather than as a mode. -->
			<Controls showLock={false} />
			<FitToGraph options={FIT_OPTIONS} />
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
			<!-- First-run / empty-canvas hint. pointer-events:none so double-click
			     (open add-node menu) and panning still reach the canvas underneath. -->
			<div class="empty-hint" data-testid="empty-hint">
				<EmptyState>
					{#snippet title()}{entered ? 'This sub-patch is empty' : 'Empty patch'}{/snippet}
					{#snippet hint()}Double-click the canvas or press <kbd>Tab</kbd> to add a node.{/snippet}
				</EmptyState>
			</div>
		{/if}

		{#if menuOpen}
			<!-- The add-node menu + its click-catcher are `position: fixed`, positioned in VIEWPORT
			     coordinates (menuPos is computed from window.innerWidth / clientX). Portal them to
			     <body> so their containing block is always the viewport, never the panel: `.panel-body`
			     is now a `container-type` query container, and though `inline-size` containment alone
			     does not trap fixed descendants, this keeps the menu correct if the body ever gains a
			     real containing-block trigger (a transform/filter) — matching the sibling overlays
			     (ViewerSettingsMenu / the link-ghost) which all portal for this reason. -->
			<div
				class="menu-overlay"
				use:portal
				onclick={() => {
					// The opening press's own release never reaches here — the one-shot window-capture
					// swallow above consumes it, wherever the clamp put the menu.
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

		<!-- The affordance that brings this editor's inspector back. It is absent exactly while
		     the pane is open, because the pane covers this corner at every width — leaving it
		     mounted meant an invisible, tabbable control under an opaque surface (D-R9's z-order
		     half). Hidden-by-any-means, one press answers "show it" (clearing a ✕ dismissal AND
		     the ◧ preference); visible-but-parked (no node selected), it is the standing
		     off-switch. -->
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

		<!-- Per-editor selection inspector — slides in within this panel. Its ✕ DISMISSES — a
		     close that holds only until the selection changes — while the ◧ above is the standing
		     off-switch. -->
		<InspectorOverlay
			node={selectedNode}
			enabled={inspectorOn}
			onClose={() => sel.dismissInspectorFor(panelId)}
		/>
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
	/* Nudge SvelteFlow's controls off the panel corner so the 16px corner grips (drag-split /
	   drag-join) stay reachable — that clearance is the whole reason for an inset here.
	   `margin: 0` is load-bearing: Flow's own `.svelte-flow__panel` sets `margin: 15px`, which
	   STACKS on these offsets, so the 20px this rule used to declare rendered as 35px. */
	.editor-panel :global(.svelte-flow__controls) {
		margin: 0;
		bottom: var(--space-8);
		left: var(--space-8);
	}
	/* On touch every control button is floored to --hit in BOTH axes — `min-height` from app.css's
	   blanket coarse rule and `min-width` from the scoped one beside it, because SvelteFlow's own
	   stylesheet sizes them 26×26. (This comment claimed the slab before the width half existed,
	   which is very plausibly why nobody re-measured it; the 44×44 is now a row in
	   `tests/e2e/tests/touch-hit-floor.spec.ts`'s SITES registry rather than a claim here.) So the
	   cluster is a slab rather than a strip, and a desktop-sized inset parks it well inside the
	   canvas instead of in its corner. It tucks in — still clear of the grip, whose 16px box is
	   clipped to its lower-left triangle, so a cluster cornered at (g, g) misses it once g + g > 16. */
	@media (hover: none) and (pointer: coarse) {
		.editor-panel :global(.svelte-flow__controls) {
			bottom: var(--space-6);
			left: var(--space-6);
		}
	}
	/* First-run hint over an empty canvas. Non-interactive so it never eats the
	   double-click that opens the add-node menu underneath it. */
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
	/* Per-editor inspector affordance (an IconButton), parked top-right. Subtle until hovered so
	   it doesn't compete with the canvas; brought forward while the inspector is on (aria-pressed). */
	.editor-panel :global(.inspector-toggle) {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 5;
		/* opacity: intentional — a ghosted affordance over the canvas, not a disabled control;
		   it rises to 1 on hover and while the inspector is on. */
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
		/* The clamp below can only SHIFT a surface that fits; a 320px menu on a 320px phone cannot
		   sit inside the clamp's own 6px margins at any offset, so the width has to give first. */
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
	/* The current (deepest) crumb reads as where-you-are: bold. The crumbs are ghost Buttons,
	   so this targets the primitive's element via :global. */
	.breadcrumb :global(.crumb-current) {
		font-weight: 600;
		color: var(--text);
	}
	.breadcrumb .sep {
		color: var(--text-muted);
	}
</style>
