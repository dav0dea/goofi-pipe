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
	import GoofiNode from './GoofiNode.svelte';
	import AddNodeMenu from './AddNodeMenu.svelte';
	import ParamPanel from '$lib/params/ParamPanel.svelte';
	import TopBar from './TopBar.svelte';
	import ErrorPanel from './ErrorPanel.svelte';
	import MetadataPanel from './MetadataPanel.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { ui, type SlotClickSeed } from '$lib/stores/ui.svelte';
	import type { LinkInfo, NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';
	import { onMount } from 'svelte';

	const g = graph();
	const uiStore = ui();

	// Snap tunables. Match goofi3's feel without porting its module.
	const SNAP_THRESHOLD = 15; // engage snap when within this many flow-units
	const SNAP_RANGE = 45; // start drawing guide hints from this far out
	const DEFAULT_NODE_W = 240; // fallback when Svelte Flow hasn't measured a node yet
	const DEFAULT_NODE_H = 120;

	let menuOpen = $state(false);
	let menuPos = $state<{ x: number; y: number }>({ x: 120, y: 120 });
	let menuSeed = $state<SlotClickSeed | null>(null);
	let selection = $state<Set<string>>(new Set());
	// Selected edges. Tracked separately from `selection` so node and edge
	// selection don't clobber each other; Delete drains both sets.
	let edgeSelection = $state<Set<string>>(new Set());

	// Watch ui.pendingSlotClick — when a node's port is clicked, open the
	// menu near the port and seed it with the dtype filter + auto-link.
	$effect(() => {
		const seed = uiStore.pendingSlotClick;
		if (!seed) return;
		uiStore.consumeSlotClick();
		menuSeed = seed;
		// Position the menu to the right of the click for output ports, to
		// the left for input ports. Clamp into viewport.
		const offsetX = seed.side === 'source' ? 12 : -332;
		menuPos = {
			x: Math.max(8, Math.min(window.innerWidth - 332, seed.clientX + offsetX)),
			y: Math.max(8, Math.min(window.innerHeight - 360, seed.clientY - 24))
		};
		menuOpen = true;
	});

	type Bounds = { left: number; right: number; top: number; bottom: number; cx: number; cy: number };
	type Guide = { x?: number; y?: number; opacity: number };

	let snapGuides = $state<Guide[]>([]);
	// Per-drag bookkeeping: original positions of every dragged node when the
	// drag started. We compute a fresh snap delta off these each frame so the
	// guides stay coherent even as Svelte Flow moves the nodes 1:1 with the
	// mouse.
	let dragStartPositions: Map<string, { x: number; y: number }> = new Map();

	// Lift backend nodes/links → Svelte Flow shapes. We rely on $derived
	// to bridge state-store snapshots into SvelteFlow's input arrays.
	let flowNodes = $state.raw<Node[]>([]);
	let flowEdges = $state.raw<Edge[]>([]);

	$effect(() => {
		const next: Node[] = g.nodes.map((n) => ({
			id: n.name,
			type: 'goofi',
			position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
			data: { node: n },
			selected: selection.has(n.name)
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
				selected: edgeSelection.has(id),
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
		const id = args.edge.id;
		const e = args.event;
		if (e.shiftKey || e.ctrlKey || e.metaKey) {
			const next = new Set(edgeSelection);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			edgeSelection = next;
		} else {
			edgeSelection = new Set([id]);
			selection = new Set();
		}
	}

	function findLinkById(id: string): LinkInfo | null {
		for (const l of g.links) if (edgeId(l) === id) return l;
		return null;
	}

	async function deleteEdgeSelection(): Promise<void> {
		const ids = Array.from(edgeSelection);
		for (const id of ids) {
			const link = findLinkById(id);
			if (!link) continue;
			try {
				await g.removeLink(link);
			} catch (err) {
				console.warn('remove edge failed', err);
			}
		}
		edgeSelection = new Set();
	}

	function nodeBoundsFromFlow(id: string, x: number, y: number): Bounds {
		const flowNode = flowNodes.find((n) => n.id === id);
		const w = flowNode?.measured?.width ?? DEFAULT_NODE_W;
		const h = flowNode?.measured?.height ?? DEFAULT_NODE_H;
		return {
			left: x,
			right: x + w,
			top: y,
			bottom: y + h,
			cx: x + w / 2,
			cy: y + h / 2
		};
	}

	/** Compute the snap delta + guide lines for the currently-dragging set.
	 *
	 * `current` carries the live mouse-tracked positions reported by Svelte
	 * Flow; `start` carries the positions at drag-start. We snap the *current*
	 * bounds toward any unselected node's edge, center, or gap.
	 */
	function computeSnapDelta(
		current: Map<string, { x: number; y: number }>,
		altKey: boolean
	): { dx: number; dy: number; guides: Guide[] } {
		if (altKey || current.size === 0) return { dx: 0, dy: 0, guides: [] };

		// Build bounds for the dragged set (using current positions) and
		// the rest of the graph (using committed positions from the store).
		const draggedBounds: Bounds[] = [];
		for (const [id, pos] of current) draggedBounds.push(nodeBoundsFromFlow(id, pos.x, pos.y));
		const targets: Bounds[] = [];
		for (const n of g.nodes) {
			if (current.has(n.name)) continue;
			targets.push(nodeBoundsFromFlow(n.name, n.pos[0], n.pos[1]));
		}
		if (targets.length === 0) return { dx: 0, dy: 0, guides: [] };

		// Multi-pass: find the best snap delta, then collect guides for the
		// post-snap configuration. Matches goofi3's `snapMultiDrag` shape.
		const V_GAPS = [0, DEFAULT_NODE_H * 0.5];
		const H_GAPS = [0, DEFAULT_NODE_W * 0.25];
		let bestDistY = Infinity;
		let bestDy = 0;
		let bestDistX = Infinity;
		let bestDx = 0;

		for (const me of draggedBounds) {
			for (const oe of targets) {
				for (const gap of V_GAPS) {
					const yPairs: [number, number][] = [
						[me.top, oe.top],
						[me.bottom, oe.bottom],
						[me.top, oe.bottom + gap],
						[me.bottom, oe.top - gap]
					];
					if (gap === 0) yPairs.push([me.cy, oe.cy]);
					for (const [myE, otherE] of yPairs) {
						const d = Math.abs(myE - otherE);
						if (d < SNAP_THRESHOLD && d < bestDistY) {
							bestDistY = d;
							bestDy = otherE - myE;
						}
					}
				}
				for (const gap of H_GAPS) {
					const xPairs: [number, number][] = [
						[me.left, oe.left],
						[me.right, oe.right],
						[me.left, oe.right + gap],
						[me.right, oe.left - gap]
					];
					if (gap === 0) xPairs.push([me.cx, oe.cx]);
					for (const [myE, otherE] of xPairs) {
						const d = Math.abs(myE - otherE);
						if (d < SNAP_THRESHOLD && d < bestDistX) {
							bestDistX = d;
							bestDx = otherE - myE;
						}
					}
				}
			}
		}

		const dx = bestDistX < Infinity ? bestDx : 0;
		const dy = bestDistY < Infinity ? bestDy : 0;

		// Guides for visual feedback. Render any edge/center/gap pairing where
		// the post-snap distance is < SNAP_RANGE; opacity fades with distance.
		const guides: Guide[] = [];
		for (const me of draggedBounds) {
			const shifted: Bounds = {
				left: me.left + dx,
				right: me.right + dx,
				top: me.top + dy,
				bottom: me.bottom + dy,
				cx: me.cx + dx,
				cy: me.cy + dy
			};
			for (const oe of targets) {
				for (const gap of V_GAPS) {
					const yPairs: [number, number][] = [
						[shifted.top, oe.top],
						[shifted.bottom, oe.bottom],
						[shifted.top, oe.bottom + gap],
						[shifted.bottom, oe.top - gap]
					];
					if (gap === 0) yPairs.push([shifted.cy, oe.cy]);
					for (const [myE, otherE] of yPairs) {
						const d = Math.abs(myE - otherE);
						if (d < SNAP_RANGE) {
							guides.push({ y: otherE, opacity: d < 0.5 ? 1 : 1 - d / SNAP_RANGE });
						}
					}
				}
				for (const gap of H_GAPS) {
					const xPairs: [number, number][] = [
						[shifted.left, oe.left],
						[shifted.right, oe.right],
						[shifted.left, oe.right + gap],
						[shifted.right, oe.left - gap]
					];
					if (gap === 0) xPairs.push([shifted.cx, oe.cx]);
					for (const [myE, otherE] of xPairs) {
						const d = Math.abs(myE - otherE);
						if (d < SNAP_RANGE) {
							guides.push({ x: otherE, opacity: d < 0.5 ? 1 : 1 - d / SNAP_RANGE });
						}
					}
				}
			}
		}

		return { dx, dy, guides };
	}

	function onNodeDragStart(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		dragStartPositions = new Map();
		for (const n of args.nodes) dragStartPositions.set(n.id, { x: n.position.x, y: n.position.y });
	}

	function onNodeDrag(args: { nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		const current = new Map<string, { x: number; y: number }>();
		for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
		const alt = (args.event as MouseEvent).altKey === true;
		const { dx, dy, guides } = computeSnapDelta(current, alt);
		snapGuides = guides;
		// Visibly preview the snap: each dragged node gets the same delta as
		// a CSS translate via the ui store. GoofiNode applies it on the
		// inner element so SvelteFlow's outer transform (which tracks the
		// raw mouse position) stays intact.
		if (dx === 0 && dy === 0) {
			uiStore.dragSnap = {};
		} else {
			const next: Record<string, { dx: number; dy: number }> = {};
			for (const n of args.nodes) next[n.id] = { dx, dy };
			uiStore.dragSnap = next;
		}
	}

	function onNodeDragStop(args: { targetNode: Node | null; nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		const current = new Map<string, { x: number; y: number }>();
		for (const n of args.nodes) current.set(n.id, { x: n.position.x, y: n.position.y });
		const alt = (args.event as MouseEvent).altKey === true;
		const { dx, dy } = computeSnapDelta(current, alt);
		for (const n of args.nodes) {
			void g.setNodePos(n.id, [Math.round(n.position.x + dx), Math.round(n.position.y + dy)]);
		}
		snapGuides = [];
		dragStartPositions = new Map();
		uiStore.dragSnap = {};
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
		if (args.event.shiftKey) return;
		selection = new Set();
		edgeSelection = new Set();
	}

	function onNodeClick(args: { node: Node; event: MouseEvent | TouchEvent }): void {
		const id = args.node.id;
		const mouse = args.event as MouseEvent;
		if (mouse.shiftKey || mouse.ctrlKey || mouse.metaKey) {
			const next = new Set(selection);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			selection = next;
		} else {
			selection = new Set([id]);
		}
	}

	const nodeTypes = { goofi: GoofiNode };

	// Keyboard shortcuts (handled at window level so they work without
	// the editor having focus).
	function onKeydown(e: KeyboardEvent): void {
		const tag = (e.target as HTMLElement | null)?.tagName ?? '';
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

		const meta = e.ctrlKey || e.metaKey;
		if (meta && e.key.toLowerCase() === 's') {
			e.preventDefault();
			void triggerSave();
		} else if (meta && e.key.toLowerCase() === 'o') {
			e.preventDefault();
			triggerLoad();
		} else if (meta && e.key.toLowerCase() === 'a') {
			e.preventDefault();
			selection = new Set(g.nodes.map((n) => n.name));
		} else if (meta && e.key.toLowerCase() === 'c') {
			void copySelection();
		} else if (meta && e.key.toLowerCase() === 'v') {
			void pasteClipboard();
		} else if (e.key === 'Delete' || e.key === 'Backspace') {
			if (selection.size === 0 && edgeSelection.size === 0) return;
			e.preventDefault();
			if (selection.size > 0) void deleteSelection();
			if (edgeSelection.size > 0) void deleteEdgeSelection();
		} else if (e.key === 'Tab' || (e.key === ' ' && !menuOpen && e.shiftKey === false)) {
			if (e.key === 'Tab') {
				e.preventDefault();
				openMenuAtCursor();
			}
		} else if (e.key === 'Escape') {
			// Escape closes any open menu *and* clears selection.
			if (menuOpen) {
				menuOpen = false;
				menuSeed = null;
			} else if (selection.size > 0 || edgeSelection.size > 0) {
				selection = new Set();
				edgeSelection = new Set();
			}
		} else if (e.key.toLowerCase() === 'f') {
			document
				.querySelector<HTMLButtonElement>('.svelte-flow__controls-fitview')
				?.click();
		}
	}

	async function deleteSelection(): Promise<void> {
		const names = Array.from(selection);
		for (const n of names) {
			try {
				await g.removeNode(n);
			} catch (e) {
				console.warn('remove failed', e);
			}
		}
		selection = new Set();
	}

	async function triggerSave(): Promise<void> {
		try {
			const { yaml, path } = await g.save(undefined, true);
			const blob = new Blob([yaml], { type: 'application/x-yaml' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = path.split('/').pop() ?? 'patch.gfi';
			a.click();
			setTimeout(() => URL.revokeObjectURL(url), 1000);
		} catch (e) {
			console.error('save failed', e);
		}
	}

	function triggerLoad(): void {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.gfi,.yaml,.yml';
		input.onchange = async () => {
			const f = input.files?.[0];
			if (!f) return;
			const content = await f.text();
			try {
				await g.loadText(content);
			} catch (e) {
				console.error('load failed', e);
			}
		};
		input.click();
	}

	async function copySelection(): Promise<void> {
		const sel = g.nodes.filter((n) => selection.has(n.name));
		if (sel.length === 0) return;
		const avg = sel.reduce(
			(acc, n) => [acc[0] + n.pos[0], acc[1] + n.pos[1]],
			[0, 0]
		);
		const avgX = avg[0] / sel.length;
		const avgY = avg[1] / sel.length;
		const links = g.links.filter(
			(l) => selection.has(l.node_in) && selection.has(l.node_out)
		);
		const payload = {
			__goofi_clip__: 1,
			nodes: sel.map((n) => ({
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
			nodes: { name: string; type: string; category: string; params: Record<string, Record<string, unknown>>; offset: [number, number] }[];
			links: LinkInfo[];
		};
		try {
			payload = JSON.parse(text);
		} catch {
			return;
		}
		if (payload?.__goofi_clip__ !== 1) return;
		// Drop new copies near the current screen-center of the canvas.
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
				// Re-apply param values via individual update calls
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
		menuOpen = true;
	}

	/** After auto-link insertion: connect the seed slot to the matching slot
	 * on the newly-added node, if one exists. */
	async function autoLink(
		seed: SlotClickSeed,
		picked: NodeTypeInfo,
		newName: string
	): Promise<void> {
		// Choose the opposite-side slots of the new node and pick the first
		// dtype match. If none, silently skip.
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

	/** Pick a flow-coordinate position for a freshly-added node. */
	function pickInsertionPos(): [number, number] {
		// Spread successive additions in a small grid so they don't pile up.
		const n = g.nodes.length;
		const col = n % 4;
		const row = Math.floor(n / 4) % 6;
		return [40 + col * 260, 40 + row * 220];
	}

	let mouseX = 0;
	let mouseY = 0;
	function trackMouse(e: MouseEvent): void {
		mouseX = e.clientX;
		mouseY = e.clientY;
	}

	function fitViewClick(): void {
		document.querySelector<HTMLButtonElement>('.svelte-flow__controls-fitview')?.click();
	}

	onMount(() => {
		window.addEventListener('keydown', onKeydown);
		window.addEventListener('mousemove', trackMouse);
		return () => {
			window.removeEventListener('keydown', onKeydown);
			window.removeEventListener('mousemove', trackMouse);
		};
	});

	const selectedNode = $derived(
		selection.size === 1
			? g.nodes.find((n) => n.name === [...selection][0]) ?? null
			: null
	);
</script>

<svelte:head>
	<title
		>{g.unsavedChanges ? '● ' : ''}{g.savePath ? g.savePath.split('/').pop() : 'goofi-pipe'}</title
	>
</svelte:head>

<SvelteFlowProvider>
	<div class="editor-root">
		<TopBar
			onAddNode={() => {
				menuPos = { x: window.innerWidth / 2 - 160, y: 80 };
				menuOpen = !menuOpen;
			}}
			onSave={triggerSave}
			onLoad={triggerLoad}
			onFitView={fitViewClick}
		/>

		<div class="canvas-wrap">
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
						const src = e.source;
						const tgt = e.target;
						const so = e.sourceHandle;
						const si = e.targetHandle;
						if (so && si)
							await g
								.removeLink({ node_out: src, node_in: tgt, slot_out: so, slot_in: si })
								.catch(() => {});
					}
				}}
				fitView
				fitViewOptions={{ maxZoom: 1, padding: 0.18 }}
				minZoom={0.05}
				maxZoom={4}
				initialViewport={{ x: 0, y: 0, zoom: 0.85 }}
			>
				<Controls />
				<MiniMap pannable zoomable />
				{#if snapGuides.length > 0}
					<ViewportPortal target="front">
						<svg class="snap-guides" data-testid="snap-guides">
							{#each snapGuides as g, i (i)}
								{#if g.x !== undefined}
									<line
										x1={g.x}
										x2={g.x}
										y1={-5000}
										y2={5000}
										stroke="var(--accent)"
										stroke-width="1"
										stroke-opacity={g.opacity}
									/>
								{:else if g.y !== undefined}
									<line
										x1={-5000}
										x2={5000}
										y1={g.y}
										y2={g.y}
										stroke="var(--accent)"
										stroke-width="1"
										stroke-opacity={g.opacity}
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
						onPick={async (typeInfo) => {
							const seed = menuSeed;
							menuOpen = false;
							menuSeed = null;
							const pos = pickInsertionPos();
							const newName = await g.addNode(typeInfo.type, typeInfo.category, pos);
							if (seed && newName) await autoLink(seed, typeInfo, newName);
						}}
						onClose={() => {
							menuOpen = false;
							menuSeed = null;
						}}
					/>
				</div>
			{/if}
		</div>

		<aside class="side-panel" class:open={selectedNode !== null}>
			<ParamPanel node={selectedNode} />
			{#if selectedNode}
				<MetadataPanel node={selectedNode} />
				<ErrorPanel
					mode="inline"
					onFocus={(name) => {
						selection = new Set([name]);
					}}
				/>
			{/if}
		</aside>

		<!-- Errors that exist even when nothing is selected get a floating
		     chip in the canvas corner so they never disappear from view. -->
		<ErrorPanel
			mode="chip"
			onFocus={(name) => {
				selection = new Set([name]);
			}}
		/>
	</div>
</SvelteFlowProvider>

<style>
	.editor-root {
		position: fixed;
		inset: 0;
		display: grid;
		grid-template-rows: 44px 1fr;
		grid-template-areas:
			'top'
			'canvas';
	}
	:global(.editor-root > :first-child) {
		grid-area: top;
	}
	.canvas-wrap {
		grid-area: canvas;
		position: relative;
		min-width: 0;
		min-height: 0;
	}
	.side-panel {
		/* Floating overlay above the canvas — slides in only when a single
		   node is selected. Canvas keeps full width underneath. */
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		width: min(420px, 33vw);
		background: color-mix(in srgb, var(--bg-elev-1) 96%, transparent);
		backdrop-filter: blur(8px);
		border-left: 1px solid var(--border);
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		min-width: 0;
		transform: translateX(100%);
		transition: transform 180ms ease;
		box-shadow: -8px 0 24px rgba(0, 0, 0, 0.35);
		z-index: 50;
	}
	.side-panel.open {
		transform: translateX(0);
	}
	.menu-overlay {
		position: fixed;
		inset: 0;
		z-index: 99;
	}
	.menu-anchor {
		position: fixed;
		z-index: 100;
		width: 320px;
	}
	.snap-guides {
		/* Lives inside SvelteFlow's viewport (via ViewportPortal), so SVG
		   coords are flow-space. Render at (0,0) with overflow:visible so
		   lines extending into negative space stay drawn. */
		position: absolute;
		left: 0;
		top: 0;
		width: 1px;
		height: 1px;
		overflow: visible;
		pointer-events: none;
	}
</style>

