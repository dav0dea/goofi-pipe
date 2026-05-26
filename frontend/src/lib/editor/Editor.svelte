<script lang="ts">
	import {
		SvelteFlow,
		Background,
		Controls,
		MiniMap,
		SvelteFlowProvider,
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
	import type { LinkInfo, NodeInstanceInfo } from '$lib/api/control';
	import { onMount } from 'svelte';

	const g = graph();

	let menuOpen = $state(false);
	let menuPos = $state<{ x: number; y: number }>({ x: 120, y: 120 });
	let selection = $state<Set<string>>(new Set());

	// Lift backend nodes/links → Svelte Flow shapes. We rely on $derived
	// to bridge state-store snapshots into SvelteFlow's input arrays.
	let flowNodes = $state.raw<Node[]>([]);
	let flowEdges = $state.raw<Edge[]>([]);

	// expanded-state survives node-list re-syncs across snapshot updates.
	let expandedSet = $state<Set<string>>(new Set());

	$effect(() => {
		const next: Node[] = g.nodes.map((n) => ({
			id: n.name,
			type: 'goofi',
			position: { x: n.pos?.[0] ?? 0, y: n.pos?.[1] ?? 0 },
			data: {
				node: n,
				expanded: expandedSet.has(n.name)
			},
			selected: selection.has(n.name)
		}));
		flowNodes = next;
	});

	$effect(() => {
		const next: Edge[] = g.links.map((l) => ({
			id: edgeId(l),
			source: l.node_out,
			sourceHandle: l.slot_out,
			target: l.node_in,
			targetHandle: l.slot_in,
			animated: false
		}));
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

	function onNodeDragStop(args: { targetNode: Node | null; nodes: Node[]; event: MouseEvent | TouchEvent }): void {
		for (const n of args.nodes) {
			void g.setNodePos(n.id, [Math.round(n.position.x), Math.round(n.position.y)]);
		}
	}

	function onPaneClick(args: { event: MouseEvent }): void {
		menuOpen = false;
		if (args.event.shiftKey) return;
		selection = new Set();
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

	function toggleExpand(name: string): void {
		const next = new Set(expandedSet);
		if (next.has(name)) next.delete(name);
		else next.add(name);
		expandedSet = next;
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
			if (selection.size === 0) return;
			e.preventDefault();
			void deleteSelection();
		} else if (e.key === 'Tab' || (e.key === ' ' && !menuOpen && e.shiftKey === false)) {
			if (e.key === 'Tab') {
				e.preventDefault();
				openMenuAtCursor();
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
				onnodedragstop={onNodeDragStop}
				onpaneclick={onPaneClick}
				onnodeclick={onNodeClick}
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
				snapGrid={[8, 8]}
				defaultViewport={{ x: 0, y: 0, zoom: 0.85 }}
			>
				<Background gap={24} size={1} />
				<Controls />
				<MiniMap pannable zoomable />
			</SvelteFlow>

			{#if menuOpen}
				<div
					class="menu-overlay"
					onclick={() => (menuOpen = false)}
					role="presentation"
				></div>
				<div class="menu-anchor" style="left: {menuPos.x}px; top: {menuPos.y}px">
					<AddNodeMenu
						onPick={async (typeInfo) => {
							menuOpen = false;
							const pos = pickInsertionPos();
							await g.addNode(typeInfo.type, typeInfo.category, pos);
						}}
						onClose={() => (menuOpen = false)}
					/>
				</div>
			{/if}
		</div>

		<aside class="side-panel">
			<ParamPanel node={selectedNode} />
			{#if selectedNode}
				<MetadataPanel node={selectedNode} />
			{/if}
			<ErrorPanel
				onFocus={(name) => {
					selection = new Set([name]);
				}}
			/>
		</aside>
	</div>
</SvelteFlowProvider>

<style>
	.editor-root {
		position: fixed;
		inset: 0;
		display: grid;
		grid-template-columns: 1fr 360px;
		grid-template-rows: 44px 1fr;
		grid-template-areas:
			'top top'
			'canvas side';
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
		grid-area: side;
		border-left: 1px solid var(--border);
		background: var(--bg-elev-1);
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		min-width: 0;
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
</style>

