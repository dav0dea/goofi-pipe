/**
 * Central reactive graph state, backed by the control WS.
 *
 * Uses Svelte 5 runes — components subscribe by importing this store and
 * reading its `$state` fields directly. The store owns the only writes
 * (driven by control events) so consumers never have to merge.
 */
import {
	getControl,
	paramValues,
	sameLink,
	type Control,
	type ControlEvent,
	type DirListing,
	type FsEntry,
	type GraphSnapshot,
	type InstanceInfo,
	type LinkInfo,
	type NodeInstanceInfo,
	type NodeTypeInfo,
	type SubPatchPort
} from '$lib/api/control';
import { ui } from './ui.svelte';
import { consoleStore } from './console.svelte';
import { selection } from './selection.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { seedInlineView, forgetInlineView, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { resolveKind } from '$lib/viewers/kind';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import type { ViewerKind } from '$lib/viewers/kind';
import { history, type Action, type ExprState } from './history.svelte';
import { captureNavContext } from '$lib/workspace/navContext';

export class GraphStore {
	nodes = $state<NodeInstanceInfo[]>([]);
	links = $state<LinkInfo[]>([]);
	/** Sub-patch instances keyed by instance id (flatten-at-runtime group nodes). */
	instances = $state<Record<string, InstanceInfo>>({});
	savePath = $state<string | null>(null);
	unsavedChanges = $state(false);
	connected = $state(false);
	hadHello = $state(false);

	/** Bumps on every *wholesale* graph load (hello / graph_replaced), never on
	 * incremental node add/remove. Editor panels watch it to fit the view to a
	 * freshly-loaded patch — without auto-fitting when nodes are placed one by
	 * one (which is what made the first interactively-placed node jump). */
	loadEpoch = $state(0);

	nodeTypes = $state<NodeTypeInfo[] | null>(null);

	/** instance_id of the manager process we last hydrated from. A change means
	 * the backend was restarted under our still-open tab — a fresh session,
	 * not a transient reconnect — so a layout-less snapshot must reset the
	 * layout instead of preserving the previous session's arrangement. */
	private _lastInstanceId: string | null = null;

	/** The control client (injectable for tests; defaults to the live WS one). */
	private ctl: Control;

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.onConnect((c) => (this.connected = c));
		ctl.on((ev) => this._handle(ev));
	}

	/** Apply a wholesale snapshot. Returns whether it came from a *new* backend
	 * session (changed `instance_id`) — the caller uses that to decide whether to
	 * re-fit / clear history (a same-session reconnect must leave both alone). */
	private _replaceSnapshot(snap: GraphSnapshot): boolean {
		// Drop ui bookkeeping for any node that's about to disappear, then
		// re-seed viewer state (collapse / kind / settings) for every node.
		for (const old of this.nodes) {
			ui().forget(old.name);
			forgetInlineView(old.name);
		}
		for (const n of snap.nodes) this._seedNodeViewerState(n);
		this.nodes = snap.nodes;
		this.links = snap.links;
		this.instances = snap.instances ?? {};
		this.savePath = snap.save_path;
		this.unsavedChanges = snap.unsaved_changes;

		// Layout resolution. A snapshot carrying a layout always drives the panel
		// arrangement (a patch loaded from disk, or the manager echoing what we
		// pushed). A layout-less snapshot is ambiguous: from a *new* backend it
		// means "blank session" (reset to default); from the *same* backend it's
		// a transient reconnect whose layout already matches ours (keep it).
		const freshSession = snap.instance_id !== this._lastInstanceId;
		this._lastInstanceId = snap.instance_id;
		if (snap.layout != null) {
			workspace().hydrate(snap.layout);
		} else if (freshSession) {
			workspace().reset();
		}
		return freshSession;
	}

	/** React to a wholesale graph load (a new backend session, or a patch loaded
	 * via the Load button). Fit the view (loadEpoch) and drop the error-console
	 * history, which belongs to the graph that just went away — without this it
	 * would show errors for nodes that no longer exist and could even merge a
	 * reused node name's count across sessions. */
	private _onWholesaleLoad(): void {
		this.loadEpoch += 1;
		consoleStore().clear();
		// Drop stale per-panel selection/inspector state: a loaded layout keeps
		// its saved panel ids, which can collide with ids used earlier this
		// session and silently apply old state to the new panels. (A transient
		// same-session reconnect doesn't come here, so its selection survives.)
		selection().forgetAll();
	}

	/** Seed collapse + kind + settings for a node's slots from its (possibly
	 * restored) `viewers` map. */
	private _seedNodeViewerState(node: NodeInstanceInfo): void {
		const slots = Object.keys(node.output_slots);
		ui().seedNodeViewers(node.name, slots, node.viewers);
		for (const slot of slots) {
			const v = node.viewers?.[slot];
			seedInlineView(node.name, slot, {
				kind: v?.kind as ViewerKind | undefined,
				settings: v?.settings as SettingsMap | undefined
			});
		}
	}

	private _viewerPushTimers = new Map<string, ReturnType<typeof setTimeout>>();
	/** Debounced push of a node's full viewer state (collapse / kind / settings)
	 * to the backend, so it round-trips into the .gfi on save. Soft UI state — the
	 * bridge op deliberately doesn't mark the patch unsaved. */
	pushNodeViewers(node: string): void {
		clearTimeout(this._viewerPushTimers.get(node));
		this._viewerPushTimers.set(
			node,
			setTimeout(() => {
				this._viewerPushTimers.delete(node);
				const n = this.nodeByName(node);
				if (!n) return;
				const viewers: Record<string, { collapsed: boolean; kind: string; settings: SettingsMap }> = {};
				for (const slot of Object.keys(n.output_slots)) {
					const view = rawInlineView(node, slot);
					viewers[slot] = {
						collapsed: !ui().isSlotExpanded(node, slot),
						kind: resolveKind(n.output_slots[slot], view.kind),
						settings: view.settings
					};
				}
				void this.ctl.call('set_node_viewers', { node, viewers }).catch(() => {});
			}, 250)
		);
	}

	private _handle(ev: ControlEvent): void {
		switch (ev.event) {
			case 'hello': {
				const fresh = this._replaceSnapshot(ev.payload);
				this.hadHello = true;
				// A reconnect to the *same* backend must not re-fit the view or wipe
				// the error history; a new backend session (or first connect) should.
				if (fresh) this._onWholesaleLoad();
				void this._refreshNodeTypes();
				break;
			}
			case 'graph_replaced':
				// A patch was loaded/replaced wholesale — always re-fit + reset history.
				this._replaceSnapshot(ev.payload);
				this._onWholesaleLoad();
				break;
			case 'subpatch_changed': {
				// Group/expand renamed members + rewrote instances. Re-sync nodes,
				// links, and instances from the fresh snapshot — but NOT layout or a
				// re-fit (it's an in-place edit, not a wholesale load).
				const snap = ev.payload;
				// Names present before vs after — group/expand/remove_instance fire
				// only this event (no per-node node_removed), so prune panel links to
				// any node OR instance id that vanished. Critical: instance ids are
				// reused (subpatch0 regenerates), so a stale link would silently
				// re-bind a panel to an unrelated sub-patch.
				const before = new Set<string>([
					...this.nodes.map((n) => n.name),
					...Object.keys(this.instances)
				]);
				const after = new Set<string>([
					...snap.nodes.map((n) => n.name),
					...Object.keys(snap.instances ?? {})
				]);
				for (const name of before) {
					if (!after.has(name)) workspace().clearNodeRefs(name);
				}
				for (const old of this.nodes) {
					forgetInlineView(old.name);
				}
				for (const n of snap.nodes) this._seedNodeViewerState(n);
				this.nodes = snap.nodes;
				this.links = snap.links;
				this.instances = snap.instances ?? {};
				break;
			}
			case 'boundary_moved': {
				const inst = this.instances[ev.payload.inst_id];
				const port = inst?.interface?.[ev.payload.bnd_id];
				if (port) port.pos = ev.payload.pos;
				break;
			}
			case 'node_renamed': {
				// Light rename fix-up (subpatch_changed re-syncs authoritatively right
				// after, but keep selection/refs coherent in the meantime).
				const { old, new: nu } = ev.payload;
				const t = this.nodeByName(old);
				if (t) t.name = nu;
				for (const l of this.links) {
					if (l.node_out === old) l.node_out = nu;
					if (l.node_in === old) l.node_in = nu;
				}
				break;
			}
			case 'node_added':
				// Seed view state for this node's output slots — from the saved
				// patch (`viewers`) if present, else the defaults in the stores.
				this._seedNodeViewerState(ev.payload);
				this.nodes = [...this.nodes.filter((n) => n.name !== ev.payload.name), ev.payload];
				break;
			case 'node_removed':
				this.nodes = this.nodes.filter((n) => n.name !== ev.payload.name);
				this.links = this.links.filter(
					(l) => l.node_in !== ev.payload.name && l.node_out !== ev.payload.name
				);
				ui().forget(ev.payload.name);
				forgetInlineView(ev.payload.name);
				consoleStore().forgetNodeDedup(ev.payload.name);
				// Empty any Parameters/Viewer/Metadata panel linked to this node.
				workspace().clearNodeRefs(ev.payload.name);
				break;
			case 'node_moved': {
				// Instances first: nodeByName now synthesizes a throwaway node for an
				// instance id, so checking it first would write pos to a discarded
				// object and the group node would snap back.
				if (this.instances[ev.payload.name]) {
					this.instances[ev.payload.name].pos = ev.payload.pos;
				} else {
					const target = this._realNode(ev.payload.name);
					if (target) target.pos = ev.payload.pos;
				}
				break;
			}
			case 'link_added':
				if (!this.links.some((l) => sameLink(l, ev.payload))) {
					this.links = [...this.links, ev.payload];
				}
				break;
			case 'link_removed':
				this.links = this.links.filter((l) => !sameLink(l, ev.payload));
				break;
			case 'state_update': {
				const t = this.nodeByName(ev.payload.node);
				if (t) {
					t.params = ev.payload.params;
					// The node advertises its SSE log endpoint here; surfacing it lets
					// the Console subscribe peer-to-peer (see $lib/stores/logStream).
					if (ev.payload.log_endpoint !== undefined) t.log_endpoint = ev.payload.log_endpoint;
				}
				break;
			}
			case 'error': {
				// Live snapshot — drives the node's red border, the floating error
				// chip, and the inspector's current-error section. Always on, via the
				// control plane, independent of whether a Console is open.
				const t = this.nodeByName(ev.payload.node);
				if (t) t.error = ev.payload.error;
				// Mirror the active error into the Console as a stderr entry so the
				// log view is the complete picture (repeats coalesce into one ×N).
				if (ev.payload.error) consoleStore().ingestError(ev.payload.node, ev.payload.error, Date.now());
				break;
			}
			case 'unsaved_changes':
				this.unsavedChanges = ev.payload.unsaved_changes;
				break;
			case 'save_path_changed':
				this.savePath = ev.payload.save_path;
				break;
			case 'layout':
				// A patch finished loading after we connected — e.g. CLI startup
				// (`goofi-pipe x.gfi`), whose nodes arrive as node_added events with
				// no graph_replaced snapshot. Hydrate its layout if any (null → keep
				// ours), then fit the view to the freshly-loaded graph. No history
				// reset here: it's the same session, and any load-time errors are
				// the just-loaded patch's own.
				if (ev.payload.layout != null) {
					workspace().hydrate(ev.payload.layout);
					// New layout's panel ids may collide with ones used earlier
					// this session — drop stale per-panel state (see _onWholesaleLoad).
					selection().forgetAll();
				}
				this.loadEpoch += 1;
				break;
			case 'manager_shutdown':
				this.connected = false;
				break;
		}
	}

	private async _refreshNodeTypes(): Promise<void> {
		try {
			const result = await this.ctl.call<{ types: NodeTypeInfo[] }>('list_nodes');
			this.nodeTypes = result.types;
		} catch (e) {
			console.warn('list_nodes failed', e);
		}
	}

	// ------------------------------------------------------------------
	// mutations (sent via control RPC; UI updates apply on response)
	// ------------------------------------------------------------------

	/** Push an action onto the history (unless a replay is in progress). */
	private _record(action: Action): void {
		if (!history().isSuspended) history().record(action);
	}

	async addNode(
		type: string,
		category: string,
		pos: [number, number],
		instId?: string
	): Promise<string> {
		// `instId` lands the node inside that sub-patch (member of the instance);
		// omitted, it goes in the root graph.
		const name =
			(await this.ctl.call<string>('add_node', { type, category, pos, inst_id: instId })) ?? '';
		// Record AFTER the call so we know the backend-assigned display name.
		if (name)
			this._record({
				kind: 'add_node',
				label: `Add ${type}`,
				domain: 'graph',
				context: captureNavContext(),
				payload: { type, category, pos, instId, assignedName: name }
			});
		return name;
	}

	async removeNode(name: string): Promise<void> {
		// Capture the full node + its links BEFORE the backend tears them down.
		const node = this.nodeByName(name);
		if (node && !history().isSuspended) {
			const links = this.links
				.filter((l) => l.node_in === name || l.node_out === name)
				.map((l) => ({ ...l }));
			this._record({
				kind: 'remove_node',
				label: `Delete ${name}`,
				domain: 'graph',
				context: captureNavContext(),
				payload: {
					name,
					node: structuredClone($state.snapshot(node)),
					links,
					membership: node.membership ?? null,
					boundPanels: workspace().panelsBoundTo(name)
				}
			});
		}
		await this.ctl.call('remove_node', { name });
	}

	async addLink(link: LinkInfo): Promise<void> {
		// The single-source rule may displace an existing wire on the input —
		// capture it before the add so undo can restore it.
		const displaced =
			this.links.find((l) => l.node_in === link.node_in && l.slot_in === link.slot_in) ?? null;
		this._record({
			kind: 'add_link',
			label: 'Connect',
			domain: 'graph',
			context: captureNavContext(),
			payload: { link: { ...link }, displaced: displaced ? { ...displaced } : null }
		});
		await this.ctl.call('add_link', link as unknown as Record<string, unknown>);
	}

	async removeLink(link: LinkInfo): Promise<void> {
		this._record({
			kind: 'remove_link',
			label: 'Disconnect',
			domain: 'graph',
			context: captureNavContext(),
			payload: { link: { ...link } }
		});
		await this.ctl.call('remove_link', link as unknown as Record<string, unknown>);
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		const oldValue = this.nodeByName(node)?.params?.[group]?.[name]?.value;
		this._record({
			kind: 'update_param',
			label: `Set ${name}`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { node, group, name, oldValue, newValue: value }
		});
		await this.ctl.call('update_param', { node, group, name, value });
	}

	async setExpression(
		node: string,
		group: string,
		name: string,
		expression: string | null,
		opts: { enabled?: boolean; triggers_process?: boolean; autoeval?: boolean } = {}
	): Promise<void> {
		const d = this.nodeByName(node)?.params?.[group]?.[name];
		const oldExpr: ExprState = {
			expression: d?.expression ?? null,
			enabled: d?.expression_enabled ?? false,
			triggers_process: d?.expression_triggers_process ?? false,
			autoeval: d?.expression_autoeval ?? false
		};
		const newExpr: ExprState = {
			expression,
			enabled: opts.enabled ?? false,
			triggers_process: opts.triggers_process ?? false,
			autoeval: opts.autoeval ?? false
		};
		this._record({
			kind: 'set_expression',
			label: `Set ${name} expression`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { node, group, name, oldExpr, newExpr }
		});
		await this.ctl.call('set_expression', {
			node,
			group,
			name,
			expression,
			expression_enabled: newExpr.enabled,
			expression_triggers_process: newExpr.triggers_process,
			expression_autoeval: newExpr.autoeval
		});
	}

	async setNodePos(name: string, pos: [number, number]): Promise<void> {
		const oldPos = this.nodeByName(name)?.pos ?? [0, 0];
		this._record({
			kind: 'set_node_pos',
			label: `Move ${name}`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { name, oldPos: [oldPos[0], oldPos[1]], newPos: pos }
		});
		await this.ctl.call('set_node_pos', { name, pos });
	}

	/** Push the current workspace layout into the running patch (manager memory)
	 * so it survives reloads and lands in the .gfi on save. Fire-and-forget:
	 * layout is soft UI state, so a dropped push is harmless. */
	async setLayout(layout: unknown): Promise<void> {
		try {
			await this.ctl.call('set_layout', { layout });
		} catch {
			/* not connected / in flight — ignore */
		}
	}

	async save(
		path?: string,
		overwrite = false,
		layout?: unknown
	): Promise<{ path: string; yaml: string }> {
		// `layout` is the frontend workspace arrangement; the backend writes it
		// into the .gfi. Omitted (undefined) → not sent → backend keeps any
		// existing layout.
		return this.ctl.call('save', { path, overwrite, layout });
	}

	async loadText(content: string): Promise<void> {
		await this.ctl.call('load_text', { content });
	}

	/** Group the named nodes into a unique (inline) sub-patch. Returns its instance id. */
	async groupNodes(members: string[], pos?: [number, number]): Promise<string> {
		const r = await this.ctl.call<{ inst_id: string }>('group_nodes', { members, pos });
		if (r?.inst_id)
			this._record({
				kind: 'group_nodes',
				label: 'Group nodes',
				domain: 'graph',
				context: captureNavContext(),
				payload: { members: [...members], instId: r.inst_id, pos }
			});
		return r.inst_id;
	}

	/** Dissolve a sub-patch instance back into its member nodes. */
	async expandInstance(instId: string): Promise<void> {
		const iface = structuredClone($state.snapshot(this.instances[instId]?.interface ?? {}));
		const r = await this.ctl.call<{ restored: string[] }>('expand_instance', { inst_id: instId });
		this._record({
			kind: 'expand_instance',
			label: 'Ungroup',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, restoredMembers: r?.restored ?? [], interface: iface }
		});
	}

	/** Promote an instance to shared and spawn a strict-mirror sibling copy. */
	async duplicateShared(instId: string, pos?: [number, number]): Promise<void> {
		const wasUnique = !this.instances[instId]?.def_id;
		const r = await this.ctl.call<{ inst_id: string }>('duplicate_shared', { inst_id: instId, pos });
		this._record({
			kind: 'duplicate_shared',
			label: 'Duplicate shared',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, newInstId: r?.inst_id ?? '', wasUnique, pos }
		});
	}

	/** Detach a shared instance into its own private (unique) copy. */
	async makeUnique(instId: string): Promise<void> {
		const defIdBefore = this.instances[instId]?.def_id ?? null;
		this._record({
			kind: 'make_unique',
			label: 'Make unique',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, defIdBefore }
		});
		await this.ctl.call('make_unique', { inst_id: instId });
	}

	/** Add a virtual In/Out boundary node to a sub-patch (unwired). Returns its id. */
	async addBoundary(
		instId: string,
		dir: 'in' | 'out',
		dtype: string,
		pos: [number, number]
	): Promise<string> {
		const r = await this.ctl.call<{ bnd_id: string }>('add_boundary', {
			inst_id: instId,
			dir,
			dtype,
			pos
		});
		if (r?.bnd_id)
			this._record({
				kind: 'add_boundary',
				label: 'Add boundary',
				domain: 'graph',
				context: captureNavContext(),
				payload: { instId, bndId: r.bnd_id, dir, dtype, pos }
			});
		return r.bnd_id;
	}

	/** Set (inner_node/slot) or clear (nulls) a boundary's single inner target. */
	async wireBoundary(
		instId: string,
		bndId: string,
		innerNode: string | null,
		innerSlot: string | null
	): Promise<void> {
		const prev = this.instances[instId]?.interface?.[bndId];
		this._record({
			kind: 'wire_boundary',
			label: 'Wire boundary',
			domain: 'graph',
			context: captureNavContext(),
			payload: {
				instId,
				bndId,
				oldInner: { node: prev?.inner_node ?? null, slot: prev?.inner_slot ?? null },
				newInner: { node: innerNode, slot: innerSlot }
			}
		});
		await this.ctl.call('wire_boundary', {
			inst_id: instId,
			bnd_id: bndId,
			inner_node: innerNode,
			inner_slot: innerSlot
		});
	}

	/** Delete an In/Out boundary node (tears down its external wires). */
	async removeBoundary(instId: string, bndId: string): Promise<void> {
		const port = this.instances[instId]?.interface?.[bndId];
		if (port)
			this._record({
				kind: 'remove_boundary',
				label: 'Remove boundary',
				domain: 'graph',
				context: captureNavContext(),
				payload: { instId, bndId, port: structuredClone($state.snapshot(port)) }
			});
		await this.ctl.call('remove_boundary', { inst_id: instId, bnd_id: bndId });
	}

	/** Move an In/Out pill inside the entered view (mirrors across shared siblings). */
	async setBoundaryPos(instId: string, bndId: string, pos: [number, number]): Promise<void> {
		const old = this.instances[instId]?.interface?.[bndId]?.pos ?? [0, 0];
		this._record({
			kind: 'set_boundary_pos',
			label: 'Move boundary',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, bndId, oldPos: [old[0], old[1]], newPos: pos }
		});
		await this.ctl.call('set_boundary_pos', { inst_id: instId, bnd_id: bndId, pos });
	}

	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return this.ctl.call<DirListing>('list_dir', { path });
	}

	/** The bundled example patches (empty under a wheel without examples/). */
	async listExamples(): Promise<{ entries: FsEntry[] }> {
		return this.ctl.call<{ entries: FsEntry[] }>('list_examples');
	}

	/** Load a patch from a BACKEND filesystem path (destructive — replaces the graph). */
	async load(path: string): Promise<void> {
		await this.ctl.call('load', { path });
	}

	/** Current patch as `.gfi` YAML, without writing to disk (for browser download). */
	async serialize(): Promise<{ yaml: string }> {
		return this.ctl.call<{ yaml: string }>('serialize');
	}
	// ------------------------------------------------------------------
	// reads
	// ------------------------------------------------------------------

	/** A real node by name (no sub-patch synthesis), or null. */
	private _realNode(name: string): NodeInstanceInfo | null {
		return this.nodes.find((n) => n.name === name) ?? null;
	}

	/** Resolve a node by name — the one accessor the rest of the app and the
	 * agent surface use. A sub-patch instance id resolves to a *virtual* node
	 * carrying a `subpatch` marker, so selection / inspector / drag treat a
	 * sub-patch exactly like a node (no node class is instantiated). */
	nodeByName(name: string): NodeInstanceInfo | null {
		const real = this._realNode(name);
		if (real) return real;
		const inst = this.instances[name];
		return inst ? this._synthSubpatchNode(name, inst) : null;
	}

	/** Other instance ids mirroring the same definition as `instId` (strict
	 * mirror siblings); empty for a unique sub-patch. */
	subpatchSiblings(instId: string): string[] {
		const def = this.instances[instId]?.def_id;
		if (!def) return [];
		return Object.keys(this.instances).filter(
			(id) => id !== instId && this.instances[id].def_id === def
		);
	}

	/** The first errored member of a sub-patch (its error string), or null. Lets
	 * the collapsed group node and its inspector surface a member's error even
	 * though the members themselves are hidden. */
	instanceError(instId: string): string | null {
		const inst = this.instances[instId];
		if (!inst) return null;
		for (const disp of Object.keys(inst.members)) {
			const err = this._realNode(disp)?.error;
			if (err) return err;
		}
		return null;
	}

	/** The display (graph) name of a member given its local name within `inst`. */
	private _memberDisplay(inst: InstanceInfo, local: string): string | null {
		for (const [disp, loc] of Object.entries(inst.members)) if (loc === local) return disp;
		return null;
	}

	/** A boundary port's data type: the stored dtype (authoritative), falling back
	 * to the wired inner node's slot dtype for legacy interface-only entries. */
	boundaryDtype(inst: InstanceInfo, port: SubPatchPort): string {
		if (port.dtype) return port.dtype;
		const disp = port.inner_node ? this._memberDisplay(inst, port.inner_node) : null;
		const n = disp ? this._realNode(disp) : null;
		if (!n || !port.inner_slot) return 'ARRAY';
		const slots = port.dir === 'in' ? n.input_slots : n.output_slots;
		return slots[port.inner_slot] ?? 'ARRAY';
	}

	/** Build the virtual NodeInstanceInfo that stands in for a sub-patch instance
	 * wherever a node is looked up by name. A sub-patch behaves exactly like a
	 * node: its WIRED boundaries become real input/output slots (so the canvas
	 * renders connectors + output viewers and seeds add-node clicks the same way
	 * as any node — the /data route splices an output boundary to its inner
	 * member's stream). Live sharing state is recomputed by the inspector from
	 * `instances`; the marker just rides the id + glyph/badge bits. */
	private _synthSubpatchNode(instId: string, inst: InstanceInfo): NodeInstanceInfo {
		const shared = Boolean(inst.def_id);
		const input_slots: Record<string, string> = {};
		const output_slots: Record<string, string> = {};
		for (const [bnd, port] of Object.entries(inst.interface)) {
			if (port.inner_node == null) continue; // unwired → not an external port yet
			const dt = this.boundaryDtype(inst, port);
			if (port.dir === 'in') input_slots[bnd] = dt;
			else output_slots[bnd] = dt;
		}
		return {
			name: instId,
			type: shared ? 'Shared sub-patch' : 'Sub-patch',
			category: 'subpatch',
			doc: '',
			input_slots,
			output_slots,
			params: {},
			pos: inst.pos,
			viewers: {},
			membership: null,
			error: this.instanceError(instId),
			subpatch: { instId, shared, memberCount: Object.keys(inst.members).length }
		};
	}

	/** Whether an output slot already feeds a link (drives the canvas "drag to
	 * link vs pop add-menu" decision). */
	isOutputConnected(node: string, slot: string): boolean {
		return this.links.some((l) => l.node_out === node && l.slot_out === slot);
	}

	/** Whether an input slot is already fed by a link. */
	isInputConnected(node: string, slot: string): boolean {
		return this.links.some((l) => l.node_in === node && l.slot_in === slot);
	}

	// ------------------------------------------------------------------
	// subgraph orchestration (clone / duplicate / paste / batch remove) —
	// shared by the editor's keyboard handlers and the agent command surface
	// ------------------------------------------------------------------

	/** Create a batch of nodes, replay their param values, then wire the given
	 * links with endpoints remapped to the new names. Returns the original→new
	 * name map. Per-node/-param/-link failures are swallowed so one bad item
	 * doesn't abort the batch. */
	async instantiateNodes(
		specs: {
			key: string;
			type: string;
			category: string;
			pos: [number, number];
			params: Record<string, Record<string, unknown>>;
		}[],
		links: LinkInfo[] = []
	): Promise<Record<string, string>> {
		const rename: Record<string, string> = {};
		for (const s of specs) {
			try {
				const newName = await this.addNode(s.type, s.category, s.pos);
				rename[s.key] = newName;
				for (const [group, params] of Object.entries(s.params)) {
					for (const [name, value] of Object.entries(params)) {
						try {
							await this.updateParam(newName, group, name, value);
						} catch {
							/* ignore a single rejected param */
						}
					}
				}
			} catch (e) {
				console.warn('instantiateNodes: add_node failed', e);
			}
		}
		for (const l of links) {
			try {
				await this.addLink({
					node_out: rename[l.node_out] ?? l.node_out,
					node_in: rename[l.node_in] ?? l.node_in,
					slot_out: l.slot_out,
					slot_in: l.slot_in
				});
			} catch {
				/* ignore a link that couldn't be remade */
			}
		}
		return rename;
	}

	/** Duplicate the named nodes (offset from their originals), carrying their
	 * current params and the links among them. Returns the original→new map so
	 * the caller can select the clones. */
	async cloneNodes(
		names: Iterable<string>,
		offset: [number, number] = [40, 40]
	): Promise<Record<string, string>> {
		const set = new Set(names);
		const nodes = this.nodes.filter((n) => set.has(n.name));
		if (nodes.length === 0) return {};
		const links = this.links.filter((l) => set.has(l.node_in) && set.has(l.node_out));
		const specs = nodes.map((n) => ({
			key: n.name,
			type: n.type,
			category: n.category,
			pos: [n.pos[0] + offset[0], n.pos[1] + offset[1]] as [number, number],
			params: paramValues(n)
		}));
		return this.instantiateNodes(specs, links);
	}

	/** Remove a batch of nodes, swallowing per-node failures. */
	async removeNodes(names: Iterable<string>): Promise<void> {
		for (const name of names) {
			try {
				await this.removeNode(name);
			} catch (e) {
				console.warn('remove node failed', e);
			}
		}
	}

	/** Remove a batch of links, swallowing per-link failures. */
	async removeLinks(links: Iterable<LinkInfo>): Promise<void> {
		for (const l of links) {
			try {
				await this.removeLink(l);
			} catch (e) {
				console.warn('remove link failed', e);
			}
		}
	}
}

let _store: GraphStore | null = null;
export function graph(): GraphStore {
	if (!_store) _store = new GraphStore();
	return _store;
}
