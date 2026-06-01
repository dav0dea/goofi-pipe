/**
 * Central reactive graph state, backed by the control WS.
 *
 * Uses Svelte 5 runes — components subscribe by importing this store and
 * reading its `$state` fields directly. The store owns the only writes
 * (driven by control events) so consumers never have to merge.
 */
import { getControl, type ControlEvent, type GraphSnapshot, type LinkInfo, type NodeInstanceInfo, type NodeTypeInfo } from '$lib/api/control';
import { ui } from './ui.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';

class GraphStore {
	nodes = $state<NodeInstanceInfo[]>([]);
	links = $state<LinkInfo[]>([]);
	savePath = $state<string | null>(null);
	unsavedChanges = $state(false);
	connected = $state(false);
	hadHello = $state(false);

	nodeTypes = $state<NodeTypeInfo[] | null>(null);

	constructor() {
		const ctl = getControl();
		ctl.onConnect((c) => (this.connected = c));
		ctl.on((ev) => this._handle(ev));
	}

	private _replaceSnapshot(snap: GraphSnapshot): void {
		// Drop ui bookkeeping for any node that's about to disappear, then
		// re-seed viewer-expand state for every node in the new snapshot.
		for (const old of this.nodes) ui().forget(old.name);
		for (const n of snap.nodes) {
			ui().seedNodeViewers(n.name, Object.keys(n.output_slots), n.viewers);
		}
		this.nodes = snap.nodes;
		this.links = snap.links;
		this.savePath = snap.save_path;
		this.unsavedChanges = snap.unsaved_changes;
		// A patch that carries a workspace layout drives the panel arrangement;
		// otherwise keep whatever layout the browser already has.
		if (snap.layout != null) workspace().hydrate(snap.layout);
	}

	private _handle(ev: ControlEvent): void {
		switch (ev.event) {
			case 'hello':
				this._replaceSnapshot(ev.payload);
				this.hadHello = true;
				void this._refreshNodeTypes();
				break;
			case 'graph_replaced':
				this._replaceSnapshot(ev.payload);
				break;
			case 'node_added':
				// Seed expand state for this node's output slots — from
				// the saved patch (`viewers`) if present, otherwise from
				// the default policy in ui.seedNodeViewers.
				ui().seedNodeViewers(
					ev.payload.name,
					Object.keys(ev.payload.output_slots),
					ev.payload.viewers
				);
				this.nodes = [...this.nodes.filter((n) => n.name !== ev.payload.name), ev.payload];
				break;
			case 'node_removed':
				this.nodes = this.nodes.filter((n) => n.name !== ev.payload.name);
				this.links = this.links.filter(
					(l) => l.node_in !== ev.payload.name && l.node_out !== ev.payload.name
				);
				ui().forget(ev.payload.name);
				break;
			case 'node_moved': {
				const target = this.nodes.find((n) => n.name === ev.payload.name);
				if (target) target.pos = ev.payload.pos;
				break;
			}
			case 'link_added':
				if (
					!this.links.some(
						(l) =>
							l.node_out === ev.payload.node_out &&
							l.node_in === ev.payload.node_in &&
							l.slot_out === ev.payload.slot_out &&
							l.slot_in === ev.payload.slot_in
					)
				) {
					this.links = [...this.links, ev.payload];
				}
				break;
			case 'link_removed':
				this.links = this.links.filter(
					(l) =>
						!(
							l.node_out === ev.payload.node_out &&
							l.node_in === ev.payload.node_in &&
							l.slot_out === ev.payload.slot_out &&
							l.slot_in === ev.payload.slot_in
						)
				);
				break;
			case 'state_update': {
				const t = this.nodes.find((n) => n.name === ev.payload.node);
				if (t) t.params = ev.payload.params;
				break;
			}
			case 'error': {
				const t = this.nodes.find((n) => n.name === ev.payload.node);
				if (t) t.error = ev.payload.error;
				break;
			}
			case 'unsaved_changes':
				this.unsavedChanges = ev.payload.unsaved_changes;
				break;
			case 'save_path_changed':
				this.savePath = ev.payload.save_path;
				break;
			case 'layout':
				// Layout restored from a patch loaded after this client connected
				// (e.g. CLI startup load). Null → patch has no layout; keep ours.
				if (ev.payload.layout != null) workspace().hydrate(ev.payload.layout);
				break;
			case 'manager_shutdown':
				this.connected = false;
				break;
		}
	}

	private async _refreshNodeTypes(): Promise<void> {
		try {
			const result = await getControl().call<{ types: NodeTypeInfo[] }>('list_nodes');
			this.nodeTypes = result.types;
		} catch (e) {
			console.warn('list_nodes failed', e);
		}
	}

	// ------------------------------------------------------------------
	// mutations (sent via control RPC; UI updates apply on response)
	// ------------------------------------------------------------------

	async addNode(type: string, category: string, pos: [number, number]): Promise<string> {
		return (await getControl().call<string>('add_node', { type, category, pos })) ?? '';
	}

	async removeNode(name: string): Promise<void> {
		await getControl().call('remove_node', { name });
	}

	async addLink(link: LinkInfo): Promise<void> {
		await getControl().call('add_link', link as unknown as Record<string, unknown>);
	}

	async removeLink(link: LinkInfo): Promise<void> {
		await getControl().call('remove_link', link as unknown as Record<string, unknown>);
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		await getControl().call('update_param', { node, group, name, value });
	}

	async setExpression(
		node: string,
		group: string,
		name: string,
		expression: string | null,
		opts: { enabled?: boolean; triggers_process?: boolean; autoeval?: boolean } = {}
	): Promise<void> {
		await getControl().call('set_expression', {
			node,
			group,
			name,
			expression,
			expression_enabled: opts.enabled ?? false,
			expression_triggers_process: opts.triggers_process ?? false,
			expression_autoeval: opts.autoeval ?? false
		});
	}

	async setNodePos(name: string, pos: [number, number]): Promise<void> {
		await getControl().call('set_node_pos', { name, pos });
	}

	async save(
		path?: string,
		overwrite = false,
		layout?: unknown
	): Promise<{ path: string; yaml: string }> {
		// `layout` is the frontend workspace arrangement; the backend writes it
		// into the .gfi. Omitted (undefined) → not sent → backend keeps any
		// existing layout.
		return getControl().call('save', { path, overwrite, layout });
	}

	async loadText(content: string): Promise<void> {
		await getControl().call('load_text', { content });
	}

	async listExamples(): Promise<{ examples: { name: string; size: number }[] }> {
		return getControl().call('list_examples');
	}

	async loadExample(name: string): Promise<void> {
		await getControl().call('load_example', { name });
	}
}

let _store: GraphStore | null = null;
export function graph(): GraphStore {
	if (!_store) _store = new GraphStore();
	return _store;
}
