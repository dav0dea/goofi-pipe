/**
 * The browser replica of goofi's control-plane document — the exact schema the Rust
 * `goofi_crdt::GraphDoc` mirrors: `nodes: Y.Map<uid, {type, name, pos:{x,y}, params:
 * Y.Map<group, Y.Map<name, {value, expr?}>>, viewers}>` and `links: Y.Array<{node_out,
 * slot_out, node_in, slot_in}>`.
 *
 * Pure reader helpers over a `Y.Doc` (no Svelte, no WebSocket) so they unit-test directly.
 * The reactive `.svelte.ts` layer subscribes to the doc and re-exposes these as runes.
 */
import * as Y from 'yjs';

export interface NodeView {
	uid: string;
	type: string;
	name: string;
	pos: [number, number];
}

export interface LinkView {
	node_out: string;
	slot_out: string;
	node_in: string;
	slot_in: string;
}

export interface BoundaryView {
	bnd_id: string;
	dir: string; // 'in' | 'out'
	dtype: string;
	name: string;
	pos: [number, number];
	inner_node?: string;
	inner_slot?: string;
}

export interface InstanceView {
	uid: string;
	name: string;
	/** Shared def id, or `undefined` when the instance is unique. */
	def_id?: string;
	/** Parent scope uid, or `'__root__'` for a top-level instance. */
	parent: string;
	pos: [number, number];
	/** local name → live member uid. */
	members: Record<string, string>;
	interface: BoundaryView[];
}

/** The `nodes` root map. */
export function nodesMap(doc: Y.Doc): Y.Map<Y.Map<unknown>> {
	return doc.getMap('nodes') as Y.Map<Y.Map<unknown>>;
}

/** The `links` root array. */
export function linksArray(doc: Y.Doc): Y.Array<Y.Map<unknown>> {
	return doc.getArray('links') as Y.Array<Y.Map<unknown>>;
}

function str(m: Y.Map<unknown> | undefined, key: string): string {
	const v = m?.get(key);
	return typeof v === 'string' ? v : '';
}

/** A node's identity view, or `null` if the uid is absent. */
export function nodeView(doc: Y.Doc, uid: string): NodeView | null {
	const n = nodesMap(doc).get(uid);
	if (!n) return null;
	const pos = n.get('pos') as Y.Map<unknown> | undefined;
	const num = (k: string) => {
		const v = pos?.get(k);
		return typeof v === 'number' ? v : 0;
	};
	return { uid, type: str(n, 'type'), name: str(n, 'name'), pos: [num('x'), num('y')] };
}

/** All node identity views, in the doc's key order. */
export function nodeViews(doc: Y.Doc): NodeView[] {
	const out: NodeView[] = [];
	for (const uid of nodesMap(doc).keys()) {
		const v = nodeView(doc, uid);
		if (v) out.push(v);
	}
	return out;
}

/** A single param's current value (`number | string | boolean`), or `undefined`. */
export function paramValue(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string
): number | string | boolean | undefined {
	const entry = paramEntry(doc, uid, group, name);
	const v = entry?.get('value');
	return typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean' ? v : undefined;
}

export interface ParamExpr {
	source: string;
	enabled: boolean;
	triggers: boolean;
}

/** The doc-owned param leaves for one node: value + optional expression binding, per group/name.
 * Exactly the `{value, expr?}` structure the doc stores (nodeAssembly merges these with the catalog
 * descriptor + runtime overlay). */
export type DocParamLeaves = Record<
	string,
	Record<string, { value?: number | string | boolean; expr?: ParamExpr }>
>;

/** Read a node's committed param leaves (value + expression binding) from the doc, per group/name. */
export function docParams(doc: Y.Doc, uid: string): DocParamLeaves {
	const out: DocParamLeaves = {};
	const params = nodesMap(doc).get(uid)?.get('params') as
		| Y.Map<Y.Map<Y.Map<unknown>>>
		| undefined;
	if (!params) return out;
	for (const [group, g] of params.entries()) {
		out[group] = {};
		for (const [name, entry] of g.entries()) {
			const leaf: DocParamLeaves[string][string] = {};
			const v = entry.get('value');
			if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') leaf.value = v;
			const expr = entry.get('expr') as Y.Map<unknown> | undefined;
			const s = expr?.get('source');
			if (typeof s === 'string') {
				leaf.expr = { source: s, enabled: expr!.get('enabled') === true, triggers: expr!.get('triggers') === true };
			}
			out[group][name] = leaf;
		}
	}
	return out;
}

function paramEntry(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string
): Y.Map<unknown> | undefined {
	const params = nodesMap(doc).get(uid)?.get('params') as Y.Map<Y.Map<unknown>> | undefined;
	const g = params?.get(group) as Y.Map<Y.Map<unknown>> | undefined;
	return g?.get(name) as Y.Map<unknown> | undefined;
}

/** Write a committed param value into the doc IN PLACE, matching the Rust `GraphDoc::set_param`
 * structure (`nodes[uid].params[group][name].value`) so the manager's `apply_client_update`
 * detects and applies it. A client leaf-write (§4.1). No-op if the node is not in this replica
 * yet (never mint a phantom node — the diff would create a conflicting node on the manager).
 * Returns whether the write landed. */
export function setParamValue(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string,
	value: number | string | boolean
): boolean {
	const entry = paramEntryForWrite(doc, uid, group, name);
	if (!entry) return false;
	if (entry.get('value') !== value) entry.set('value', value);
	return true;
}

/** Get-or-insert the (stable) `Y.Map` at `parent[key]`. Never replaces an existing map, so nested
 * leaves survive. */
function getOrInsertMap(parent: Y.Map<unknown>, key: string): Y.Map<unknown> {
	let m = parent.get(key) as Y.Map<unknown> | undefined;
	if (!m) {
		m = new Y.Map<unknown>();
		parent.set(key, m);
	}
	return m;
}

/** The `nodes[uid].params[group][name]` entry map, get-or-inserted, or `undefined` if the node is
 * absent from this replica (never mint a phantom node — the diff would create a conflicting node). */
function paramEntryForWrite(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string
): Y.Map<unknown> | undefined {
	const node = nodesMap(doc).get(uid);
	if (!node) return undefined;
	return getOrInsertMap(getOrInsertMap(getOrInsertMap(node, 'params'), group), name);
}

/** Write (or clear) a param's expression binding into the doc IN PLACE, matching the Rust
 * `GraphDoc::set_param` expr structure (`…params[group][name].expr = {source, enabled, triggers}`),
 * so the manager's `apply_client_update` detects it and applies it via `set_member_expression`. A
 * client leaf-write (§4). `null` clears the binding. No-op if the node is absent. Returns whether it
 * landed. The runtime `expression_error` is NOT in the doc — it rides the `state_update` echo the
 * manager emits after applying, exactly as the retired `set_expression` RPC did. */
export function setParamExpr(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string,
	expr: { source: string; enabled: boolean; triggers: boolean } | null
): boolean {
	const entry = paramEntryForWrite(doc, uid, group, name);
	if (!entry) return false;
	if (expr) {
		const ex = getOrInsertMap(entry, 'expr');
		if (ex.get('source') !== expr.source) ex.set('source', expr.source);
		if (ex.get('enabled') !== expr.enabled) ex.set('enabled', expr.enabled);
		if (ex.get('triggers') !== expr.triggers) ex.set('triggers', expr.triggers);
	} else if (entry.get('expr') !== undefined) {
		entry.delete('expr');
	}
	return true;
}

/** Write a committed node/instance position `[x, y]` into the doc IN PLACE, matching the Rust
 * `GraphDoc::set_node_pos` / `set_pos_if_changed` structure (`{uid}.pos.{x,y}`), so the manager's
 * `apply_client_update` detects the move and applies it via `set_member_pos`. A client leaf-write
 * (§4) committed on drag-stop — live drag stays local (Svelte Flow), never the doc. The uid is a
 * real node OR a sub-patch instance box (both carry a top-level `pos`). No-op if the uid is in
 * neither map (never mint a phantom). Idempotent: re-writing the current position writes nothing.
 * Returns whether the write landed. */
export function setNodePos(doc: Y.Doc, uid: string, pos: [number, number]): boolean {
	const target = nodesMap(doc).get(uid) ?? instancesMap(doc).get(uid);
	if (!target) return false;
	let p = target.get('pos') as Y.Map<unknown> | undefined;
	if (!p) {
		p = new Y.Map<unknown>();
		target.set('pos', p);
	}
	if (p.get('x') !== pos[0]) p.set('x', pos[0]);
	if (p.get('y') !== pos[1]) p.set('y', pos[1]);
	return true;
}

/** A node's opaque per-slot viewer blob (`{slot: {collapsed, kind, settings}}`), or `undefined`. */
export function viewersJson(doc: Y.Doc, uid: string): unknown {
	const v = nodesMap(doc).get(uid)?.get('viewers');
	if (typeof v !== 'string') return undefined;
	try {
		return JSON.parse(v);
	} catch {
		return undefined;
	}
}

/** Write a node's opaque per-slot viewer blob into the doc IN PLACE as a JSON string, matching the
 * Rust `GraphDoc::set_viewers` representation, so the manager's `apply_client_update` detects it
 * and persists it (set_node_viewers → .gfi). A client leaf-write (§4) — the view-state is soft,
 * client-originated, human-rate (debounced on the caller). No-op if the node is absent (never mint
 * a phantom). Idempotent: re-writing the same blob writes nothing. Returns whether it landed.
 *
 * The blob stays an opaque STRING for now (typed per-slot G4 comes with multi-user): the manager's
 * `set_viewers` guard parse-compares, so a differently-ordered write is accepted as-is. */
export function setViewers(doc: Y.Doc, uid: string, viewers: unknown): boolean {
	const node = nodesMap(doc).get(uid);
	if (!node) return false;
	const s = JSON.stringify(viewers ?? {});
	if (node.get('viewers') !== s) node.set('viewers', s);
	return true;
}

/** The `instances` root map (the sub-patch forest). */
export function instancesMap(doc: Y.Doc): Y.Map<Y.Map<unknown>> {
	return doc.getMap('instances') as Y.Map<Y.Map<unknown>>;
}

function pos2(m: Y.Map<unknown> | undefined): [number, number] {
	const p = m?.get('pos') as Y.Map<unknown> | undefined;
	const n = (k: string) => {
		const v = p?.get(k);
		return typeof v === 'number' ? v : 0;
	};
	return [n('x'), n('y')];
}

/** A sub-patch instance's forest view, or `null` if the uid is absent. */
export function instanceView(doc: Y.Doc, uid: string): InstanceView | null {
	const inst = instancesMap(doc).get(uid);
	if (!inst) return null;
	const members: Record<string, string> = {};
	const mm = inst.get('members') as Y.Map<unknown> | undefined;
	if (mm) for (const [local, u] of mm.entries()) if (typeof u === 'string') members[local] = u;
	const iface: BoundaryView[] = [];
	const im = inst.get('interface') as Y.Map<Y.Map<unknown>> | undefined;
	if (im) {
		for (const [bnd, b] of im.entries()) {
			const inner_node = b.get('inner_node');
			const inner_slot = b.get('inner_slot');
			iface.push({
				bnd_id: bnd,
				dir: str(b, 'dir'),
				dtype: str(b, 'dtype'),
				name: str(b, 'name'),
				pos: pos2(b),
				inner_node: typeof inner_node === 'string' ? inner_node : undefined,
				inner_slot: typeof inner_slot === 'string' ? inner_slot : undefined
			});
		}
	}
	const def = inst.get('def_id');
	return {
		uid,
		name: str(inst, 'name'),
		def_id: typeof def === 'string' ? def : undefined,
		parent: str(inst, 'parent'),
		pos: pos2(inst),
		members,
		interface: iface
	};
}

/** All sub-patch instance views, in the doc's key order. */
export function instanceViews(doc: Y.Doc): InstanceView[] {
	const out: InstanceView[] = [];
	for (const uid of instancesMap(doc).keys()) {
		const v = instanceView(doc, uid);
		if (v) out.push(v);
	}
	return out;
}

/** All links, in array order. */
export function linkViews(doc: Y.Doc): LinkView[] {
	return linksArray(doc)
		.toArray()
		.map((m) => ({
			node_out: str(m, 'node_out'),
			slot_out: str(m, 'slot_out'),
			node_in: str(m, 'node_in'),
			slot_in: str(m, 'slot_in')
		}));
}

// ── Globals ────────────────────────────────────────────────────────────────────────────────────
// The `globals` root map — patch-scoped named scalars, the fourth doc root beside nodes/links/
// instances. Each entry is `{value, type, system}` (the Rust `GraphDoc` globals mirror). `type`
// disambiguates float↔int after JS's number normalization; `system` marks a code-owned global that
// the panel locks (no delete/rename) and the manager refuses to delete.

/** A global's declared scalar type — mirrors `GlobalValue::type_tag` on the Rust side. */
export type GlobalType = 'float' | 'int' | 'bool' | 'string';

export interface GlobalView {
	name: string;
	value: number | string | boolean;
	type: GlobalType;
	/** A system global (editable value, but never deletable/renamable). */
	system: boolean;
}

/** The `globals` root map. */
export function globalsMap(doc: Y.Doc): Y.Map<Y.Map<unknown>> {
	return doc.getMap('globals') as Y.Map<Y.Map<unknown>>;
}

/** All globals, in the doc's key order (system-first, then user in creation order). */
export function globalViews(doc: Y.Doc): GlobalView[] {
	const out: GlobalView[] = [];
	for (const [name, g] of globalsMap(doc).entries()) {
		const value = g.get('value');
		const type = g.get('type');
		if (
			(typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') &&
			(type === 'float' || type === 'int' || type === 'bool' || type === 'string')
		) {
			out.push({ name, value, type, system: g.get('system') === true });
		}
	}
	return out;
}

/** Whether `name` is a legal global identifier — the exact mirror of the Rust `is_valid_global_name`:
 * `[A-Za-z_][A-Za-z0-9_]*`, and not the reserved namespace token `globals`. The panel gates a
 * rename/add on this so an illegal name never reaches the doc (the manager would reject it anyway). */
export function isValidGlobalName(name: string): boolean {
	return name !== 'globals' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
}

/** Edit an EXISTING global's value in place (system or user), matching the Rust globals mirror so the
 * manager's `apply_client_update` detects it and routes it through `apply_global_change`. Leaves
 * `type`/`system` untouched (a system global stays system, its type stable). No-op if the global is
 * absent (never mint via this path — use `addGlobal`). Idempotent. Returns whether it landed. */
export function setGlobalValue(doc: Y.Doc, name: string, value: number | string | boolean): boolean {
	const entry = globalsMap(doc).get(name);
	if (!entry) return false;
	if (entry.get('value') !== value) entry.set('value', value);
	return true;
}

/** Mint a NEW user global (the panel's "add row"). Refuses an invalid name or a collision with an
 * existing global (system or user). The manager's diff sees the new top-level entry and adds it via
 * `apply_global_change(name, Some(..))`. Returns whether it landed. */
export function addGlobal(
	doc: Y.Doc,
	name: string,
	value: number | string | boolean,
	type: GlobalType
): boolean {
	const g = globalsMap(doc);
	if (!isValidGlobalName(name) || g.has(name)) return false;
	const entry = new Y.Map<unknown>();
	entry.set('value', value);
	entry.set('type', type);
	entry.set('system', false);
	g.set(name, entry);
	return true;
}

/** Remove a USER global. Refuses a system global (the panel also locks it; the manager rejects it
 * regardless) or an absent one. Returns whether it landed. */
export function removeGlobal(doc: Y.Doc, name: string): boolean {
	const g = globalsMap(doc);
	const entry = g.get(name);
	if (!entry || entry.get('system') === true) return false;
	g.delete(name);
	return true;
}

/** Rename a USER global, carrying its value + type to the new key. A CRDT map can't rename in place,
 * so this is delete-old + add-new — the manager sees a remove + an add, exactly the spec's "no rename
 * cascade": a stale `globals.<old>` then throws the natural missing-key error at eval time. Refuses a
 * system global, an invalid/colliding new name, or an absent old one. Returns whether it landed. */
export function renameGlobal(doc: Y.Doc, oldName: string, newName: string): boolean {
	const g = globalsMap(doc);
	const entry = g.get(oldName);
	if (!entry || entry.get('system') === true) return false;
	if (oldName === newName) return true;
	if (!isValidGlobalName(newName) || g.has(newName)) return false;
	const next = new Y.Map<unknown>();
	next.set('value', entry.get('value'));
	next.set('type', entry.get('type'));
	next.set('system', false);
	g.set(newName, next);
	g.delete(oldName);
	return true;
}
