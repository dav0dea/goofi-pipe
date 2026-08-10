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
import type { Arrangement } from '$lib/workspace/arrangement';

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
	/** Parent scope uid, or `'__root__'` for a top-level scope. */
	parent: string;
	pos: [number, number];
	/** member uid → whether the member is itself a nested scope (flat model: keyed by uid, no
	 * template-local names). */
	members: Record<string, boolean>;
	/** The scope's boundary stubs (read from the doc's `stubs` map). */
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

/** Write a param value into the doc IN PLACE at `nodes[uid].params[group][name].value`, matching
 * the Rust `GraphDoc` mirror structure. The browser replica is read-only in production (the manager
 * owns all writes); this remains a TEST-SEED double, letting store tests write a value into the doc
 * to simulate the manager's mirror. No-op (returns false) if the node is absent — never mint a
 * phantom. Returns whether the write landed. */
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
 * absent from this replica (never mint a phantom node). Backs the test-seed `setParam*` doubles. */
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

/** Write (or clear) a param's expression binding into the doc IN PLACE at
 * `…params[group][name].expr = {source, enabled, triggers}`, matching the Rust `GraphDoc` mirror.
 * The browser replica is read-only in production (the manager owns all writes); this remains a
 * TEST-SEED double, letting store tests write a binding into the doc to simulate the manager's
 * mirror. `null` clears the binding. No-op (returns false) if the node is absent — never mint a
 * phantom. Returns whether it landed. */
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
	const members: Record<string, boolean> = {};
	const mm = inst.get('members') as Y.Map<Y.Map<unknown>> | undefined;
	if (mm) for (const [muid, m] of mm.entries()) members[muid] = Boolean(m?.get('is_instance'));
	const iface: BoundaryView[] = [];
	const im = inst.get('stubs') as Y.Map<Y.Map<unknown>> | undefined;
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
	return {
		uid,
		name: str(inst, 'name'),
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

// ── The arrangement ────────────────────────────────────────────────────────────────────────────
// The `arrangement` root map — the editor's panel layout, held FLAT and id-keyed (the fifth root).
// Every page, split and panel is one entry naming its `parent` and its `order`; the tree is rebuilt
// at render time by `$lib/workspace/arrangement`.

/** The `arrangement` root map. */
export function arrangementMap(doc: Y.Doc): Y.Map<unknown> {
	return doc.getMap('arrangement');
}

/** Every arrangement ENTRY, by id. The manager's id counter rides the same root under `#seq` as a
 * bare number, so an entry is recognised by carrying a `kind` — read the values, never the keys. */
export function arrangementEntries(doc: Y.Doc): Arrangement {
	const out: Arrangement = {};
	for (const [id, raw] of arrangementMap(doc).entries()) {
		if (!(raw instanceof Y.Map)) continue;
		const e = raw as Y.Map<unknown>;
		const kind = e.get('kind');
		if (kind !== 'page' && kind !== 'split' && kind !== 'panel') continue;
		const order = e.get('order');
		if (typeof order !== 'number') continue;
		const opt = (k: string): string | undefined => {
			const v = e.get(k);
			return typeof v === 'string' ? v : undefined;
		};
		const size = e.get('size');
		out[id] = {
			kind,
			order,
			name: opt('name'),
			parent: opt('parent'),
			size: typeof size === 'number' ? size : undefined,
			axis: opt('axis'),
			panel_type: opt('panel_type'),
			state: opt('state')
		};
	}
	return out;
}

/** Whether `name` is a legal global identifier — the exact mirror of the Rust `is_valid_global_name`:
 * `[A-Za-z_][A-Za-z0-9_]*`, and not the reserved namespace token `globals`. The panel gates a
 * rename/add on this so an illegal name never reaches the doc (the manager would reject it anyway). */
export function isValidGlobalName(name: string): boolean {
	return name !== 'globals' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
}
