/**
 * The browser replica of goofi's control-plane document, as `goofi_bridge::projection` builds it.
 * Every reader is total: an absent or wrongly-typed leaf answers a default rather than throwing.
 */
import { EMPTY_PANEL_TYPE, SCOPE_TYPE, boundaryType } from '$lib/api/vocab';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { LayoutNode, Workspace } from 'panelty';

export type Doc = Record<string, unknown>;

export function emptyDoc(): Doc {
	return { nodes: {}, links: [], globals: {}, arrangement: {} };
}

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
	/** member uid → whether the member is itself a nested scope. */
	members: Record<string, boolean>;
	/** The scope's boundary ports, in document order. */
	interface: BoundaryView[];
}

type Obj = Record<string, unknown>;

const obj = (v: unknown): Obj => (v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : {});
const str = (m: Obj | undefined, key: string): string => {
	const v = m?.[key];
	return typeof v === 'string' ? v : '';
};
const optStr = (m: Obj | undefined, key: string): string | undefined => {
	const v = m?.[key];
	return typeof v === 'string' ? v : undefined;
};

export function nodesMap(doc: Doc): Record<string, Obj> {
	return obj(doc.nodes) as Record<string, Obj>;
}

export function linksArray(doc: Doc): Obj[] {
	return Array.isArray(doc.links) ? (doc.links as Obj[]) : [];
}

/** The scope a record names, or `'__root__'`. The doc omits the key at the top level, because a
 * merge patch spends `null` on "delete this key" and could not tell that from a move out. */
function scopeOf(rec: Obj | undefined): string {
	return optStr(rec, 'scope') ?? ROOT_ID;
}

/** Every record of one kind, keyed by uid. One map carries leaves, facades and ports alike, so
 * which kind a record is is a question about its `type`. */
function records(doc: Doc, want: (type: string) => boolean): [string, Obj][] {
	return Object.entries(nodesMap(doc)).filter(([, n]) => want(str(n, 'type')));
}

export function globalsMap(doc: Doc): Record<string, Obj> {
	return obj(doc.globals) as Record<string, Obj>;
}

function pos2(m: Obj | undefined): [number, number] {
	const p = obj(m?.pos);
	const n = (k: string) => (typeof p[k] === 'number' ? (p[k] as number) : 0);
	return [n('x'), n('y')];
}

export function nodeView(doc: Doc, uid: string): NodeView | null {
	const n = nodesMap(doc)[uid];
	if (!n) return null;
	return { uid, type: str(n, 'type'), name: str(n, 'name'), pos: pos2(n) };
}

/** Every node the canvas draws — leaves and boundary ports alike. A port has no thread behind it,
 * but it is a node in every way the editor addresses one, so it is not a second kind here. A
 * FACADE is left out: it is a scope, and `instanceView` reads it. */
export function nodeViews(doc: Doc): NodeView[] {
	const out: NodeView[] = [];
	for (const [uid] of records(doc, (t) => t !== SCOPE_TYPE)) {
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

export type DocParamLeaves = Record<
	string,
	Record<string, { value?: number | string | boolean; expr?: ParamExpr }>
>;

export function docParams(doc: Doc, uid: string): DocParamLeaves {
	const out: DocParamLeaves = {};
	const params = obj(nodesMap(doc)[uid]?.params);
	for (const [group, g] of Object.entries(params)) {
		out[group] = {};
		for (const [name, raw] of Object.entries(obj(g))) {
			const entry = obj(raw);
			const leaf: DocParamLeaves[string][string] = {};
			const v = entry.value;
			if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') leaf.value = v;
			const expr = obj(entry.expr);
			if (typeof expr.source === 'string') {
				leaf.expr = { source: expr.source, enabled: expr.enabled === true, triggers: expr.triggers === true };
			}
			out[group][name] = leaf;
		}
	}
	return out;
}

/** The param entry, get-or-inserted, or `undefined` when the node is absent — never mint a phantom node. */
function paramEntry(doc: Doc, uid: string, group: string, name: string): Obj | undefined {
	const node = nodesMap(doc)[uid];
	if (!node) return undefined;
	const into = (parent: Obj, key: string): Obj => {
		if (!parent[key] || typeof parent[key] !== 'object') parent[key] = {};
		return parent[key] as Obj;
	};
	return into(into(into(node, 'params'), group), name);
}

/** Write a param value — a test-seed double, because the replica is READ-ONLY in production. */
export function setParamValue(
	doc: Doc,
	uid: string,
	group: string,
	name: string,
	value: number | string | boolean
): boolean {
	const entry = paramEntry(doc, uid, group, name);
	if (!entry) return false;
	entry.value = value;
	return true;
}

/** Write (or, with `null`, clear) a param's expression binding — a test-seed double, as [`setParamValue`] is. */
export function setParamExpr(
	doc: Doc,
	uid: string,
	group: string,
	name: string,
	expr: ParamExpr | null
): boolean {
	const entry = paramEntry(doc, uid, group, name);
	if (!entry) return false;
	if (expr) entry.expr = { ...expr };
	else delete entry.expr;
	return true;
}

/** A node's opaque per-slot viewer blob (`{slot: {collapsed, kind, settings}}`), or `undefined`. */
export function viewersJson(doc: Doc, uid: string): unknown {
	const v = nodesMap(doc)[uid]?.viewers;
	if (typeof v !== 'string') return undefined;
	try {
		return JSON.parse(v);
	} catch {
		return undefined;
	}
}

export function instanceView(doc: Doc, uid: string): InstanceView | null {
	const inst = nodesMap(doc)[uid];
	if (!inst || str(inst, 'type') !== SCOPE_TYPE) return null;
	const members: Record<string, boolean> = {};
	for (const [muid, m] of Object.entries(nodesMap(doc))) {
		if (scopeOf(m) === uid) members[muid] = str(m, 'type') === SCOPE_TYPE;
	}
	// A port's inner wire is a link, so it is read where every other cable is read.
	const wires = linkViews(doc);
	const iface: BoundaryView[] = [];
	for (const [bnd, b] of records(doc, (t) => !!boundaryType(t))) {
		if (scopeOf(b) !== uid) continue;
		const kind = boundaryType(str(b, 'type'))!;
		const wire =
			kind.dir === 'in'
				? wires.find((l) => l.node_out === bnd)
				: wires.find((l) => l.node_in === bnd);
		const inner = kind.dir === 'in'
			? wire && { node: wire.node_in, slot: wire.slot_in }
			: wire && { node: wire.node_out, slot: wire.slot_out };
		iface.push({
			bnd_id: bnd,
			dir: kind.dir,
			dtype: kind.dtype,
			name: str(b, 'name'),
			pos: pos2(b),
			inner_node: inner?.node,
			inner_slot: inner?.slot
		});
	}
	return {
		uid,
		name: str(inst, 'name'),
		parent: scopeOf(inst),
		pos: pos2(inst),
		members,
		interface: iface
	};
}

export function instanceViews(doc: Doc): InstanceView[] {
	const out: InstanceView[] = [];
	for (const [uid] of records(doc, (t) => t === SCOPE_TYPE)) {
		const v = instanceView(doc, uid);
		if (v) out.push(v);
	}
	return out;
}

export function linkViews(doc: Doc): LinkView[] {
	return linksArray(doc).map((m) => ({
		node_out: str(m, 'node_out'),
		slot_out: str(m, 'slot_out'),
		node_in: str(m, 'node_in'),
		slot_in: str(m, 'slot_in')
	}));
}

/** A global's declared scalar type — it disambiguates float↔int after JS's number normalization. */
export type GlobalType = 'float' | 'int' | 'bool' | 'string';

export interface GlobalView {
	name: string;
	value: number | string | boolean;
	type: GlobalType;
	/** A system global (editable value, but never deletable/renamable). */
	system: boolean;
}

/** All globals, in the document's key order (system-first, then user in creation order). */
export function globalViews(doc: Doc): GlobalView[] {
	const out: GlobalView[] = [];
	for (const [name, raw] of Object.entries(globalsMap(doc))) {
		const g = obj(raw);
		const value = g.value;
		const type = g.type;
		if (
			(typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') &&
			(type === 'float' || type === 'int' || type === 'bool' || type === 'string')
		) {
			out.push({ name, value, type, system: g.system === true });
		}
	}
	return out;
}

/** Parse one layout node, answering the share it takes of its parent — a split carries its children's shares. */
function layoutNode(raw: unknown, root: boolean): { node: LayoutNode; size: number } | null {
	const n = obj(raw);
	const id = optStr(n, 'id');
	if (!id) return null;
	// A root fills its tab and carries no share on the wire.
	const size = root ? 1 : typeof n.size === 'number' ? n.size : 0;
	if (n.kind === 'panel') {
		return {
			node: { kind: 'panel', id, panelType: optStr(n, 'panel_type') ?? EMPTY_PANEL_TYPE, state: panelState(n.state) },
			size
		};
	}
	if (n.kind !== 'split' || !Array.isArray(n.children)) return null;
	const children: LayoutNode[] = [];
	const sizes: number[] = [];
	for (const c of n.children) {
		const parsed = layoutNode(c, false);
		if (!parsed) continue;
		children.push(parsed.node);
		sizes.push(parsed.size);
	}
	if (children.length === 0) return null;
	return {
		node: { kind: 'split', id, direction: n.axis === 'column' ? 'column' : 'row', children, sizes },
		size
	};
}

/** A panel's opaque bag, out of its JSON string leaf — a string, because a null leaf would make the merge patch ambiguous. */
function panelState(raw: unknown): unknown {
	if (typeof raw !== 'string') return undefined;
	try {
		const v: unknown = JSON.parse(raw);
		return v === null ? undefined : v;
	} catch {
		return undefined;
	}
}

/** The tab strip as the panel system draws it; a tab whose root will not parse is dropped. */
export function arrangementTabs(doc: Doc): Workspace[] {
	const raw = obj(doc.arrangement).tabs;
	if (!Array.isArray(raw)) return [];
	const out: Workspace[] = [];
	for (const t of raw) {
		const tab = obj(t);
		const id = optStr(tab, 'id');
		const parsed = layoutNode(tab.root, true);
		if (!id || !parsed) continue;
		out.push({ id, name: optStr(tab, 'name') ?? '', root: parsed.node });
	}
	return out;
}

/** Whether `name` is a legal global identifier — the exact mirror of the Rust `is_valid_global_name`. */
export function isValidGlobalName(name: string): boolean {
	return name !== 'globals' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
}
