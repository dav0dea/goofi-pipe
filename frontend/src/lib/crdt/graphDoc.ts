/**
 * The browser replica of goofi's control-plane document — the exact shape
 * `goofi_bridge::projection` builds: `nodes: {uid: {type, name, pos:{x,y}, params: {group: {name:
 * {value, expr?}}}, viewers}}`, `links: [{node_out, slot_out, node_in, slot_in}]`, plus
 * `instances`, `globals` and `arrangement`.
 *
 * Plain JSON, and plain readers over it (no Svelte, no WebSocket) so they unit-test directly. The
 * reactive `.svelte.ts` layer holds the document and re-exposes these as runes.
 *
 * Every reader is total: an absent or wrongly-typed leaf answers a default rather than throwing.
 * The manager is the sole author, so a surprise here means the two ends have drifted — and a
 * half-drawn graph is a better report of that than a blank page.
 */
import { EMPTY_PANEL_TYPE } from '$lib/api/vocab';
import type { LayoutNode, Workspace } from '$lib/workspace/model';

/** The document, as it arrives. */
export type Doc = Record<string, unknown>;

/** An empty document: the five roots present, so a reader never has to invent one. */
export function emptyDoc(): Doc {
	return { nodes: {}, links: [], instances: {}, globals: {}, arrangement: {} };
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
	/** member uid → whether the member is itself a nested scope (flat model: keyed by uid, no
	 * template-local names). */
	members: Record<string, boolean>;
	/** The scope's boundary stubs (read from the doc's `stubs` map). */
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

/** The `nodes` root, by uid. */
export function nodesMap(doc: Doc): Record<string, Obj> {
	return obj(doc.nodes) as Record<string, Obj>;
}

/** The `links` root, in order. */
export function linksArray(doc: Doc): Obj[] {
	return Array.isArray(doc.links) ? (doc.links as Obj[]) : [];
}

/** The `instances` root (the sub-patch forest), by uid. */
export function instancesMap(doc: Doc): Record<string, Obj> {
	return obj(doc.instances) as Record<string, Obj>;
}

/** The `globals` root, by name. */
export function globalsMap(doc: Doc): Record<string, Obj> {
	return obj(doc.globals) as Record<string, Obj>;
}

/** `{x, y}` as the pair the editor draws with. */
function pos2(m: Obj | undefined): [number, number] {
	const p = obj(m?.pos);
	const n = (k: string) => (typeof p[k] === 'number' ? (p[k] as number) : 0);
	return [n('x'), n('y')];
}

/** A node's identity view, or `null` if the uid is absent. */
export function nodeView(doc: Doc, uid: string): NodeView | null {
	const n = nodesMap(doc)[uid];
	if (!n) return null;
	return { uid, type: str(n, 'type'), name: str(n, 'name'), pos: pos2(n) };
}

/** All node identity views, in the document's key order. */
export function nodeViews(doc: Doc): NodeView[] {
	const out: NodeView[] = [];
	for (const uid of Object.keys(nodesMap(doc))) {
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

/** The document-owned param leaves for one node: value + optional expression binding, per
 * group/name. Exactly the `{value, expr?}` structure the document stores (nodeAssembly merges
 * these with the catalog descriptor + runtime overlay). */
export type DocParamLeaves = Record<
	string,
	Record<string, { value?: number | string | boolean; expr?: ParamExpr }>
>;

/** Read a node's committed param leaves (value + expression binding), per group/name. */
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

/** The `nodes[uid].params[group][name]` entry, get-or-inserted, or `undefined` when the node is
 * absent from this replica — never mint a phantom node. Backs the test-seed writers below. */
function paramEntry(doc: Doc, uid: string, group: string, name: string): Obj | undefined {
	const node = nodesMap(doc)[uid];
	if (!node) return undefined;
	const into = (parent: Obj, key: string): Obj => {
		if (!parent[key] || typeof parent[key] !== 'object') parent[key] = {};
		return parent[key] as Obj;
	};
	return into(into(into(node, 'params'), group), name);
}

/** Write a param value at `nodes[uid].params[group][name].value`. The replica is READ-ONLY in
 * production — the manager owns every write — so this is a test-seed double, letting a store test
 * stand in for the manager's projection. Answers whether it landed. */
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

/** Write (or, with `null`, clear) a param's expression binding. A test-seed double, as
 * [`setParamValue`] is. Answers whether it landed. */
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

/** A sub-patch instance's forest view, or `null` if the uid is absent. */
export function instanceView(doc: Doc, uid: string): InstanceView | null {
	const inst = instancesMap(doc)[uid];
	if (!inst) return null;
	const members: Record<string, boolean> = {};
	for (const [muid, m] of Object.entries(obj(inst.members))) {
		members[muid] = obj(m).is_instance === true;
	}
	const iface: BoundaryView[] = [];
	for (const [bnd, raw] of Object.entries(obj(inst.stubs))) {
		const b = obj(raw);
		iface.push({
			bnd_id: bnd,
			dir: str(b, 'dir'),
			dtype: str(b, 'dtype'),
			name: str(b, 'name'),
			pos: pos2(b),
			inner_node: optStr(b, 'inner_node'),
			inner_slot: optStr(b, 'inner_slot')
		});
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

/** All sub-patch instance views, in the document's key order. */
export function instanceViews(doc: Doc): InstanceView[] {
	const out: InstanceView[] = [];
	for (const uid of Object.keys(instancesMap(doc))) {
		const v = instanceView(doc, uid);
		if (v) out.push(v);
	}
	return out;
}

/** All links, in array order. */
export function linkViews(doc: Doc): LinkView[] {
	return linksArray(doc).map((m) => ({
		node_out: str(m, 'node_out'),
		slot_out: str(m, 'slot_out'),
		node_in: str(m, 'node_in'),
		slot_in: str(m, 'slot_in')
	}));
}

// ── Globals ────────────────────────────────────────────────────────────────────────────────────
// The `globals` root — patch-scoped named scalars. Each entry is `{value, type, system}`. `type`
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

// ── The arrangement ────────────────────────────────────────────────────────────────────────────
// The `arrangement` root — the editor's panel layout, held as a TREE: `tabs` is an array whose
// order IS the strip order, and each tab holds one root node. This reads it straight into the shape
// the panel system draws; there is no intermediate.

/** Parse one node, answering the share it takes of its parent alongside it — a split carries its
 * children's shares, so they are collected on the way up rather than stored on each child twice. */
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

/** A panel's opaque bag, out of its JSON string leaf. It rides as a STRING because a panel clears a
 * key with an explicit `null`, and a null leaf in the document would make the merge patch ambiguous. */
function panelState(raw: unknown): unknown {
	if (typeof raw !== 'string') return undefined;
	try {
		const v: unknown = JSON.parse(raw);
		return v === null ? undefined : v;
	} catch {
		return undefined;
	}
}

/** The tab strip as the panel system draws it. A tab whose root will not parse is dropped rather
 * than drawn as a hole. */
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

/** Whether `name` is a legal global identifier — the exact mirror of the Rust `is_valid_global_name`:
 * `[A-Za-z_][A-Za-z0-9_]*`, and not the reserved namespace token `globals`. The panel gates a
 * rename/add on this so an illegal name never reaches the manager (which would reject it anyway). */
export function isValidGlobalName(name: string): boolean {
	return name !== 'globals' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
}
