/**
 * The browser replica of goofi's control-plane document, as `goofi_bridge::projection` builds it.
 * Every reader is total: an absent or wrongly-typed leaf answers a default rather than throwing.
 */
import { EMPTY_PANEL_TYPE, SCOPE_TYPE, boundaryType } from '$lib/api/vocab';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { PARAM_MODES, type ParamMode } from '$lib/api/types';
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
	/** The scope this record is drawn in; `ROOT_ID` when it names none. */
	scope: string;
}

export interface LinkView {
	node_out: string;
	slot_out: string;
	node_in: string;
	slot_in: string;
}

/** What a sub-patch facade exposes, derived from the records that name it. */
export interface FacadeFace {
	input_slots: Record<string, string>;
	output_slots: Record<string, string>;
	slot_labels: Record<string, string>;
	memberCount: number;
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

function globalsMap(doc: Doc): Record<string, Obj> {
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
	return { uid, type: str(n, 'type'), name: str(n, 'name'), pos: pos2(n), scope: scopeOf(n) };
}

/** Every record the canvas draws: leaf, sub-patch facade and boundary port alike, in one list,
 * because the document carries them in one map and they are one kind of thing to the editor. */
export function nodeViews(doc: Doc): NodeView[] {
	const out: NodeView[] = [];
	for (const uid of Object.keys(nodesMap(doc))) {
		const v = nodeView(doc, uid);
		if (v) out.push(v);
	}
	return out;
}

/** Each facade's face, keyed by its uid: a PORT is the facade's slot, keyed by the port's stable
 * uid and labelled with its renameable name, so a rename relabels without re-keying the wire. */
export function facadeFaces(doc: Doc): Map<string, FacadeFace> {
	const out = new Map<string, FacadeFace>();
	const at = (uid: string): FacadeFace => {
		let f = out.get(uid);
		if (!f) out.set(uid, (f = { input_slots: {}, output_slots: {}, slot_labels: {}, memberCount: 0 }));
		return f;
	};
	for (const [uid, rec] of Object.entries(nodesMap(doc))) {
		if (str(rec, 'type') === SCOPE_TYPE) at(uid);
		const parent = optStr(rec, 'scope');
		if (!parent) continue;
		const face = at(parent);
		face.memberCount++;
		const bnd = boundaryType(str(rec, 'type'));
		if (!bnd) continue;
		(bnd.dir === 'in' ? face.input_slots : face.output_slots)[uid] = bnd.dtype;
		face.slot_labels[uid] = str(rec, 'name');
	}
	return out;
}

/** A param's source record as the document carries it: the mode, and the texts it retains. */
export interface ParamSource {
	mode: ParamMode;
	expr?: string;
	ref?: string;
	triggers?: boolean;
}

export type DocParamLeaves = Record<
	string,
	Record<string, { value?: number | string | boolean; source?: ParamSource }>
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
			if (typeof entry.mode === 'string' && (PARAM_MODES as readonly string[]).includes(entry.mode)) {
				const source: ParamSource = { mode: entry.mode as ParamMode };
				if (typeof entry.expr === 'string') source.expr = entry.expr;
				if (typeof entry.ref === 'string') source.ref = entry.ref;
				if (entry.triggers === true) source.triggers = true;
				leaf.source = source;
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

/** Write (or, with `null`, clear) a param's source record — a test-seed double, as [`setParamValue`] is. */
export function setParamSource(
	doc: Doc,
	uid: string,
	group: string,
	name: string,
	source: ParamSource | null
): boolean {
	const entry = paramEntry(doc, uid, group, name);
	if (!entry) return false;
	delete entry.mode;
	delete entry.expr;
	delete entry.ref;
	delete entry.triggers;
	if (source) {
		entry.mode = source.mode;
		if (source.expr !== undefined) entry.expr = source.expr;
		if (source.ref !== undefined) entry.ref = source.ref;
		if (source.triggers) entry.triggers = true;
	}
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
	/** A machine-owned global: the value is read-only too. */
	locked: boolean;
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
			out.push({ name, value, type, system: g.system === true, locked: g.locked === true });
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

/** Python's keywords, plus goofi's own namespace token `globals`: a regex reads each as an
 * identifier and a parser does not. */
const RESERVED = new Set(
	`globals False None True and as assert async await break class continue def del elif else
	 except finally for from global if import in is lambda nonlocal not or pass raise return try
	 while with yield`.split(/\s+/)
);

/** Whether `name` is legal in the ONE expression namespace — the exact mirror of the Rust
 * `is_valid_identifier`. Every name an expression can spell is held to it, because an expression
 * reads one as an ATTRIBUTE: `globals.gain`, and a sub-patch's slot in `nd('chain').drain`. */
export function isValidIdentifier(name: string): boolean {
	return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !RESERVED.has(name);
}

/** The NODE name rule, the mirror of the Rust `is_valid_name`: a letter then letters or digits, not
 * a keyword — a reference spells `node.slot`, so no underscore either. Globals keep the rule above. */
export function isValidName(name: string): boolean {
	return /^[A-Za-z][A-Za-z0-9]*$/.test(name) && !RESERVED.has(name);
}
