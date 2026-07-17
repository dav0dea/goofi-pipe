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

/** A param's expression source (`nd('…')`), or `undefined` if it has no binding. */
export function paramExprSource(
	doc: Y.Doc,
	uid: string,
	group: string,
	name: string
): string | undefined {
	const expr = paramEntry(doc, uid, group, name)?.get('expr') as Y.Map<unknown> | undefined;
	const s = expr?.get('source');
	return typeof s === 'string' ? s : undefined;
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
