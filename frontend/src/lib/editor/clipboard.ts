/** Copy/paste payload for a selection of nodes + the links among them. */
import { paramValues, type LinkInfo, type NodeInstanceInfo } from '$lib/api/control';

const CLIP_VERSION = 2;

export interface ClipNode {
	/** The source node's uid (its identity) — the spec key paste remaps links by. */
	uid: string;
	type: string;
	category: string;
	params: Record<string, Record<string, unknown>>;
	/** Position relative to the selection's centroid. */
	offset: [number, number];
}

export interface Clipboard {
	__goofi_clip__: number;
	nodes: ClipNode[];
	links: LinkInfo[];
}

/** Build a clipboard payload from selected nodes and the links among them. */
export function serializeClipboard(nodes: NodeInstanceInfo[], links: LinkInfo[]): Clipboard {
	const cx = nodes.reduce((a, n) => a + n.pos[0], 0) / nodes.length;
	const cy = nodes.reduce((a, n) => a + n.pos[1], 0) / nodes.length;
	return {
		__goofi_clip__: CLIP_VERSION,
		nodes: nodes.map((n) => ({
			uid: n.uid,
			type: n.type,
			category: n.category,
			params: paramValues(n),
			offset: [n.pos[0] - cx, n.pos[1] - cy]
		})),
		links
	};
}

/** Parse clipboard text; null if it isn't a goofi clipboard payload. */
export function parseClipboard(text: string): Clipboard | null {
	let payload: unknown;
	try {
		payload = JSON.parse(text);
	} catch {
		return null;
	}
	if (
		typeof payload !== 'object' ||
		payload === null ||
		(payload as Clipboard).__goofi_clip__ !== CLIP_VERSION ||
		!Array.isArray((payload as Clipboard).nodes)
	) {
		return null;
	}
	const clip = payload as Clipboard;
	return { __goofi_clip__: CLIP_VERSION, nodes: clip.nodes, links: clip.links ?? [] };
}

/** Map clipboard nodes to instantiation specs anchored at `at`, the paste base point. */
export function clipToSpecs(
	clip: Clipboard,
	at: [number, number]
): {
	key: string;
	type: string;
	category: string;
	pos: [number, number];
	params: Record<string, Record<string, unknown>>;
}[] {
	return clip.nodes.map((n) => ({
		key: n.uid,
		type: n.type,
		category: n.category,
		pos: [Math.round(at[0] + n.offset[0]), Math.round(at[1] + n.offset[1])],
		params: n.params
	}));
}
