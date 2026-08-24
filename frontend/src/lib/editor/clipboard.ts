/** The clipboard payload: goofi's own graph fragment, and the version that says so.
 * Putting it THERE is `$lib/clipboard`'s job; this module is only the shape. */
/** Bumped when the payload shape changes, so an older tab's text is refused rather than half-read. */
const CLIP_VERSION = 2;

/** What a copy puts on the clipboard: goofi's own graph fragment, in the shape a `.gfi` carries,
 * so what the manager reads back is the format it already writes. */
export interface Clipboard {
	__goofi_clip__: number;
	doc: GraphFragment;
}

export interface GraphFragment {
	nodes: Record<string, { pos?: [number, number]; scope?: string }>;
	links?: unknown[];
}

export function serializeClipboard(doc: GraphFragment): Clipboard {
	return { __goofi_clip__: CLIP_VERSION, doc };
}

/** Parse clipboard text; null if it isn't a goofi clipboard payload of this version. */
export function parseClipboard(text: string): Clipboard | null {
	let payload: unknown;
	try {
		payload = JSON.parse(text);
	} catch {
		return null;
	}
	const clip = payload as Clipboard | null;
	if (typeof clip !== 'object' || clip === null || clip.__goofi_clip__ !== CLIP_VERSION) return null;
	if (typeof clip.doc !== 'object' || clip.doc === null || typeof clip.doc.nodes !== 'object') return null;
	return clip;
}

/** The centre of a fragment's ROOTS, which is what a paste anchors at a point. A record naming a
 * scope inside the fragment is drawn in that scope's own space, so its position is not on the
 * canvas the anchor is measured on. */
export function fragmentCentre(doc: GraphFragment): [number, number] {
	const at = Object.values(doc.nodes ?? {})
		.filter((n) => n.scope === undefined)
		.map((n) => n.pos ?? [0, 0]);
	if (at.length === 0) return [0, 0];
	return [
		at.reduce((a, p) => a + (p[0] ?? 0), 0) / at.length,
		at.reduce((a, p) => a + (p[1] ?? 0), 0) / at.length
	];
}
