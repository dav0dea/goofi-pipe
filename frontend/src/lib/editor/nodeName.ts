/**
 * A node's display name may be qualified with its sub-patch path, e.g.
 * `subpatch0::filter0` (the backend qualifies a grouped member's display name to
 * `inst_id::local`). When we let the user rename a node, only the final segment —
 * the bare name — is editable; the leading path is shown faint and preserved.
 *
 * The path is everything up to and including the last `::`; the base is the
 * remaining final segment. A top-level node has an empty path.
 */
const SEP = '::';

export function splitNodeName(name: string): { path: string; base: string } {
	const idx = name.lastIndexOf(SEP);
	if (idx < 0) return { path: '', base: name };
	return { path: name.slice(0, idx + SEP.length), base: name.slice(idx + SEP.length) };
}

/** Inverse of `splitNodeName`: re-attach an edited base to its faint path. */
export function joinNodeName(path: string, base: string): string {
	return `${path}${base}`;
}
