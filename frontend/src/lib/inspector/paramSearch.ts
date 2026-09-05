import type { ParamDescriptor } from '$lib/api/types';

export type ParamHit = { group: string; name: string; descriptor: ParamDescriptor };

/** A node's params as the document carries them: group, then name. */
export type ParamGroups = Record<string, Record<string, ParamDescriptor>>;

/** True where `i` starts a word in `name`: the first letter, one after a separator, or a capital. */
function isWordStart(name: string, i: number): boolean {
	if (i === 0) return true;
	const prev = name[i - 1];
	const cur = name[i];
	if (!/[a-zA-Z0-9]/.test(prev)) return true;
	return prev === prev.toLowerCase() && cur !== cur.toLowerCase();
}

function atWordStart(name: string, q: string): boolean {
	const lower = name.toLowerCase();
	for (let i = lower.indexOf(q); i >= 0; i = lower.indexOf(q, i + 1)) {
		if (isWordStart(name, i)) return true;
	}
	return false;
}

/**
 * How well one param answers `q`, best first and 0 for a miss. The name outranks the family, which
 * outranks the doc, so typing a family name still surfaces that family without burying an exact hit.
 */
export function scoreParam(name: string, group: string, doc: string, q: string): number {
	const n = name.toLowerCase();
	if (n === q) return 6;
	if (n.startsWith(q)) return 5;
	if (atWordStart(name, q)) return 4;
	if (n.includes(q)) return 3;
	if (group.toLowerCase().includes(q)) return 2;
	if (doc.toLowerCase().includes(q)) return 1;
	return 0;
}

/**
 * Every param matching `query`, across all families, best first. An empty query matches nothing —
 * the caller shows its tabs instead.
 */
export function matchParams(params: ParamGroups, query: string): ParamHit[] {
	const q = query.trim().toLowerCase();
	if (q === '') return [];
	const scored: { hit: ParamHit; rank: number }[] = [];
	for (const [group, named] of Object.entries(params ?? {})) {
		for (const [name, descriptor] of Object.entries(named ?? {})) {
			const rank = scoreParam(name, group, descriptor?.doc ?? '', q);
			if (rank > 0) scored.push({ hit: { group, name, descriptor }, rank });
		}
	}
	// Stable within a rank: Object.entries keeps the document's order, which is the plugin's own.
	return scored.map((s, i) => ({ s, i })).sort((a, b) => b.s.rank - a.s.rank || a.i - b.i).map(({ s }) => s.hit);
}
