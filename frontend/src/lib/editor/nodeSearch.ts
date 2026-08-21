/** Ranking for the add-node menu's type search. */
import type { NodeTypeInfo } from '$lib/api/control';

/** Match quality, best first. */
const TIER = {
	none: 0,
	doc: 1,
	category: 2,
	nameSubstring: 3,
	nameWord: 4, // query starts a CamelCase / separated word inside the name
	namePrefix: 5,
	nameExact: 6
} as const;

const isUpper = (c: string): boolean => c >= 'A' && c <= 'Z';
const isLower = (c: string): boolean => c >= 'a' && c <= 'z';
const isLetter = (c: string): boolean => isUpper(c) || isLower(c);
const isDigit = (c: string): boolean => c >= '0' && c <= '9';

/** Is index `i` the first character of a word in `name`? */
function isWordStart(name: string, i: number): boolean {
	if (i === 0) return true;
	const c = name[i];
	const prev = name[i - 1];
	const next = name[i + 1];
	if (isLetter(c) && !isLetter(prev)) return true; // after separator / digit
	if (isUpper(c) && isLower(prev)) return true; // camelCase hump: ...oU...
	if (isUpper(c) && isUpper(prev) && next !== undefined && isLower(next)) return true; // acronym tail: OSC|Out
	if (isDigit(c) && !isDigit(prev)) return true; // letters → digit
	return false;
}

function hasWordStartMatch(name: string, lowerName: string, q: string): boolean {
	for (let i = 0; i <= name.length - q.length; i++) {
		if (isWordStart(name, i) && lowerName.startsWith(q, i)) return true;
	}
	return false;
}

function tierFor(t: NodeTypeInfo, q: string): number {
	const name = t.type.toLowerCase();
	if (name === q) return TIER.nameExact;
	if (name.startsWith(q)) return TIER.namePrefix;
	if (hasWordStartMatch(t.type, name, q)) return TIER.nameWord;
	if (name.includes(q)) return TIER.nameSubstring;
	if (t.category.toLowerCase().includes(q)) return TIER.category;
	if (t.doc.toLowerCase().includes(q)) return TIER.doc;
	return TIER.none;
}

/** Rank `types` for `rawQuery`, best match first. An empty query returns the input order. */
export function rankNodeTypes(types: NodeTypeInfo[], rawQuery: string): NodeTypeInfo[] {
	const q = rawQuery.trim().toLowerCase();
	if (!q) return types;

	const scored = types
		.map((t) => {
			const idx = t.type.toLowerCase().indexOf(q);
			return { t, tier: tierFor(t, q), idx: idx < 0 ? Number.MAX_SAFE_INTEGER : idx };
		})
		.filter((s) => s.tier !== TIER.none);

	scored.sort((a, b) => {
		if (a.tier !== b.tier) return b.tier - a.tier; // higher tier first
		if (a.idx !== b.idx) return a.idx - b.idx; // earlier in-name match first
		if (a.t.type.length !== b.t.type.length) return a.t.type.length - b.t.type.length; // tighter
		return a.t.type.localeCompare(b.t.type); // stable, alphabetical
	});

	return scored.map((s) => s.t);
}
