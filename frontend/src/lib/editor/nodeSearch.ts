/** The add-node menu's facets, and the ranking of its search on the bare name. */
import type { NodeTypeInfo } from '$lib/api/control';
import { TAGS } from '$lib/api/vocab';
import { bareName, engineOf } from './typeId';
import { nodeTypeSource } from './nodeTypeSource';

type Tag = NodeTypeInfo['tags'][number];

/** The palette tab that facets nothing. */
export const ALL_TAB = 'all';
/** The tab a plugin wears instead of its engine's, so the two never list one type twice. */
export const VST_TAB = 'vst';

/** Match quality, best first. */
const TIER = {
	none: 0,
	doc: 1,
	source: 2,
	tags: 3,
	nameSubstring: 4,
	nameWord: 5, // query starts a CamelCase / separated word inside the name
	namePrefix: 6,
	nameExact: 7
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
	const bare = bareName(t.type);
	const name = bare.toLowerCase();
	if (name === q) return TIER.nameExact;
	if (name.startsWith(q)) return TIER.namePrefix;
	if (hasWordStartMatch(bare, name, q)) return TIER.nameWord;
	if (name.includes(q)) return TIER.nameSubstring;
	if (t.tags.some((tag) => tag.includes(q))) return TIER.tags;
	if (nodeTypeSource(t).includes(q)) return TIER.source;
	if (t.doc.toLowerCase().includes(q)) return TIER.doc;
	return TIER.none;
}

/** Rank `types` for `rawQuery`, best match first. An empty query returns the input order. */
export function rankNodeTypes(types: NodeTypeInfo[], rawQuery: string): NodeTypeInfo[] {
	const q = rawQuery.trim().toLowerCase();
	if (!q) return types;

	const scored = types
		.map((t) => {
			const name = bareName(t.type);
			const idx = name.toLowerCase().indexOf(q);
			return { t, name, tier: tierFor(t, q), idx: idx < 0 ? Number.MAX_SAFE_INTEGER : idx };
		})
		.filter((s) => s.tier !== TIER.none);

	scored.sort((a, b) => {
		if (a.tier !== b.tier) return b.tier - a.tier; // higher tier first
		if (a.idx !== b.idx) return a.idx - b.idx; // earlier in-name match first
		if (a.name.length !== b.name.length) return a.name.length - b.name.length; // tighter
		return a.name.localeCompare(b.name); // stable, alphabetical
	});

	return scored.map((s) => s.t);
}

/** The tab `t` belongs to: its engine, or `vst` where an engine found it on its own account. */
export function tabOf(t: NodeTypeInfo): string | null {
	return t.source === 'plugin' ? VST_TAB : engineOf(t.type);
}

/** The tabs `types` offer, `all` first. A structural type has no engine and so no tab of its own. */
export function paletteTabs(types: NodeTypeInfo[]): string[] {
	const tabs: string[] = [];
	for (const t of types) {
		const tab = tabOf(t);
		if (tab && !tabs.includes(tab)) tabs.push(tab);
	}
	// A plugin format is not an engine, so its tab sits after every engine's.
	return [ALL_TAB, ...tabs.filter((t) => t !== VST_TAB), ...tabs.filter((t) => t === VST_TAB)];
}

/** `types` on one tab. `all` keeps every type, a structural one included. */
export function byTab(types: NodeTypeInfo[], tab: string): NodeTypeInfo[] {
	if (tab === ALL_TAB) return types;
	return types.filter((t) => tabOf(t) === tab);
}

/** `types` carrying EVERY selected tag, so each chip narrows what the one before it left. */
export function byTags(types: NodeTypeInfo[], tags: readonly Tag[]): NodeTypeInfo[] {
	if (tags.length === 0) return types;
	return types.filter((t) => tags.every((tag) => t.tags.includes(tag)));
}

/** The tags `types` carry, in the vocabulary's order: a chip that would show nothing is not offered. */
export function facetTags(types: NodeTypeInfo[]): Tag[] {
	return TAGS.filter((tag) => types.some((t) => t.tags.includes(tag)));
}
