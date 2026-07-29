/**
 * The resolver behind `TitleTip` — which `title=` a press is asking about.
 *
 * The app carries ~40 `title=` tooltips, and a dozen of them are the ONLY rendering of what they
 * say: the palette's "missing dependency: …", a connector pill's dtype, "Double-click to rename".
 * On a device with no hover none of that is reachable, which CLAUDE.md forbids outright. The door
 * is one long-press layer over the whole app rather than a coarse cue per site, so a tooltip added
 * anywhere later is reachable on touch by construction.
 *
 * Pure and DOM-free (the repo's vitest has no jsdom): it asks a node only for its `title` and its
 * parent, which every `HTMLElement` answers structurally.
 */

/** The two things the walk needs. A real `HTMLElement` satisfies this without a cast. */
export interface Titled {
	getAttribute(name: string): string | null;
	readonly parentElement: Titled | null;
}

/**
 * The innermost element at or above `from` that carries a non-blank `title`, with that text
 * trimmed. Innermost wins because tooltips nest — a console row's copy button sits inside a row
 * inside a panel — and the thing under the finger is the thing being asked about.
 */
export function nearestTitle(from: Titled | null): { el: Titled; text: string } | null {
	for (let n = from; n; n = n.parentElement) {
		const text = n.getAttribute('title')?.trim();
		if (text) return { el: n, text };
	}
	return null;
}
