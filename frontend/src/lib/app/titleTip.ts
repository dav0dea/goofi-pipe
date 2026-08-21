/** DOM-free so the walk runs without jsdom; a real `HTMLElement` satisfies it without a cast. */
export interface Titled {
	getAttribute(name: string): string | null;
	readonly parentElement: Titled | null;
}

/** The innermost element at or above `from` with a non-blank `title`, trimmed. */
export function nearestTitle(from: Titled | null): { el: Titled; text: string } | null {
	for (let n = from; n; n = n.parentElement) {
		const text = n.getAttribute('title')?.trim();
		if (text) return { el: n, text };
	}
	return null;
}
