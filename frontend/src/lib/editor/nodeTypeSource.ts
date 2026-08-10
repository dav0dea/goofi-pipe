/** The one word a palette row carries beside a node's name.
 *
 * The add menu is a flat list — categories were the grouping AND the trailing label, and both are
 * gone. What replaces them is provenance: a node the open patch brought with it looks exactly like
 * a shipped one otherwise, and that is the difference worth a word.
 *
 * `unavailable` wins over provenance because it used to be a category too, and its meaning must
 * survive the category's removal: greyed-and-unclickable says a row cannot be picked, not why. (The
 * reason itself is the row's `title`, phrased by the backend — see `nodeTypeTitle`.)
 *
 * Lives here rather than inside the component so the precedence is unit-testable.
 */
import type { NodeTypeInfo } from '$lib/api/control';

export function nodeTypeSource(t: NodeTypeInfo): string {
	if (!t.available) return 'unavailable';
	return t.source === 'patch' ? 'this patch' : 'builtin';
}
