import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/* The empty-canvas hint is the ONLY `<kbd>` in the entire UI — the single place the app names a
 * keyboard route to the user — and it named `+`, which no handler in `frontend/src` matches (and
 * `@xyflow`'s Pane registers no keydown at all). It rendered on the first-run screen and on the
 * screen right after creating a sub-patch.
 *
 * Guarded here rather than in e2e because the hint only renders when the editor's scope is EMPTY,
 * and the suite's shared backend patch is not empty by the time any spec could assert on it. This
 * pins the two ends against each other inside the one file that holds both: the word the hint
 * prints, and the keydown branch that word promises. `menu-position.spec.ts` separately proves that
 * pressing that key really does open the menu, so the chain is complete.
 */

const SRC = readFileSync(
	fileURLToPath(new URL('./NodeEditorPanel.svelte', import.meta.url)),
	'utf8'
);

/** Escape a key name for embedding in a RegExp (`+` is a quantifier). */
function esc(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

describe('the empty-canvas keyboard hint', () => {
	it('names exactly one key', () => {
		expect([...SRC.matchAll(/<kbd>([^<]+)<\/kbd>/g)].map((m) => m[1])).toHaveLength(1);
	});

	it('names a key whose keydown branch opens the add-node menu', () => {
		const key = /<kbd>([^<]+)<\/kbd>/.exec(SRC)![1];
		// `[^{]*\)` rather than a bare `\)`: a branch may carry further conjuncts after the key test
		// (R-C scoped Tab to `!e.shiftKey && canvasHasKeys(t)`), and this guard is about WHICH key the
		// hint names, not about how narrowly that key is claimed. The body capture still stops at the
		// first `}`, so a comment inside the branch must not spell one.
		const branch = new RegExp(`e\\.key === '${esc(key)}'[^{]*\\)\\s*\\{([^}]*)`).exec(SRC);
		expect(branch, `no keydown branch in this file matches \`${key}\``).not.toBeNull();
		expect(branch![1], `the \`${key}\` branch is not the add-node menu`).toContain('openAddMenu');
	});
});
