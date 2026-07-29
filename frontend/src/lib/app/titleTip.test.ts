import { describe, it, expect } from 'vitest';
import { nearestTitle, type Titled } from './titleTip';

/** A DOM-shaped stub: `titleTip` only ever asks for the attribute and the parent. */
function el(title: string | null, parent: Titled | null = null): Titled {
	return { getAttribute: (n: string) => (n === 'title' ? title : null), parentElement: parent };
}

describe('nearestTitle', () => {
	it('finds the title on the pressed element itself', () => {
		const target = el('missing dependency: numpy');
		expect(nearestTitle(target)).toEqual({ el: target, text: 'missing dependency: numpy' });
	});

	it('walks up to the nearest titled ancestor', () => {
		// The real shape this exists for: a node's connector pill carries the dtype title, and the
		// press lands on the SvelteFlow handle drawn inside it.
		const pill = el('array');
		const handle = el(null, pill);
		expect(nearestTitle(handle)?.el).toBe(pill);
	});

	it('stops at the FIRST titled ancestor, so the innermost tooltip wins', () => {
		const outer = el('Change panel content');
		const inner = el('Copy message', outer);
		expect(nearestTitle(el(null, inner))?.text).toBe('Copy message');
	});

	it('is null when nothing in the chain carries a title', () => {
		expect(nearestTitle(el(null, el(null)))).toBeNull();
		expect(nearestTitle(null)).toBeNull();
	});

	it('treats a blank title as no title, and keeps walking past it', () => {
		// `title=""` is how an attribute spread lands when the value is undefined — it must not
		// open an empty bubble, nor mask a real tooltip further up.
		const outer = el('Restore');
		expect(nearestTitle(el('   ', outer))?.text).toBe('Restore');
		expect(nearestTitle(el(''))).toBeNull();
	});

	it('trims the text it reports', () => {
		expect(nearestTitle(el('  update rate  '))?.text).toBe('update rate');
	});
});
