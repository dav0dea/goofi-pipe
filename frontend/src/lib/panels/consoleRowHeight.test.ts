import { describe, it, expect } from 'vitest';
import { COLLAPSE_LINES, LINE_H, PAD, BORDER, estimateRowHeight } from './consoleRowHeight';

describe('console row height model', () => {
	it('is the text box plus the padding and border the row really draws', () => {
		expect(estimateRowHeight(1, false)).toBe(LINE_H + PAD + BORDER);
		expect(estimateRowHeight(2, false)).toBe(2 * LINE_H + PAD + BORDER);
	});

	it('clamps a collapsed row to the lines it actually shows, and an expanded one to all of them', () => {
		const many = COLLAPSE_LINES + 5;
		expect(estimateRowHeight(many, false)).toBe(COLLAPSE_LINES * LINE_H + PAD + BORDER);
		expect(estimateRowHeight(many, true)).toBe(many * LINE_H + PAD + BORDER);
	});

	/* C16. Under the coarse floor every control a row hosts — the node chip, the copy button — is
	   --hit tall, so the ROW is, while its text still says one 16px line. The estimator has to carry
	   that floor or `layout.cum` is ~28px short for every row the ResizeObserver has not reached,
	   which on a long buffer is most of them, and the scrollbar lies about the log's length. */
	it('carries the content floor a coarse pointer imposes, without inflating the padding', () => {
		expect(estimateRowHeight(1, false, 44)).toBe(44 + PAD + BORDER);
		// The floor is a MINIMUM, not an override: a tall row is still its own height.
		expect(estimateRowHeight(6, true, 44)).toBe(6 * LINE_H + PAD + BORDER);
		// And it is absent by default, which is the fine-pointer answer.
		expect(estimateRowHeight(1, false)).toBeLessThan(estimateRowHeight(1, false, 44));
	});
});
