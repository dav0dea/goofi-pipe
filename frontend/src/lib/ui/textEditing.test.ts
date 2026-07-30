import { describe, it, expect } from 'vitest';
import { isTextEditingTarget } from './textEditing';

/* The guard both keyboard scopes ask, now that a text-editing surface in this app is not necessarily
 * a form element. `INPUT | TEXTAREA | SELECT` was a complete answer for as long as every editable
 * thing was one of those; X's expression editor is a contenteditable `<div>`, so the same Ctrl+Z that
 * used to undo TEXT would have undone the GRAPH, and the same Ctrl+A would have selected every node
 * instead of the expression. */

/** A duck-typed target — the two fields the predicate reads, so it needs no DOM. */
const target = (tagName: string, isContentEditable = false) =>
	({ tagName, isContentEditable }) as unknown as EventTarget;

describe('isTextEditingTarget', () => {
	for (const tag of ['INPUT', 'TEXTAREA', 'SELECT']) {
		it(`${tag} is a text-editing target`, () => expect(isTextEditingTarget(target(tag))).toBe(true));
	}

	it('a contenteditable element is one too, whatever its tag', () => {
		expect(isTextEditingTarget(target('DIV', true))).toBe(true);
		expect(isTextEditingTarget(target('PRE', true))).toBe(true);
	});

	it('an ordinary element is not', () => {
		expect(isTextEditingTarget(target('DIV'))).toBe(false);
		expect(isTextEditingTarget(target('BUTTON'))).toBe(false);
		expect(isTextEditingTarget(target('BODY'))).toBe(false);
	});

	it('no target at all is not', () => {
		expect(isTextEditingTarget(null)).toBe(false);
	});
});
