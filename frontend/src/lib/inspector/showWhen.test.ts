import { describe, it, expect } from 'vitest';
import { evalShowWhen, type ShowWhenPredicate } from './showWhen';

// The declarative field-dependency predicate (spec §5, D-N5). A field with no predicate always
// shows; a field whose predicate is false is hidden. Pure over the node's live param values.
// Fail-closed: a dependency on a param key ABSENT from `values` is false (hide) — a dependent whose
// controller isn't there cannot be shown. A key present with value `undefined` is a real value and
// follows the same rule as any other. Kept pure + unit-tested so ParamForm is a thin filter.

describe('evalShowWhen', () => {
	describe('equals', () => {
		it('is true when the controlling value strictly equals', () => {
			expect(evalShowWhen({ param: 'mode', equals: 'filter' }, { mode: 'filter' })).toBe(true);
		});
		it('is false when the value differs', () => {
			expect(evalShowWhen({ param: 'mode', equals: 'filter' }, { mode: 'raw' })).toBe(false);
		});
		it('is false when the controlling param is absent (fail-closed)', () => {
			expect(evalShowWhen({ param: 'mode', equals: 'filter' }, {})).toBe(false);
		});
		it('treats a present-but-undefined value as a real value, not an absent key', () => {
			expect(evalShowWhen({ param: 'mode', equals: undefined }, { mode: undefined })).toBe(true);
			expect(evalShowWhen({ param: 'mode', equals: 'filter' }, { mode: undefined })).toBe(false);
		});
	});

	describe('in', () => {
		it('is true when the value is a member of the set', () => {
			expect(evalShowWhen({ param: 'mode', in: ['filter', 'psd'] }, { mode: 'psd' })).toBe(true);
		});
		it('is false when the value is not a member', () => {
			expect(evalShowWhen({ param: 'mode', in: ['filter', 'psd'] }, { mode: 'raw' })).toBe(false);
		});
		it('is false when the controlling param is absent (fail-closed)', () => {
			expect(evalShowWhen({ param: 'mode', in: ['filter', 'psd'] }, {})).toBe(false);
		});
	});

	describe('truthy', () => {
		it('truthy:true is true for a truthy value, false for a falsy one', () => {
			expect(evalShowWhen({ param: 'advanced', truthy: true }, { advanced: true })).toBe(true);
			expect(evalShowWhen({ param: 'advanced', truthy: true }, { advanced: 1 })).toBe(true);
			expect(evalShowWhen({ param: 'advanced', truthy: true }, { advanced: false })).toBe(false);
			expect(evalShowWhen({ param: 'advanced', truthy: true }, { advanced: 0 })).toBe(false);
			expect(evalShowWhen({ param: 'advanced', truthy: true }, { advanced: '' })).toBe(false);
		});
		it('truthy:false inverts — true for a falsy value, false for a truthy one', () => {
			expect(evalShowWhen({ param: 'advanced', truthy: false }, { advanced: false })).toBe(true);
			expect(evalShowWhen({ param: 'advanced', truthy: false }, { advanced: true })).toBe(false);
		});
		it('is false when the controlling param is absent (fail-closed), for both polarities', () => {
			expect(evalShowWhen({ param: 'advanced', truthy: true }, {})).toBe(false);
			expect(evalShowWhen({ param: 'advanced', truthy: false }, {})).toBe(false);
		});
	});

	describe('function escape hatch', () => {
		it('calls the predicate with the live values', () => {
			const pred: ShowWhenPredicate = (v) => v.a === 1 && v.b === 2;
			expect(evalShowWhen(pred, { a: 1, b: 2 })).toBe(true);
			expect(evalShowWhen(pred, { a: 1, b: 3 })).toBe(false);
		});
		it('does not fail-close a function predicate — the function owns absence semantics', () => {
			const pred: ShowWhenPredicate = (v) => !('a' in v);
			expect(evalShowWhen(pred, {})).toBe(true);
		});
	});
});
