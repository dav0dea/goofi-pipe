import { describe, it, expect } from 'vitest';
import { controlKind } from './controlKind';
import type {
	BaseParam,
	FloatParam,
	IntParam,
	BoolParam,
	StringParam,
	UnknownParam
} from '$lib/api/types';

// The pure descriptor → control discriminant (spec §2, D-N2). First-match over the descriptor:
// expression_enabled wins over type; numeric = float|int; a plain bool
// is a toggle; a string with options OR that is refreshable is a select (an empty-but-refreshable
// list still gets a dropdown so its ⟳ re-scan survives); a plain string is text; anything else is
// unknown. Kept pure + unit-tested so ParamField is a thin switch and the mapping is one SSOT.
const base: Omit<BaseParam, 'value'> = {
	doc: null,
	refreshable: false,
	expression: null,
	expression_enabled: false,
	expression_triggers_process: false,
	expression_error: null
};

const floatParam = (over: Partial<FloatParam> = {}): FloatParam => ({
	...base,
	type: 'float',
	value: 0,
	vmin: 0,
	vmax: 1,
	...over
});
const intParam = (over: Partial<IntParam> = {}): IntParam => ({
	...base,
	type: 'int',
	value: 0,
	vmin: 0,
	vmax: 10,
	...over
});
const boolParam = (over: Partial<BoolParam> = {}): BoolParam => ({
	...base,
	type: 'bool',
	value: false,
	...over
});
const stringParam = (over: Partial<StringParam> = {}): StringParam => ({
	...base,
	type: 'string',
	value: '',
	options: null,
	...over
});
const unknownParam = (over: Partial<UnknownParam> = {}): UnknownParam => ({
	...base,
	type: 'unknown',
	value: null,
	...over
});

describe('controlKind', () => {
	it('maps float and int to numeric', () => {
		expect(controlKind(floatParam())).toBe('numeric');
		expect(controlKind(intParam())).toBe('numeric');
	});

	it('maps a string with a non-empty options list to select', () => {
		expect(controlKind(stringParam({ options: ['a'] }))).toBe('select');
	});

	it('maps a refreshable string with an EMPTY options list to select (⟳ re-scan survives)', () => {
		expect(controlKind(stringParam({ options: [], refreshable: true }))).toBe('select');
	});

	it('maps a refreshable string with null options to select', () => {
		expect(controlKind(stringParam({ options: null, refreshable: true }))).toBe('select');
	});

	it('maps a plain string (no options, not refreshable) to text', () => {
		expect(controlKind(stringParam({ options: [], refreshable: false }))).toBe('text');
		expect(controlKind(stringParam({ options: null, refreshable: false }))).toBe('text');
	});

	it('maps an unknown param to unknown', () => {
		expect(controlKind(unknownParam())).toBe('unknown');
	});

	it('lets expression_enabled override the type (a fx-active float is expression, not numeric)', () => {
		expect(controlKind(floatParam({ expression_enabled: true }))).toBe('expression');
	});

	it('lets expression_enabled override every other type', () => {
		expect(controlKind(intParam({ expression_enabled: true }))).toBe('expression');
		expect(controlKind(boolParam({ expression_enabled: true }))).toBe('expression');
		expect(controlKind(stringParam({ options: ['a'], expression_enabled: true }))).toBe('expression');
		expect(controlKind(unknownParam({ expression_enabled: true }))).toBe('expression');
	});
});
