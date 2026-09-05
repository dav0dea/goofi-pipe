import { describe, it, expect } from 'vitest';
import { controlKind } from './controlKind';
import type {
	BaseParam,
	FloatParam,
	IntParam,
	BoolParam,
	StringParam,
	PulseParam,
	UnknownParam
} from '$lib/api/types';

// The pure descriptor → control discriminant (spec §2, D-N2). First-match over the descriptor: a
// pulse is a button whatever its mode; a driven mode wins over type; numeric = float|int; a plain
// bool is a toggle; a string with options OR that is refreshable is a select (an
// empty-but-refreshable list still gets a dropdown so its ⟳ re-scan survives); a plain string is
// text; anything else is unknown. Kept pure + unit-tested so ParamField is a thin switch and the
// mapping is one SSOT.
const base: Omit<BaseParam, 'value'> = {
	doc: null,
	refreshable: false,
	expression: null,
	mode: 'constant',
	reference: null,
	triggers: false,
	error: null
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
const pulseParam = (over: Partial<PulseParam> = {}): PulseParam => ({
	...base,
	type: 'pulse',
	value: null,
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

	it('maps a pulse to its own kind (a button, and no value)', () => {
		expect(controlKind(pulseParam())).toBe('pulse');
	});

	it('keeps a pulse a pulse in reference mode, so the button stays and the chips show', () => {
		expect(controlKind(pulseParam({ mode: 'reference', reference: 'clock.out' }))).toBe('pulse');
	});

	it('maps an unknown param to unknown', () => {
		expect(controlKind(unknownParam())).toBe('unknown');
	});

	it('lets the expression mode override the type (a driven float is expression, not numeric)', () => {
		expect(controlKind(floatParam({ mode: 'expression' }))).toBe('expression');
	});

	it('lets a driven mode override every other type, and the reference mode is its own control', () => {
		expect(controlKind(intParam({ mode: 'expression' }))).toBe('expression');
		expect(controlKind(boolParam({ mode: 'expression' }))).toBe('expression');
		expect(controlKind(stringParam({ options: ['a'], mode: 'expression' }))).toBe('expression');
		expect(controlKind(unknownParam({ mode: 'expression' }))).toBe('expression');
		expect(controlKind(floatParam({ mode: 'reference' }))).toBe('reference');
		expect(controlKind(stringParam({ mode: 'reference' }))).toBe('reference');
	});
});
