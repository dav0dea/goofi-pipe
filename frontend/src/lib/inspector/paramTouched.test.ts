import { describe, expect, it } from 'vitest';
import type { ParamDescriptor } from '$lib/api/types';
import { isModified, touchedCount } from './paramTouched';

const base = {
	doc: null,
	refreshable: false,
	expression: null,
	mode: 'constant',
	reference: null,
	triggers: false,
	error: null
} as const;

const float = (value: number, dflt: number): ParamDescriptor =>
	({ ...base, type: 'float', value, default: dflt, vmin: 0, vmax: 1 }) as ParamDescriptor;

describe('isModified', () => {
	it('is false for a param still sitting on its declared default', () => {
		expect(isModified(float(0.5, 0.5))).toBe(false);
	});

	it('is true once the value has moved — which is how a knob turned in a plugin window shows up', () => {
		expect(isModified(float(0.7, 0.5))).toBe(true);
	});

	it('is true for a driven param even when its value happens to equal the default', () => {
		const driven = { ...float(0.5, 0.5), mode: 'expression', expression: 't' } as ParamDescriptor;
		expect(isModified(driven)).toBe(true);
	});

	it('is false for a pulse, which holds no value to compare', () => {
		const pulse = { ...base, type: 'pulse', value: null, default: null } as ParamDescriptor;
		expect(isModified(pulse)).toBe(false);
	});

	it('treats a string param off its default as touched', () => {
		const s = { ...base, type: 'string', value: 'hard', default: 'soft', options: null } as ParamDescriptor;
		expect(isModified(s)).toBe(true);
	});
});

describe('touchedCount', () => {
	it('counts across every group, so the badge matches what a filter would show', () => {
		expect(
			touchedCount({
				osc: { tune: float(0.2, 0.5), shape: float(0.5, 0.5) },
				env1: { attack: float(0.9, 0.5) }
			})
		).toBe(2);
	});

	it('is zero for a node nobody has touched', () => {
		expect(touchedCount({ osc: { tune: float(0.5, 0.5) } })).toBe(0);
	});

	it('is zero for no node at all', () => {
		expect(touchedCount(undefined)).toBe(0);
	});
});
