import { describe, it, expect } from 'vitest';
import { seedSlot } from './seedSlot';
import { typeInfo } from '$lib/test/typeInfo';

const type = (input_slots: Record<string, string>, output_slots: Record<string, string>) =>
	typeInfo({ input_slots, output_slots });
const seed = (side: 'source' | 'target', dtype: string) =>
	({ node: 'n', slot: 's', side, dtype }) as Parameters<typeof seedSlot>[0];

describe('the seed rule is the manager\'s link rule', () => {
	it('an audio output seeds an audio input, or an array one through the tap', () => {
		expect(seedSlot(seed('source', 'AUDIO'), type({ a: 'AUDIO' }, {}))).toBe('a');
		expect(seedSlot(seed('source', 'AUDIO'), type({ x: 'ARRAY' }, {}))).toBe('x');
		expect(seedSlot(seed('source', 'AUDIO'), type({ s: 'STRING' }, {}))).toBeUndefined();
	});
	it('nothing but audio feeds an audio input, and an array input takes an audio source', () => {
		expect(seedSlot(seed('target', 'AUDIO'), type({}, { out: 'ARRAY' }))).toBeUndefined();
		expect(seedSlot(seed('target', 'ARRAY'), type({}, { out: 'AUDIO' }))).toBe('out');
		expect(seedSlot(seed('target', 'STRING'), type({}, { out: 'AUDIO' }))).toBeUndefined();
	});
});
