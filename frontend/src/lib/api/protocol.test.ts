import { describe, expect, it } from 'vitest';
import { PROTOCOL_VERSION, isProtocolCompatible } from './control';

describe('control protocol version', () => {
	it('accepts an exact match', () => {
		expect(isProtocolCompatible(PROTOCOL_VERSION)).toBe(true);
	});

	it('rejects a newer or older backend version', () => {
		expect(isProtocolCompatible(PROTOCOL_VERSION + 1)).toBe(false);
		expect(isProtocolCompatible(PROTOCOL_VERSION - 1)).toBe(false);
	});

	it('rejects an absent/non-numeric version (backend predates the field = skew)', () => {
		expect(isProtocolCompatible(undefined)).toBe(false);
		expect(isProtocolCompatible(null)).toBe(false);
		expect(isProtocolCompatible('1')).toBe(false);
	});
});
