import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { decodeData, type ArrayData, type DataFrame } from './decode';

/**
 * Cross-language codec parity. The fixture (tests/codec_golden.json) is generated
 * by the PYTHON encoder (the wire-format source of truth) and is also asserted by
 * tests/test_codec.py. Decoding it here proves the hand-mirrored TS port stays in
 * lock-step with the Python encoder — any drift in tag/header/body fails both.
 */
const GOLDEN = JSON.parse(
	readFileSync(fileURLToPath(new URL('../../../../tests/codec_golden.json', import.meta.url)), 'utf-8')
) as { version: number; entries: { name: string; hex: string; decoded: Decoded }[] };

interface Decoded {
	dtype: 'ARRAY' | 'STRING' | 'TABLE';
	meta: Record<string, unknown>;
	arrayDtype?: string;
	shape?: number[];
	values?: number[];
	value?: string;
	entries?: Record<string, Decoded>;
}

function hexToBytes(hex: string): Uint8Array {
	const out = new Uint8Array(hex.length / 2);
	for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
	return out;
}

function checkDecoded(frame: DataFrame, exp: Decoded): void {
	expect(frame.dtype).toBe(exp.dtype);
	expect(frame.meta).toEqual(exp.meta);
	if (exp.dtype === 'ARRAY') {
		const a = frame.data as ArrayData;
		expect(a.dtype).toBe(exp.arrayDtype);
		expect(a.shape).toEqual(exp.shape);
		// Number() coerces BigInt64Array (i8/u8) elements; all fixture values < 2^53.
		expect(Array.from(a.values, (v) => Number(v))).toEqual(exp.values);
	} else if (exp.dtype === 'STRING') {
		expect(frame.data).toBe(exp.value);
	} else {
		const tbl = frame.data as Record<string, DataFrame>;
		expect(Object.keys(tbl).sort()).toEqual(Object.keys(exp.entries!).sort());
		for (const [k, sub] of Object.entries(exp.entries!)) checkDecoded(tbl[k], sub);
	}
}

describe('codec golden parity (py encode → ts decode)', () => {
	it('fixture targets wire version 2', () => {
		expect(GOLDEN.version).toBe(2);
	});

	for (const entry of GOLDEN.entries) {
		it(`decodes ${entry.name} identically to the Python encoder`, () => {
			checkDecoded(decodeData(hexToBytes(entry.hex)), entry.decoded);
		});
	}
});
