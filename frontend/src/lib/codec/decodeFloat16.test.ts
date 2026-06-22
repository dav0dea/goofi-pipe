import { describe, expect, it } from 'vitest';
import { decodeData, type ArrayData } from './decode';

/** Build a minimal GOOF ARRAY frame (no meta) with a raw body. */
function buildArrayFrame(dtypeStr: string, shape: number[], body: Uint8Array): ArrayBuffer {
	const dt = new TextEncoder().encode(dtypeStr);
	const bodyLen = 2 + dt.length + 4 * shape.length + body.length;
	const buf = new ArrayBuffer(14 + bodyLen);
	const view = new DataView(buf);
	const u8 = new Uint8Array(buf);
	u8.set([0x47, 0x4f, 0x4f, 0x46], 0); // 'GOOF'
	view.setUint8(4, 2); // version
	view.setUint8(5, 0); // ARRAY
	view.setUint32(6, 0, true); // meta_len = 0
	view.setUint32(10, bodyLen, true);
	let off = 14;
	view.setUint8(off++, shape.length); // ndim
	view.setUint8(off++, dt.length); // dtype_str_len
	u8.set(dt, off);
	off += dt.length;
	for (const s of shape) {
		view.setUint32(off, s, true);
		off += 4;
	}
	u8.set(body, off);
	return buf;
}

/** Half-float (IEEE-754 binary16) bit patterns, little-endian byte pairs. */
const HALF = {
	'1.0': [0x00, 0x3c],
	'2.0': [0x00, 0x40],
	'0.5': [0x00, 0x38],
	'0.0': [0x00, 0x00],
	'-1.0': [0x00, 0xbc]
};

describe('float16 (<f2) decode', () => {
	it('upcasts a half-float body to a Float32Array', () => {
		const body = new Uint8Array([...HALF['1.0'], ...HALF['2.0'], ...HALF['0.5']]);
		const frame = decodeData(buildArrayFrame('<f2', [3], body));
		const arr = (frame.data as ArrayData).values;
		expect(arr.length).toBe(3);
		expect(arr[0]).toBeCloseTo(1.0, 5);
		expect(arr[1]).toBeCloseTo(2.0, 5);
		expect(arr[2]).toBeCloseTo(0.5, 5);
		expect(arr instanceof Float32Array).toBe(true);
		expect((frame.data as ArrayData).dtype).toBe('<f2');
	});

	it('handles zero and negative half-floats', () => {
		const body = new Uint8Array([...HALF['0.0'], ...HALF['-1.0']]);
		const frame = decodeData(buildArrayFrame('<f2', [2], body));
		const arr = (frame.data as ArrayData).values;
		expect(arr[0]).toBe(0);
		expect(arr[1]).toBeCloseTo(-1.0, 5);
	});
});
