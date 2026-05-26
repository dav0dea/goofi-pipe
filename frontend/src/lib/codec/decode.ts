/**
 * TypeScript port of goofi.codec — decode side only.
 *
 * Mirrors the GOOF wire format documented in src/goofi/codec.py:
 *
 *   offset  size  field
 *   ------  ----  ----------------------------------------------
 *   0       4     magic "GOOF"
 *   4       1     version (=1)
 *   5       1     dtype tag (0=ARRAY, 1=STRING, 2=TABLE)
 *   6       2     meta_len   (LE u16) — msgpack-encoded meta dict
 *   8       4     body_len   (LE u32)
 *   12      *     meta bytes
 *   12+ml   *     body
 *
 * ARRAY body:
 *   u8 ndim, u8 dtype_str_len, dtype_str ascii, ndim × u32 shape, raw bytes
 * STRING body:
 *   utf-8
 * TABLE body:
 *   u32 n, then repeated: u16 key_len, key utf-8, u32 value_len, encoded Data
 *
 * NOTE: in Python the DataType enum starts at ARRAY=0 — see goofi/data.py.
 */
import { decode as msgpackDecode } from '@msgpack/msgpack';

export type DataType = 'ARRAY' | 'STRING' | 'TABLE';

const DTYPE_TAG: Record<number, DataType> = {
	0: 'ARRAY',
	1: 'STRING',
	2: 'TABLE'
};

/** A decoded Data frame. */
export interface DataFrame {
	dtype: DataType;
	/** ARRAY → TypedArray + shape; STRING → string; TABLE → Record<string, DataFrame> */
	data: ArrayData | string | Record<string, DataFrame>;
	meta: Record<string, unknown>;
}

export interface ArrayData {
	/** numpy dtype string, e.g. '<f4', '<i8', '|u1'. */
	dtype: string;
	shape: number[];
	/** Linear element buffer in row-major order. */
	values: ArrayLike<number> & { length: number };
}

const MAGIC = new Uint8Array([0x47, 0x4f, 0x4f, 0x46]); // 'GOOF'

const decoder = new TextDecoder('utf-8');

function checkMagic(view: DataView, off: number): void {
	if (
		view.getUint8(off) !== MAGIC[0] ||
		view.getUint8(off + 1) !== MAGIC[1] ||
		view.getUint8(off + 2) !== MAGIC[2] ||
		view.getUint8(off + 3) !== MAGIC[3]
	) {
		throw new Error('Invalid Data frame: bad magic');
	}
}

/** Decode an encoded GOOF buffer into a DataFrame. */
export function decodeData(buf: ArrayBuffer | Uint8Array, offset = 0, length?: number): DataFrame {
	const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
	const view = new DataView(bytes.buffer, bytes.byteOffset + offset, length ?? bytes.byteLength - offset);
	return decodeInto(bytes, view, 0).frame;
}

interface DecodeStep {
	frame: DataFrame;
	consumed: number;
}

function decodeInto(bytes: Uint8Array, view: DataView, off: number): DecodeStep {
	checkMagic(view, off);
	const version = view.getUint8(off + 4);
	if (version !== 1) throw new Error(`Unsupported GOOF version ${version}`);
	const dtypeTag = view.getUint8(off + 5);
	const dtype = DTYPE_TAG[dtypeTag];
	if (!dtype) throw new Error(`Unknown dtype tag ${dtypeTag}`);
	const metaLen = view.getUint16(off + 6, true);
	const bodyLen = view.getUint32(off + 8, true);
	const headerEnd = off + 12;
	const meta =
		metaLen > 0
			? ((): Record<string, unknown> => {
					const slice = new Uint8Array(view.buffer, view.byteOffset + headerEnd, metaLen);
					const m = msgpackDecode(slice);
					return (m && typeof m === 'object' ? (m as Record<string, unknown>) : {}) ?? {};
				})()
			: {};
	const bodyStart = headerEnd + metaLen;
	const bodyEnd = bodyStart + bodyLen;
	let data: ArrayData | string | Record<string, DataFrame>;
	if (dtype === 'ARRAY') {
		data = decodeArray(bytes, view, bodyStart, bodyEnd);
	} else if (dtype === 'STRING') {
		data = decoder.decode(new Uint8Array(view.buffer, view.byteOffset + bodyStart, bodyLen));
	} else {
		data = decodeTable(bytes, view, bodyStart);
	}
	return { frame: { dtype, data, meta }, consumed: bodyEnd - off };
}

function decodeArray(bytes: Uint8Array, view: DataView, start: number, end: number): ArrayData {
	let off = start;
	const ndim = view.getUint8(off);
	off += 1;
	const dtypeStrLen = view.getUint8(off);
	off += 1;
	const dtypeStr = decoder.decode(new Uint8Array(view.buffer, view.byteOffset + off, dtypeStrLen));
	off += dtypeStrLen;
	const shape: number[] = [];
	for (let i = 0; i < ndim; i++) {
		shape.push(view.getUint32(off, true));
		off += 4;
	}
	const nBytes = end - off;
	const values = readTypedArray(dtypeStr, view.buffer, view.byteOffset + off, nBytes);
	return { dtype: dtypeStr, shape, values };
}

function decodeTable(
	bytes: Uint8Array,
	view: DataView,
	start: number
): Record<string, DataFrame> {
	let off = start;
	const n = view.getUint32(off, true);
	off += 4;
	const out: Record<string, DataFrame> = {};
	for (let i = 0; i < n; i++) {
		const keyLen = view.getUint16(off, true);
		off += 2;
		const key = decoder.decode(new Uint8Array(view.buffer, view.byteOffset + off, keyLen));
		off += keyLen;
		const valueLen = view.getUint32(off, true);
		off += 4;
		const sub = decodeInto(bytes, view, off);
		out[key] = sub.frame;
		off += valueLen;
	}
	return out;
}

function readTypedArray(
	dtypeStr: string,
	buffer: ArrayBufferLike,
	byteOffset: number,
	nBytes: number
): ArrayLike<number> & { length: number } {
	// numpy dtype strings start with byteorder ('<', '>', '=', '|') then a
	// kind char (i/u/f/b) then itemsize. We only support little-endian or
	// byte-agnostic types (most goofi data is one of those — and js
	// TypedArrays are LE on every platform we care about).
	const bo = dtypeStr.charAt(0);
	if (bo === '>') {
		throw new Error(`Big-endian arrays unsupported: ${dtypeStr}`);
	}
	const tail = bo === '<' || bo === '=' || bo === '|' ? dtypeStr.slice(1) : dtypeStr;
	const kind = tail.charAt(0);
	const itemsize = parseInt(tail.slice(1), 10);
	const count = nBytes / itemsize;
	// Slice into a fresh buffer so the consumer can outlive the WS message
	// frame (which may be reused by the runtime).
	const slice = buffer.slice(byteOffset, byteOffset + nBytes);
	switch (kind + itemsize) {
		case 'f4':
			return new Float32Array(slice, 0, count);
		case 'f8':
			return new Float64Array(slice, 0, count);
		case 'i1':
		case 'b1':
			return new Int8Array(slice, 0, count);
		case 'i2':
			return new Int16Array(slice, 0, count);
		case 'i4':
			return new Int32Array(slice, 0, count);
		case 'i8':
			return new BigInt64Array(slice, 0, count) as unknown as ArrayLike<number> & {
				length: number;
			};
		case 'u1':
			return new Uint8Array(slice, 0, count);
		case 'u2':
			return new Uint16Array(slice, 0, count);
		case 'u4':
			return new Uint32Array(slice, 0, count);
		case 'u8':
			return new BigUint64Array(slice, 0, count) as unknown as ArrayLike<number> & {
				length: number;
			};
		default:
			throw new Error(`Unsupported numpy dtype: ${dtypeStr}`);
	}
}

/** Helpers consumed by viewers. */
export function isArrayFrame(f: DataFrame): f is DataFrame & { data: ArrayData } {
	return f.dtype === 'ARRAY';
}
export function isStringFrame(f: DataFrame): f is DataFrame & { data: string } {
	return f.dtype === 'STRING';
}
export function isTableFrame(f: DataFrame): f is DataFrame & { data: Record<string, DataFrame> } {
	return f.dtype === 'TABLE';
}

/** Stride along the given dimension (row-major). */
export function strideOf(shape: number[], dim: number): number {
	let s = 1;
	for (let i = dim + 1; i < shape.length; i++) s *= shape[i];
	return s;
}
