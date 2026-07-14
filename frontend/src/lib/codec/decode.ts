/**
 * TypeScript port of goofi.codec — decode side only.
 *
 * Mirrors the GOOF wire format documented in src/goofi/codec.py:
 *
 *   offset  size  field
 *   ------  ----  ----------------------------------------------
 *   0       4     magic "GOOF"
 *   4       1     version (=2)
 *   5       1     dtype tag (0=ARRAY, 1=STRING, 2=TABLE)
 *   6       4     meta_len   (LE u32) — msgpack-encoded meta dict
 *   10      4     body_len   (LE u32)
 *   14      *     meta bytes
 *   14+ml   *     body
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
	return decodeInto(view, 0).frame;
}

interface DecodeStep {
	frame: DataFrame;
	consumed: number;
}

function decodeInto(view: DataView, off: number): DecodeStep {
	checkMagic(view, off);
	const version = view.getUint8(off + 4);
	if (version !== 2) throw new Error(`Unsupported GOOF version ${version}`);
	const dtypeTag = view.getUint8(off + 5);
	const dtype = DTYPE_TAG[dtypeTag];
	if (!dtype) throw new Error(`Unknown dtype tag ${dtypeTag}`);
	const metaLen = view.getUint32(off + 6, true);
	const bodyLen = view.getUint32(off + 10, true);
	const headerEnd = off + 14;
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
		data = decodeArray(view, bodyStart, bodyEnd);
	} else if (dtype === 'STRING') {
		data = decoder.decode(new Uint8Array(view.buffer, view.byteOffset + bodyStart, bodyLen));
	} else {
		data = decodeTable(view, bodyStart);
	}
	return { frame: { dtype, data, meta }, consumed: bodyEnd - off };
}

function decodeArray(view: DataView, start: number, end: number): ArrayData {
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

function decodeTable(view: DataView, start: number): Record<string, DataFrame> {
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
		const sub = decodeInto(view, off);
		out[key] = sub.frame;
		off += valueLen;
	}
	return out;
}

/** IEEE-754 binary16 → number. JS has no reliable Float16Array, so upcast by hand. */
function halfToFloat(h: number): number {
	const sign = h & 0x8000 ? -1 : 1;
	const exp = (h & 0x7c00) >> 10;
	const frac = h & 0x03ff;
	if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024); // subnormal / zero
	if (exp === 0x1f) return frac ? NaN : sign * Infinity;
	return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
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
		case 'f2': {
			// Half-float (line/trajectory/topomap viewer-adapter transport): read each
			// LE u16 and upcast to a Float32Array so viewers consume float32 as before.
			const u16 = new Uint16Array(slice, 0, count);
			const out = new Float32Array(count);
			for (let i = 0; i < count; i++) out[i] = halfToFloat(u16[i]);
			return out;
		}
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
/** True if a numpy dtype string names a (little-endian / byte-agnostic) float. */
export function isFloatDtype(dtype: string): boolean {
	return dtype.startsWith('<f') || dtype.startsWith('|f') || dtype.startsWith('=f');
}
/** True if a numpy dtype string names an 8-bit unsigned integer (a raw image byte). */
export function isU8Dtype(dtype: string): boolean {
	return dtype === '<u1' || dtype === '|u1' || dtype === '=u1';
}
export function isStringFrame(f: DataFrame): f is DataFrame & { data: string } {
	return f.dtype === 'STRING';
}
export function isTableFrame(f: DataFrame): f is DataFrame & { data: Record<string, DataFrame> } {
	return f.dtype === 'TABLE';
}
