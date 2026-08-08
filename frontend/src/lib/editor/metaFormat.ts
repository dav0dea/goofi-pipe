/**
 * Pure formatting for the metadata inspector (no DOM, unit-tested).
 *
 * The inspector renders each TOP-LEVEL meta field as a collapsible section; this
 * module turns a field's value into readable text for that section's body:
 *   - scalars  → plain text (no quotes)
 *   - lists    → INLINE on one line (`[a, b, c]`); the DOM word-wraps long lists
 *                so they fill the width instead of one entry per line. Lists are
 *                capped at {@link ARRAY_CAP} elements so a huge coordinate array
 *                never builds a megabyte string on every (rAF-rate) frame.
 *   - dicts    → indented multi-line `key: value` (arrays within still inline)
 *
 * `metaPreview` is the short hint shown on the collapsed header — every field starts
 * collapsed, so it is the only thing most values ever show.
 */

/** Max list elements rendered inline before truncating with a `… (+N more)` tail. */
const ARRAY_CAP = 200;
/** Decimals a number keeps on the collapsed header line. */
const PREVIEW_DECIMALS = 2;

/** A numeric TypedArray (msgpack `bin` decodes to Uint8Array; a node may stash one
 * in meta). Render it like a plain list, not an indexed `{0:.., 1:..}` object. */
function isTypedArray(v: unknown): v is ArrayLike<number> {
	return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

function isList(v: unknown): v is ArrayLike<unknown> {
	return Array.isArray(v) || isTypedArray(v);
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
	return v !== null && typeof v === 'object' && !Array.isArray(v) && !isTypedArray(v);
}

function formatScalar(v: unknown): string {
	if (v === null) return 'null';
	if (v === undefined) return 'undefined';
	if (typeof v === 'bigint') return v.toString();
	if (typeof v === 'string') return v;
	return String(v);
}

/** Header-line form of a scalar: {@link formatScalar}, but a number written in plain
 * decimal is capped at {@link PREVIEW_DECIMALS} places with trailing zeros trimmed.
 * The header is a scannable hint — the expanded body is where the real value lives, so
 * it keeps full precision. A magnitude JS already writes in exponential form (|x| < 1e-6
 * or >= 1e21) is left alone: capping ITS decimals would drop the only digits it has. */
function formatScalarPreview(v: unknown): string {
	const s = formatScalar(v);
	if (typeof v !== 'number' || !Number.isFinite(v) || s.includes('e')) return s;
	const capped = v.toFixed(PREVIEW_DECIMALS).replace(/\.?0+$/, '');
	// A value that rounds away reads as 0; a sign on that zero would be noise, since
	// the header no longer carries the digit that earned it.
	return capped === '-0' ? '0' : capped;
}

/** Compact single-line form — used for lists (and any object nested inside one),
 * so a list never explodes into many lines. Long lists are capped. */
function formatInline(v: unknown): string {
	if (isList(v)) {
		const arr = Array.from(v as ArrayLike<unknown>);
		const head = arr.slice(0, ARRAY_CAP).map(formatInline).join(', ');
		const tail = arr.length > ARRAY_CAP ? `, … (+${arr.length - ARRAY_CAP} more)` : '';
		return '[' + head + tail + ']';
	}
	if (isPlainObject(v)) {
		return '{' + Object.entries(v).map(([k, x]) => `${k}: ${formatInline(x)}`).join(', ') + '}';
	}
	return formatScalar(v);
}

/** Reconstruct the TRUE original meta from a node-reduced frame (Option C, Change
 * C2). A reduced frame carries `meta.reduced = {axis: {orig_len, method, orig_coord?}}`
 * plus a reduced `shape` + co-reduced `channels`; the inspector must show what the
 * node actually produced, not the reduction artifacts. So we restore each reduced
 * axis's original length into `shape`, restore its coord from `orig_coord` when
 * carried (else drop the co-reduced coord — it's an artifact, not real metadata),
 * and hide the `reduced` key itself. No `reduced` block → returns the meta as-is. */
export function reconstructMeta(meta: Record<string, unknown>): Record<string, unknown> {
	const reduced = meta['reduced'];
	if (!isPlainObject(reduced)) return meta;
	const out: Record<string, unknown> = { ...meta };
	delete out['reduced'];
	const shape = Array.isArray(meta['shape']) ? [...(meta['shape'] as number[])] : null;
	const channels = isPlainObject(meta['channels'])
		? { ...(meta['channels'] as Record<string, unknown>) }
		: null;
	for (const [axisStr, info] of Object.entries(reduced)) {
		if (!isPlainObject(info)) continue;
		const axis = Number(axisStr);
		const origLen = typeof info['orig_len'] === 'number' ? (info['orig_len'] as number) : null;
		if (shape && origLen !== null && Number.isInteger(axis) && axis >= 0 && axis < shape.length) {
			shape[axis] = origLen;
		}
		if (channels) {
			const key = 'dim' + axisStr;
			if (Array.isArray(info['orig_coord'])) channels[key] = info['orig_coord'];
			else delete channels[key];
		}
	}
	if (shape) out['shape'] = shape;
	if (channels) out['channels'] = channels;
	return out;
}

/** Top-level entries of a meta dict, in insertion order. Reduction artifacts are
 * reconstructed to the node's true meta (see {@link reconstructMeta}); any
 * `__*__`-namespaced key (the codebase's idiom for internal, non-user markers) is
 * plumbing, not node metadata, so it's hidden from the inspector. */
export function metaEntries(meta: unknown): [string, unknown][] {
	if (!isPlainObject(meta)) return [];
	return Object.entries(reconstructMeta(meta)).filter(
		([k]) => !(k.startsWith('__') && k.endsWith('__'))
	);
}

/** Readable text for one meta value: lists inline, dicts indented multi-line. */
export function formatMetaValue(value: unknown, indent = 0): string {
	if (isList(value)) return formatInline(value);
	if (isPlainObject(value)) {
		const entries = Object.entries(value);
		if (entries.length === 0) return '{}';
		const pad = '  '.repeat(indent);
		return entries
			.map(([k, v]) =>
				isPlainObject(v) && Object.keys(v).length > 0
					? `${pad}${k}:\n${formatMetaValue(v, indent + 1)}`
					: `${pad}${k}: ${formatMetaValue(v, indent + 1)}`
			)
			.join('\n');
	}
	return formatScalar(value);
}

/** Short hint for a collapsed field header: container size, or the scalar itself. */
export function metaPreview(value: unknown): string {
	if (isList(value)) return `[${(value as ArrayLike<unknown>).length}]`;
	if (isPlainObject(value)) return `{${Object.keys(value).length}}`;
	const s = formatScalarPreview(value);
	return s.length > 60 ? s.slice(0, 59) + '…' : s;
}
