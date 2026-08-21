import type { DataFrame } from '$lib/codec/decode';

/** Child (key, frame) pairs of a TABLE frame; empty for a non-table leaf. */
export function tableChildren(f: DataFrame): [string, DataFrame][] {
	return f.dtype === 'TABLE' ? Object.entries(f.data as Record<string, DataFrame>) : [];
}

/** A one-line summary of a table cell, used as the collapsed label in TableTree. */
export function leafSummary(f: DataFrame, decimals: number): string {
	if (f.dtype === 'STRING') return String(f.data).slice(0, 120);
	if (f.dtype === 'TABLE') return `{${tableChildren(f).length} fields}`;
	const a = f.data as { shape: number[]; values: ArrayLike<number> };
	if (a.shape.length === 0 || (a.shape.length === 1 && a.shape[0] === 1)) {
		const n = Number(a.values[0]);
		return Number.isFinite(n) ? n.toFixed(Math.max(0, Math.min(10, decimals))) : String(n);
	}
	return `array[${a.shape.join('×')}]`;
}
