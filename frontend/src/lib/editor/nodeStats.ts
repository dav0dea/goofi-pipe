/**
 * Pure formatting of a node's execution telemetry (no DOM, unit-tested).
 *
 * The backend pushes `{updates_per_second, mean_process_ms, total_ticks}` on the
 * status plane (~1 Hz). This turns the two displayed metrics — update rate and
 * mean process() duration — into labelled rows for the metadata panel, with
 * number widths chosen to stay readable.
 */
import type { NodeStats } from '$lib/api/control';

/** Tick cadence, e.g. `12.4` / `1000` (integer once it gets big enough to be long). */
function fmtRate(ups: number): string {
	return ups >= 100 ? Math.round(ups).toString() : ups.toFixed(1);
}

/** Mean process() duration, narrowing decimals as the magnitude grows. */
function fmtMs(ms: number): string {
	if (ms >= 100) return Math.round(ms).toString();
	if (ms >= 10) return ms.toFixed(1);
	return ms.toFixed(2);
}

/** Labelled rows for the metadata panel, or `[]` when no stats have arrived yet. */
export function nodeStatsRows(stats: NodeStats | null | undefined): { label: string; value: string }[] {
	if (!stats) return [];
	return [
		{ label: 'Update rate', value: `${fmtRate(stats.updates_per_second)} upd/s` },
		{ label: 'Process time', value: `${fmtMs(stats.mean_process_ms)} ms` }
	];
}
