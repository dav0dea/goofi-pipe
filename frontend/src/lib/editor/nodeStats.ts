/**
 * Pure formatting of a node's execution telemetry (no DOM, unit-tested).
 *
 * The backend pushes `{updates_per_second, mean_process_ms, total_ticks}` on the
 * status plane (~1 Hz). This turns it into a compact canvas overlay and a labelled
 * row set for the inspector, with number widths chosen to stay readable on a node.
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

/** A compact one-line overlay (`12.4 upd/s · 3.12 ms`), or null when no stats
 * have arrived yet (the node hasn't pushed its first NODE_STATS). */
export function formatNodeStats(stats: NodeStats | null | undefined): string | null {
	if (!stats) return null;
	return `${fmtRate(stats.updates_per_second)} upd/s · ${fmtMs(stats.mean_process_ms)} ms`;
}

/** Labelled rows for the inspector, or `[]` when no stats have arrived yet. */
export function nodeStatsRows(stats: NodeStats | null | undefined): { label: string; value: string }[] {
	if (!stats) return [];
	return [
		{ label: 'Update rate', value: `${fmtRate(stats.updates_per_second)} upd/s` },
		{ label: 'Process time', value: `${fmtMs(stats.mean_process_ms)} ms` },
		{ label: 'Total ticks', value: stats.total_ticks.toString() }
	];
}
