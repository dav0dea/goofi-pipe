/**
 * Pure formatting of a node's execution telemetry (no DOM, unit-tested).
 *
 * The backend pushes `{updates_per_second}` on the status plane (2 Hz). This turns
 * it into the header readout and a labelled row for the metadata panel, with number
 * widths chosen to stay readable.
 */
import type { NodeStats } from '$lib/api/control';

/** Tick cadence, e.g. `12.4` / `1000` (integer once it gets big enough to be long). */
function fmtRate(ups: number): string {
	return ups >= 100 ? Math.round(ups).toString() : ups.toFixed(1);
}

/** Just the update rate (`12.4 upd/s`) for the compact, glanceable node-header
 * readout. Null until the node's first NODE_STATS push. */
export function formatUpdateRate(stats: NodeStats | null | undefined): string | null {
	if (!stats) return null;
	return `${fmtRate(stats.updates_per_second)} upd/s`;
}

/** Labelled rows for the metadata panel, or `[]` when no stats have arrived yet. */
export function nodeStatsRows(stats: NodeStats | null | undefined): { label: string; value: string }[] {
	if (!stats) return [];
	return [{ label: 'Update rate', value: `${fmtRate(stats.updates_per_second)} upd/s` }];
}
