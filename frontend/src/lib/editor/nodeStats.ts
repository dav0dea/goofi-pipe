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

/** Labelled rows for the metadata panel, or `[]` when no stats have arrived yet.
 *
 * `drops` is the SELECTED SLOT's coalescing rate (`frames.dropRate`), which is why it sits here
 * rather than in the app header: fps up there is one number for the whole page, so a drop total
 * summed across streams could not be read against it. Null — no stream bound — omits the row
 * entirely, while a running stream dropping nothing still shows `0.0/s`, because that reading is
 * the one a user checks against and its absence would be ambiguous with "not running". */
export function nodeStatsRows(
	stats: NodeStats | null | undefined,
	drops?: number | null
): { label: string; value: string }[] {
	if (!stats) return [];
	const rows = [{ label: 'Update rate', value: `${fmtRate(stats.updates_per_second)} upd/s` }];
	if (drops != null) rows.push({ label: 'Dropped', value: `${drops.toFixed(1)}/s` });
	return rows;
}
