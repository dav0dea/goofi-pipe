/** Pure formatting of a node's execution telemetry. */
import type { NodeStats } from '$lib/api/control';

/** Tick cadence, e.g. `12.4` / `1000`. */
function fmtRate(ups: number): string {
	return ups >= 100 ? Math.round(ups).toString() : ups.toFixed(1);
}

/** The node header's update-rate readout; null until the first NODE_STATS push. */
export function formatUpdateRate(stats: NodeStats | null | undefined): string | null {
	if (!stats) return null;
	return `${fmtRate(stats.updates_per_second)} upd/s`;
}

/** Labelled rows for the metadata panel, or `[]` before any stats arrive. `drops` is the selected
 * slot's coalescing rate; null omits the row, while 0 still shows, meaning "running, dropping none". */
export function nodeStatsRows(
	stats: NodeStats | null | undefined,
	drops?: number | null
): { label: string; value: string }[] {
	if (!stats) return [];
	const rows = [{ label: 'Update rate', value: `${fmtRate(stats.updates_per_second)} upd/s` }];
	if (drops != null) rows.push({ label: 'Dropped', value: `${drops.toFixed(1)}/s` });
	return rows;
}
