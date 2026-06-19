/**
 * Layout-domain executors. The layout tree is frontend-authoritative and
 * in-memory, so undo/redo just restore a `WorkspaceState` snapshot — no RPCs.
 * Every tracked layout action carries `{ before, after }` and shares this one
 * snapshot executor; the kind only distinguishes them for labels/history.
 */
import type { Executor, LayoutAction, LayoutActionKind } from '$lib/stores/history.svelte';

const snapshotExecutor: Executor<LayoutAction> = {
	async forward(action, deps) {
		deps.workspace.restore(action.payload.after);
	},
	async inverse(action, deps) {
		deps.workspace.restore(action.payload.before);
	}
};

const KINDS: LayoutActionKind[] = [
	'split_panel',
	'close_panel',
	'resize_split',
	'move_panel',
	'set_panel_type',
	'link_node_to_panel',
	'add_tab',
	'close_tab',
	'duplicate_tab',
	'rename_tab',
	'reorder_tab'
];

export const layoutExecutors: Record<string, Executor<LayoutAction>> = Object.fromEntries(
	KINDS.map((k) => [k, snapshotExecutor])
);
