/**
 * Unified undo/redo history — one stack spanning the graph domain (replayed as
 * inverse/forward RPCs; the backend stays authoritative and never learns "undo"
 * exists) and the layout domain (restored as `WorkspaceState` snapshots).
 *
 * Recording happens at the store-method layer (graph + workspace) behind the
 * `suspend` guard so replays don't re-record. Each action carries a
 * `NavContext` restored on undo/redo so the change is highlighted where it
 * happened. See `docs/superpowers/specs/2026-06-19-undo-redo-redesign-design.md`.
 */
import type {
	Control,
	ControlEvent,
	InstanceInfo,
	LinkInfo,
	NodeInstanceInfo,
	SubPatchPort
} from '$lib/api/control';
import type { WorkspaceState } from '$lib/workspace/model';
import type { graph } from './graph.svelte';
import type { workspace } from '$lib/workspace/workspace.svelte';

export type ActionDomain = 'graph' | 'layout';

/** Where an action was performed — restored before its inverse/forward runs so
 * the undone/redone change is highlighted in the right tab/panel/sub-patch. */
export interface NavContext {
	activeWorkspaceId: string;
	activePanelId: string | null;
	/** per editor panel id: the sub-patch instance-id stack (root → deepest). */
	enteredPath: Record<string, string[]>;
	/** per panel id: selected node + edge ids at record time. */
	selection: Record<string, { nodes: string[]; edges: string[] }>;
}

export interface ExprState {
	expression: string | null;
	enabled: boolean;
	triggers_process: boolean;
	autoeval: boolean;
}

export interface BaseAction {
	kind: string;
	/** Human label for the undo/redo button + tooltip. */
	label: string;
	domain: ActionDomain;
	context: NavContext;
}

// --- graph domain: replayed as RPCs ------------------------------------------
export type GraphAction =
	| (BaseAction & {
			kind: 'add_node';
			domain: 'graph';
			payload: {
				type: string;
				category: string;
				pos: [number, number];
				instId?: string;
				assignedName?: string;
			};
	  })
	| (BaseAction & {
			kind: 'remove_node';
			domain: 'graph';
			payload: {
				name: string;
				node: NodeInstanceInfo;
				links: LinkInfo[];
				membership: { instance: string; local_name: string } | null;
				boundPanels: Array<{ panelId: string; state: unknown }>;
			};
	  })
	| (BaseAction & { kind: 'add_link'; domain: 'graph'; payload: { link: LinkInfo; displaced: LinkInfo | null } })
	| (BaseAction & { kind: 'remove_link'; domain: 'graph'; payload: { link: LinkInfo } })
	| (BaseAction & {
			kind: 'update_param';
			domain: 'graph';
			payload: { node: string; group: string; name: string; oldValue: unknown; newValue: unknown };
	  })
	| (BaseAction & {
			kind: 'set_expression';
			domain: 'graph';
			payload: { node: string; group: string; name: string; oldExpr: ExprState; newExpr: ExprState };
	  })
	| (BaseAction & {
			kind: 'set_node_pos';
			domain: 'graph';
			payload: { name: string; oldPos: [number, number]; newPos: [number, number] };
	  })
	| (BaseAction & {
			kind: 'group_nodes';
			domain: 'graph';
			payload: { members: string[]; instId: string; pos?: [number, number] };
	  })
	| (BaseAction & {
			kind: 'expand_instance';
			domain: 'graph';
			payload: { instId: string; restoredMembers: string[]; interface: Record<string, SubPatchPort> };
	  })
	| (BaseAction & {
			kind: 'duplicate_shared';
			domain: 'graph';
			payload: { instId: string; newInstId: string; wasUnique: boolean; pos?: [number, number] };
	  })
	| (BaseAction & { kind: 'make_unique'; domain: 'graph'; payload: { instId: string; defIdBefore: string | null } })
	| (BaseAction & {
			kind: 'add_boundary';
			domain: 'graph';
			payload: { instId: string; bndId: string; dir: 'in' | 'out'; dtype: string; pos: [number, number] };
	  })
	| (BaseAction & {
			kind: 'wire_boundary';
			domain: 'graph';
			payload: {
				instId: string;
				bndId: string;
				oldInner: { node: string | null; slot: string | null };
				newInner: { node: string | null; slot: string | null };
			};
	  })
	| (BaseAction & { kind: 'remove_boundary'; domain: 'graph'; payload: { instId: string; bndId: string; port: SubPatchPort } })
	| (BaseAction & {
			kind: 'set_boundary_pos';
			domain: 'graph';
			payload: { instId: string; bndId: string; oldPos: [number, number]; newPos: [number, number] };
	  })
	| (BaseAction & {
			kind: 'load_patch';
			domain: 'graph';
			payload: {
				beforeYaml: string;
				afterYaml: string;
				beforeLayout: WorkspaceState | null;
				afterLayout: WorkspaceState | null;
				instanceId: string;
			};
	  });

// --- layout domain: replayed as WorkspaceState snapshot restores -------------
export type LayoutActionKind =
	| 'split_panel'
	| 'close_panel'
	| 'resize_split'
	| 'move_panel'
	| 'set_panel_type'
	| 'link_node_to_panel'
	| 'add_tab'
	| 'close_tab'
	| 'duplicate_tab'
	| 'rename_tab'
	| 'reorder_tab';

export type LayoutAction = BaseAction & {
	domain: 'layout';
	kind: LayoutActionKind;
	payload: { before: WorkspaceState; after: WorkspaceState };
};

export type Action = GraphAction | LayoutAction;

// --- executors ---------------------------------------------------------------
export type GraphStoreT = ReturnType<typeof graph>;
export type WorkspaceStoreT = ReturnType<typeof workspace>;

/** Injected dependencies so executors are unit-testable against fakes. */
export interface ExecutorDeps {
	control: Control;
	graph: GraphStoreT;
	workspace: WorkspaceStoreT;
}

export interface Executor<A extends Action = Action> {
	/** Re-apply (redo). May mutate the action to record fresh ids (e.g. a
	 *  re-grouped instId) so the next inverse targets the right thing. */
	forward(action: A, deps: ExecutorDeps): Promise<void>;
	/** Reverse (undo). */
	inverse(action: A, deps: ExecutorDeps): Promise<void>;
}

/** The merged dispatch registry. Filled by `graphExecutors` (Phase 2/3) and
 * `layoutExecutors` (Phase 4) — see the merge in those modules' wiring. */
export const executors: Record<string, Executor> = {};

// A `ControlEvent` re-export keeps test imports terse.
export type { ControlEvent, InstanceInfo };

// --- the store ---------------------------------------------------------------
export class HistoryStore {
	canUndo = $state(false);
	canRedo = $state(false);
	undoLabel = $state<string | null>(null);
	redoLabel = $state<string | null>(null);

	private undoStack: Action[] = [];
	private redoStack: Action[] = [];
	private suspendDepth = 0;

	/** Record a completed action. No-op while suspended. Clears the redo stack
	 * (a new edit invalidates any redo future). */
	record(action: Action): void {
		if (this.suspendDepth > 0) return;
		this.undoStack.push(action);
		this.redoStack = [];
		this._recompute();
	}

	/** Phase 1: stack mechanics only. Phase 2 wires NavContext restore +
	 * `executors[kind].inverse`. */
	async undo(): Promise<void> {
		if (!this.canUndo) return;
		const action = this.undoStack.pop()!;
		this.redoStack.push(action);
		this._recompute();
	}

	async redo(): Promise<void> {
		if (!this.canRedo) return;
		const action = this.redoStack.pop()!;
		this.undoStack.push(action);
		this._recompute();
	}

	/** Run `fn` with recording disabled (executors replay through recording
	 * store methods without pushing new actions). Reentrant. */
	suspend<T>(fn: () => T): T {
		this.suspendDepth += 1;
		try {
			return fn();
		} finally {
			this.suspendDepth -= 1;
		}
	}

	get isSuspended(): boolean {
		return this.suspendDepth > 0;
	}

	/** Hard reset — only on a new backend session (see graph store). */
	reset(): void {
		this.undoStack = [];
		this.redoStack = [];
		this.suspendDepth = 0;
		this._recompute();
	}

	private _recompute(): void {
		this.canUndo = this.undoStack.length > 0;
		this.canRedo = this.redoStack.length > 0;
		this.undoLabel = this.canUndo ? this.undoStack[this.undoStack.length - 1].label : null;
		this.redoLabel = this.canRedo ? this.redoStack[this.redoStack.length - 1].label : null;
	}
}

let _history: HistoryStore | null = null;
export function history(): HistoryStore {
	if (!_history) _history = new HistoryStore();
	return _history;
}
