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
import { getControl } from '$lib/api/control';
import type { WorkspaceState } from '$lib/workspace/model';
import type { ViewerKind } from '$lib/viewers/kind';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import { graph } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { graphExecutors } from './graphExecutors';
import { layoutExecutors } from '$lib/workspace/layoutExecutors';
import { viewExecutors } from '$lib/viewers/viewExecutors';
import { restoreNavContext } from '$lib/workspace/navContext';
import { pulseRestored } from './undoFlash';

export type ActionDomain = 'graph' | 'layout' | 'view';

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
	/** When set, a freshly-recorded action whose key matches the current top of
	 * the undo stack MERGES into it instead of pushing a new entry — so a
	 * continuous gesture (a splitter drag firing resize() per mousemove)
	 * collapses to a single undo step. Distinct from time-window coalescing:
	 * any intervening action with a different key breaks the run. */
	coalesceKey?: string;
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

/** Several primitive actions grouped into one undo step (e.g. a slot-click that
 * creates a node AND wires it). Children are replayed in order on redo and in
 * reverse on undo, each through its own executor. */
export type CompoundAction = BaseAction & {
	domain: 'graph';
	kind: 'compound';
	payload: { children: Action[] };
};

// --- view domain: a data viewer's kind/settings (frontend-only state) --------
/** A viewer's kind + cog-menu settings snapshot. */
export interface ViewSnapshot {
	kind?: ViewerKind;
	settings: SettingsMap;
}

/** Where a viewer's view-state lives: on a node's inline body viewer, or in a
 * docked Viewer panel's layout state. */
export type ViewTarget =
	| { kind: 'inline'; node: string; slot: string }
	| { kind: 'panel'; panelId: string };

export type ViewAction = BaseAction & {
	domain: 'view';
	kind: 'set_view';
	payload: { target: ViewTarget; before: ViewSnapshot; after: ViewSnapshot };
};

export type Action = GraphAction | LayoutAction | CompoundAction | ViewAction;

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

// A `ControlEvent` re-export keeps test imports terse.
export type { ControlEvent, InstanceInfo };

/** Replays a CompoundAction's children through their own executors — forward in
 * order, inverse in reverse. */
const compoundExecutor: Executor = {
	async forward(action, deps) {
		const a = action as CompoundAction;
		for (const child of a.payload.children) await executors[child.kind]?.forward(child, deps);
	},
	async inverse(action, deps) {
		const a = action as CompoundAction;
		for (const child of [...a.payload.children].reverse()) await executors[child.kind]?.inverse(child, deps);
	}
};

/** The merged dispatch registry. Graph executors land here in Phase 2/3; layout
 * executors are spread in by Phase 4 (widening cast — each narrows internally);
 * the compound executor groups several primitives into one step. */
export const executors: Record<string, Executor> = {
	...graphExecutors,
	...(layoutExecutors as Record<string, Executor>),
	...viewExecutors,
	compound: compoundExecutor
};

/** Build the live dependency bundle for replaying actions. Lazy singletons, so
 * this is safe despite the history ↔ graph import cycle (never called at
 * module-eval time). */
function liveDeps(): ExecutorDeps {
	return { control: getControl(), graph: graph(), workspace: workspace() };
}

// --- the store ---------------------------------------------------------------
export class HistoryStore {
	canUndo = $state(false);
	canRedo = $state(false);
	undoLabel = $state<string | null>(null);
	redoLabel = $state<string | null>(null);
	/** Set when an undo/redo replay rejects; surfaced as a toast (Phase 6). */
	lastError = $state<string | null>(null);

	private undoStack: Action[] = [];
	private redoStack: Action[] = [];
	private suspendDepth = 0;
	/** Re-entrancy guard for undo()/redo(): two awaits sit between reading the
	 * top action and pop(), so a held Ctrl+Z could double-replay and double-pop
	 * the same action. While a replay is in flight, further undo/redo are
	 * dropped (report B13). */
	private replaying = false;
	/** While a transaction is open, records collect here instead of pushing —
	 * see `transaction`. Null when no transaction is active. */
	private txBuffer: Action[] | null = null;
	private txLabel = '';
	/** How undo/redo resolve the stores+control to replay against. Defaults to
	 * the live singletons; tests override it to inject a FakeControl-backed
	 * store. */
	private depsProvider: () => ExecutorDeps = liveDeps;

	/** Test seam: replay against an injected store/control instead of the live
	 * singletons. */
	configureDeps(provider: () => ExecutorDeps): void {
		this.depsProvider = provider;
	}

	/** Record a completed action. No-op while suspended. Clears the redo stack
	 * (a new edit invalidates any redo future). */
	record(action: Action): void {
		if (this.suspendDepth > 0) return;
		// Inside a transaction: collect rather than push — flushed as one entry.
		if (this.txBuffer) {
			this.txBuffer.push(action);
			return;
		}
		const top = this.undoStack[this.undoStack.length - 1];
		// Gesture coalescing: merge into the matching top entry (keep its `before`,
		// adopt the new `after`) rather than pushing a fresh step.
		if (
			action.coalesceKey &&
			top &&
			top.coalesceKey === action.coalesceKey &&
			top.domain === 'layout' &&
			action.domain === 'layout'
		) {
			top.payload.after = action.payload.after;
			this.redoStack = [];
			this._recompute();
			return;
		}
		this.undoStack.push(action);
		this.redoStack = [];
		this._recompute();
	}

	/** Replay the top action's inverse: restore its nav context, run the inverse
	 * (recording suspended), then move it to the redo stack. Atomic — on RPC
	 * failure the action stays on the undo stack and `lastError` is set. */
	async undo(): Promise<void> {
		if (this.replaying || !this.canUndo) return;
		const action = this.undoStack[this.undoStack.length - 1];
		const exec = executors[action.kind];
		if (!exec) return;
		this.replaying = true;
		try {
			await restoreNavContext(action.context);
			await this.suspend(() => exec.inverse(action, this.depsProvider()));
		} catch (e) {
			this.lastError = `Undo failed: ${(e as Error).message ?? e}`;
			return; // leave the stacks untouched (atomic-or-nothing)
		} finally {
			this.replaying = false;
		}
		this.undoStack.pop();
		this.redoStack.push(action);
		this._recompute();
		pulseRestored(action.context, this.depsProvider());
	}

	async redo(): Promise<void> {
		if (this.replaying || !this.canRedo) return;
		const action = this.redoStack[this.redoStack.length - 1];
		const exec = executors[action.kind];
		if (!exec) return;
		this.replaying = true;
		try {
			await restoreNavContext(action.context);
			await this.suspend(() => exec.forward(action, this.depsProvider()));
		} catch (e) {
			this.lastError = `Redo failed: ${(e as Error).message ?? e}`;
			return;
		} finally {
			this.replaying = false;
		}
		this.redoStack.pop();
		this.undoStack.push(action);
		this._recompute();
		pulseRestored(action.context, this.depsProvider());
	}

	/** Run `fn` with recording disabled, then resume. Reentrant, and async-aware:
	 * if `fn` returns a promise the guard stays up until it settles (so an
	 * awaited replay or batch doesn't re-record mid-flight). */
	suspend<T>(fn: () => T): T {
		this.suspendDepth += 1;
		let result: T;
		try {
			result = fn();
		} catch (e) {
			this.suspendDepth -= 1;
			throw e;
		}
		if (result && typeof (result as { then?: unknown }).then === 'function') {
			return (result as unknown as Promise<unknown>).finally(() => {
				this.suspendDepth -= 1;
			}) as unknown as T;
		}
		this.suspendDepth -= 1;
		return result;
	}

	/** Group every action recorded inside `fn` into ONE undo step. A single
	 * recorded child is kept as-is (no wrapper); two or more become a `compound`.
	 * Async-aware; nested transactions fold into the outer one. */
	async transaction<T>(label: string, fn: () => Promise<T>): Promise<T> {
		if (this.txBuffer || this.suspendDepth > 0) return fn(); // nested / suspended → passthrough
		this.txBuffer = [];
		this.txLabel = label;
		let result: T;
		try {
			result = await fn();
		} finally {
			const children = this.txBuffer ?? [];
			this.txBuffer = null;
			if (children.length === 1) {
				this.record(children[0]);
			} else if (children.length > 1) {
				this.record({
					kind: 'compound',
					domain: 'graph',
					label,
					context: children[0].context,
					payload: { children }
				});
			}
		}
		return result;
	}

	get isSuspended(): boolean {
		return this.suspendDepth > 0;
	}

	/** Number of undoable steps currently on the stack. */
	get length(): number {
		return this.undoStack.length;
	}

	/** Hard reset — only on a new backend session (see graph store). Drops the
	 * stacks and any pending error; the deps provider is config, not state, so it
	 * is left intact. */
	reset(): void {
		this.undoStack = [];
		this.redoStack = [];
		this.suspendDepth = 0;
		this.lastError = null;
		this._recompute();
	}

	/** Dismiss the last undo/redo error (e.g. after its toast is shown). */
	clearError(): void {
		this.lastError = null;
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
