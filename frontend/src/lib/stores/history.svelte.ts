/** Unified undo/redo — ONE client stack of markers that delegate to the manager's own history. A
 * node's INLINE viewer state is the one exception: no command owns it, so it replays a snapshot. */
import type { Control, ControlEvent } from '$lib/api/control';
import { getControl } from '$lib/api/control';
import type { ViewerKind } from '$lib/viewers/kind';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import { graph } from './graph.svelte';
import { viewExecutors } from '$lib/viewers/viewExecutors';
import { restoreNavContext } from '$lib/stores/navContext';
import { pulseRestored } from './undoFlash';
import { notify } from './notify.svelte';

export type ActionDomain = 'graph' | 'view';

/** Where an action was performed, restored before its inverse/forward runs. */
export interface NavContext {
	activeWorkspaceId: string;
	activePanelId: string | null;
	/** per editor panel id: the sub-patch instance-id stack (root → deepest). */
	enteredPath: Record<string, string[]>;
	/** per panel id: selected node + edge ids at record time. */
	selection: Record<string, { nodes: string[]; edges: string[] }>;
}

export interface BaseAction {
	kind: string;
	/** Human label for the undo/redo button + tooltip. */
	label: string;
	domain: ActionDomain;
	context: NavContext;
}

/** A manager command. The manager owns the exact inverse, so this carries no payload. */
export type GraphAction = BaseAction & {
	kind: 'graph_cmd';
	domain: 'graph';
};

/** Several actions grouped into one undo step, replayed in order and reversed in reverse. */
export type CompoundAction = BaseAction & {
	domain: 'graph';
	kind: 'compound';
	payload: { children: Action[] };
};

/** A viewer's kind + cog-menu settings snapshot. */
export interface ViewSnapshot {
	kind?: ViewerKind;
	settings: SettingsMap;
}

/** Only a node's INLINE body viewer is client-local; a docked Viewer panel rides its panel state. */
export type ViewTarget = { kind: 'inline'; node: string; slot: string };

export type ViewAction = BaseAction & {
	domain: 'view';
	kind: 'set_view';
	payload: { target: ViewTarget; before: ViewSnapshot; after: ViewSnapshot };
};

export type Action = GraphAction | CompoundAction | ViewAction;

export type GraphStoreT = ReturnType<typeof graph>;

/** Injected dependencies so executors are unit-testable against fakes. */
export interface ExecutorDeps {
	control: Control;
	graph: GraphStoreT;
}

export interface Executor<A extends Action = Action> {
	/** Re-apply (redo). May mutate the action to record fresh ids for the next inverse. */
	forward(action: A, deps: ExecutorDeps): Promise<void>;
	/** Reverse (undo). */
	inverse(action: A, deps: ExecutorDeps): Promise<void>;
}


/** Replays a CompoundAction's children — forward in order, inverse in reverse. */
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

/** The one MANAGER executor: undo/redo delegate to its per-session command history. */
const graphExecutor: Executor = {
	async forward(_action, deps) {
		await deps.control.call('redo', {});
	},
	async inverse(_action, deps) {
		await deps.control.call('undo', {});
	}
};

/** The merged dispatch registry. */
export const executors: Record<string, Executor> = {
	graph_cmd: graphExecutor,
	...viewExecutors,
	compound: compoundExecutor
};

/** The live dependency bundle. Lazy singletons, so the history ↔ graph import cycle is safe. */
function liveDeps(): ExecutorDeps {
	return { control: getControl(), graph: graph() };
}

export class HistoryStore {
	canUndo = $state(false);
	canRedo = $state(false);
	undoLabel = $state<string | null>(null);
	redoLabel = $state<string | null>(null);

	private undoStack: Action[] = [];
	private redoStack: Action[] = [];
	private suspendDepth = 0;
	/** Re-entrancy guard: two awaits sit between reading the top action and popping it, so a held
	 * Ctrl+Z would otherwise double-replay it. */
	private replaying = false;
	/** While a transaction is open, records collect here instead of pushing. */
	private txBuffer: Action[] | null = null;
	/** How undo/redo resolve the stores+control to replay against. */
	private depsProvider: () => ExecutorDeps = liveDeps;

	/** Test seam: replay against an injected store/control. */
	configureDeps(provider: () => ExecutorDeps): void {
		this.depsProvider = provider;
	}

	/** Record a completed action, clearing the redo stack. No-op while suspended. */
	record(action: Action): void {
		if (this.suspendDepth > 0) return;
		if (this.txBuffer) {
			this.txBuffer.push(action);
			return;
		}
		this.undoStack.push(action);
		this.redoStack = [];
		this._recompute();
	}

	/** Move the top action of `from` to `to`, replaying it through `direction`. Atomic: on failure
	 * the action stays on `from` and the failure is raised as a toast. */
	private async _replay(
		from: Action[],
		to: Action[],
		direction: 'inverse' | 'forward',
		verb: string
	): Promise<void> {
		if (this.replaying || from.length === 0) return;
		const action = from[from.length - 1];
		const exec = executors[action.kind];
		if (!exec) return;
		this.replaying = true;
		try {
			await restoreNavContext(action.context);
			await this.suspend(() => exec[direction](action, this.depsProvider()));
		} catch (e) {
			notify().failure(verb, e);
			return; // leave the stacks untouched (atomic-or-nothing)
		} finally {
			this.replaying = false;
		}
		from.pop();
		to.push(action);
		this._recompute();
		pulseRestored(action.context, this.depsProvider());
	}

	async undo(): Promise<void> {
		return this._replay(this.undoStack, this.redoStack, 'inverse', 'Undo');
	}

	async redo(): Promise<void> {
		return this._replay(this.redoStack, this.undoStack, 'forward', 'Redo');
	}

	/** Run `fn` with recording disabled. Reentrant, and async-aware: a promise keeps the guard up. */
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

	/** Group every action recorded inside `fn` into ONE undo step; nested transactions fold in. */
	async transaction<T>(label: string, fn: () => Promise<T>): Promise<T> {
		if (this.txBuffer || this.suspendDepth > 0) return fn(); // nested / suspended → passthrough
		this.txBuffer = [];
		let result: T;
		try {
			result = await fn();
		} catch (e) {
			// A thrown transaction did not complete, so its buffered children are discarded whole.
			this.txBuffer = null;
			throw e;
		}
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
		return result;
	}

	get isSuspended(): boolean {
		return this.suspendDepth > 0;
	}

	/** Number of undoable steps currently on the stack. */
	get length(): number {
		return this.undoStack.length;
	}

	/** Hard reset on a new backend session; the deps provider is config, not state, so it stays. */
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
