import type { Page } from '@playwright/test';

/** Add a node via the command façade; returns its stable uid. */
export function addNode(
	page: Page,
	type: string,
	category = 'inputs',
	pos: [number, number] = [0, 0]
): Promise<string> {
	return page.evaluate(
		([t, c, p]) => (window as any).goofi.commands.addNode(t, c, p),
		[type, category, pos] as const
	);
}

/** The current graph nodes (identity view). */
export function nodes(page: Page): Promise<Array<{ uid: string; type: string; name: string }>> {
	return page.evaluate(() => (window as any).goofi.query.graph().nodes);
}

/** Wait until a node with `uid` is present in the reactive graph. `addNode` returns the uid as
 * soon as the command resolves, but the node only appears in the store after the doc round-trip
 * (mirror → broadcast → apply → reconcile), so callers wait on this before reading/editing it. */
export async function waitForNode(page: Page, uid: string): Promise<void> {
	await page.waitForFunction(
		(u) => ((window as any).goofi.query.graph().nodes as Array<{ uid: string }>).some((n) => n.uid === u),
		uid,
		{ timeout: 10_000 }
	);
}

/** A node's params object (`{group: {name: {value, …}}}`). */
export function nodeParams(page: Page, uid: string): Promise<any> {
	return page.evaluate((u) => (window as any).goofi.query.nodeParams(u), uid);
}

/** Wait until a node with `uid` is ABSENT from the reactive graph (an undo-of-add / delete
 * round-trip has landed). */
export async function waitForNoNode(page: Page, uid: string): Promise<void> {
	await page.waitForFunction(
		(u) => !((window as any).goofi.query.graph().nodes as Array<{ uid: string }>).some((n) => n.uid === u),
		uid,
		{ timeout: 10_000 }
	);
}

/** Edit a committed param value (a doc leaf-write). */
export function updateParam(
	page: Page,
	uid: string,
	group: string,
	name: string,
	value: unknown
): Promise<void> {
	return page.evaluate(
		([u, g, n, v]) => (window as any).goofi.commands.updateParam(u, g, n, v),
		[uid, group, name, value] as const
	);
}

/** Add a node that is permanently in error, and wait for it to reach the graph store.
 *
 * The mechanism, stated once here so no call site has to restate it: `LempelZiv` declares its
 * `data` input **required** (`process()` reads `data.data` unconditionally), and it is added with
 * nothing connected, so that slot's last-store is empty. `execute_node` refuses the tick and
 * records the error BEFORE `process` is entered — the same per-node `error` field that drives the
 * floating chip, the console rows and the inspector traceback. It is permanent (nothing here ever
 * connects the slot) and environment-independent (no missing dependency; numpy is in both venvs).
 *
 * `autotrigger` is what makes it tick at all: the required check fires on a TICK, not on the
 * configuration, and an unwired single-slot Python node has a trigger input with autotrigger off,
 * so left alone it never runs — a disconnected node floating in space, silent by design.
 *
 * `common.max_frequency` is capped FIRST, before autotrigger, because it defaults to `0.0` =
 * uncapped: an autotriggered unwired node otherwise free-runs at ~10 kHz (~12 % of a core) for the
 * rest of the spec. Those ticks never reach Python — the required check returns before `process` is
 * entered — but each one re-runs that check and reallocates the node's `last_error`, on the tick
 * thread of a backend every spec on this Playwright worker shares. 2 Hz is ample — the error is
 * permanent, and the console reports error TRANSITIONS rather than ticks.
 *
 * The caller removes the node when done: the backend graph is shared by every spec on the worker.
 */
export async function addErroringNode(page: Page): Promise<string> {
	const uid = await addNode(page, 'LempelZiv', 'python');
	// `waitForNode` first, and not merely to settle: the client's `updateParam` guards on the param
	// EXISTING in its replica, so an edit sent before the doc round-trip is refused outright.
	await waitForNode(page, uid);
	await updateParam(page, uid, 'common', 'max_frequency', 2);
	await updateParam(page, uid, 'common', 'autotrigger', true);
	return uid;
}

/** Undo the last action. */
export function undo(page: Page): Promise<void> {
	return page.evaluate(() => (window as any).goofi.commands.undo());
}
/** Redo the last undone action. */
export function redo(page: Page): Promise<void> {
	return page.evaluate(() => (window as any).goofi.commands.redo());
}
/** Whether an undo is available. */
export function canUndo(page: Page): Promise<boolean> {
	return page.evaluate(() => (window as any).goofi.query.canUndo());
}
/** Add a user global; returns whether it landed. */
export function addGlobal(
	page: Page,
	name: string,
	value: number | string | boolean,
	type: 'float' | 'int' | 'bool' | 'string'
): Promise<void> {
	// A command op — resolves void on success, rejects on a server refusal (invalid/collision).
	return page.evaluate(
		([n, v, t]) => (window as any).goofi.commands.addGlobal(n, v, t),
		[name, value, type] as const
	);
}
/** All patch globals (system + user). */
export function globals(
	page: Page
): Promise<Array<{ name: string; value: unknown; type: string; system: boolean }>> {
	return page.evaluate(() => (window as any).goofi.query.globals());
}
