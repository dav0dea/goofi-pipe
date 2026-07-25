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
/** Edit an existing global's value (command op — resolves void, rejects on refusal). */
export function setGlobalValue(
	page: Page,
	name: string,
	value: number | string | boolean
): Promise<void> {
	return page.evaluate(
		([n, v]) => (window as any).goofi.commands.setGlobalValue(n, v),
		[name, value] as const
	);
}
/** All patch globals (system + user). */
export function globals(
	page: Page
): Promise<Array<{ name: string; value: unknown; type: string; system: boolean }>> {
	return page.evaluate(() => (window as any).goofi.query.globals());
}
