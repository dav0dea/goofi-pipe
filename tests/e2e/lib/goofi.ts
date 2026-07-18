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
