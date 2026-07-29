import { test, expect, type Page } from '@playwright/test';
import { closeAddedTab, waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * Two clients on one patch — a desktop and the phone next to it.
 *
 * The panel arrangement is not a CRDT doc root: it is opaque view state, deliberately not a
 * command, so it travels on the manager's `layout` event or not at all. What that event carries is
 * split along the same line the dirty taxonomy draws (R spec §4 / D-R3): an *authored* change —
 * splitting a panel, adding a tab — is what the patch IS, and reaches everyone; *navigation* is
 * where one client is LOOKING, and must move nobody else.
 *
 * The second half is the receiving side of the same rule. A peer's blob carries its viewpoint as
 * well as its structure, so applying it is a merge, never a hydrate — a phone three sub-patches
 * deep stays there while the desktop adds a panel around it.
 *
 * Two browser CONTEXTS, not two pages: the session id that lets a client skip its own echo lives
 * in `sessionStorage`, which a second page in the same context would share. One backend serves
 * both (`fullyParallel: false`, `workers: 1`), so every test here hands the workspace back.
 */

/** Comfortably past AppShell's 400ms layout debounce plus the RPC round trip. */
const PAST_DEBOUNCE = 1200;

const tabs = (page: Page) => page.getByTestId('workspace-tabs').locator('.tab');

/** Count this page's OWN `set_layout` pushes. Must be attached before `goto` — `websocket` only
 * fires for sockets opened after the listener. */
function countLayoutPushes(page: Page): () => number {
	let n = 0;
	page.on('websocket', (ws) => {
		if (!ws.url().endsWith('/control')) return;
		ws.on('framesent', (f) => {
			if (typeof f.payload === 'string' && f.payload.includes('"set_layout"')) n += 1;
		});
	});
	return () => n;
}

/** Two nodes grouped into a sub-patch, on the page that will own the teardown. */
async function makeSubPatch(page: Page): Promise<{ osc: string; buf: string; inst: string }> {
	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);
	const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
	await waitForNode(page, buf);
	const inst: string = await page.evaluate(
		([o, b]) => (window as any).goofi.commands.groupNodes([o, b], [120, 120]),
		[osc, buf] as const
	);
	await expect(page.getByTestId('subpatch-node')).toBeVisible();
	return { osc, buf, inst };
}

test('an authored panel change reaches the other client, and is not echoed back', async ({
	browser
}) => {
	const ctxA = await browser.newContext();
	const ctxB = await browser.newContext();
	const a = await ctxA.newPage();
	const b = await ctxB.newPage();
	const pushesB = countLayoutPushes(b);
	try {
		await a.goto('/');
		await waitForApp(a);
		await b.goto('/');
		await waitForApp(b);
		// Past B's own boot push, so what is counted below is only what the peer caused.
		await b.waitForTimeout(PAST_DEBOUNCE);
		const before = pushesB();

		await a.evaluate(() => (window as any).goofi.commands.addTab());

		await expect(tabs(b), 'the tab A added arrived').toHaveCount(2);
		await expect(
			tabs(b).first(),
			'…and B is still looking at its own tab, not the one A brought to the front'
		).toHaveAttribute('aria-selected', 'true');
		await expect(tabs(a).nth(1), 'while A is on the tab it just made').toHaveAttribute(
			'aria-selected',
			'true'
		);

		// No echo, no storm: applying a peer's arrangement must schedule no push of its own. Waited
		// well past the debounce, so a push that was merely slow would still be counted.
		await b.waitForTimeout(PAST_DEBOUNCE);
		expect(pushesB() - before, 'B must not push a peer’s arrangement back').toBe(0);
	} finally {
		if ((await tabs(a).count()) > 1) await closeAddedTab(a);
		await ctxA.close();
		await ctxB.close();
	}
});

test('navigating does not move the other client, and a peer’s edit does not move us', async ({
	browser
}) => {
	const ctxA = await browser.newContext();
	const ctxB = await browser.newContext();
	const a = await ctxA.newPage();
	const b = await ctxB.newPage();
	let made: { osc: string; buf: string; inst: string } | null = null;
	try {
		await a.goto('/');
		await waitForApp(a);
		await b.goto('/');
		await waitForApp(b);

		made = await makeSubPatch(a);
		await expect(b.getByTestId('subpatch-node'), 'the sub-patch reached B through the doc').toBeVisible();

		// B descends into it. That is navigation — where B is looking — so A must stay put.
		await b.getByTestId('subpatch-node').dblclick();
		await expect(b.getByTestId('subpatch-breadcrumb'), 'B is inside the sub-patch').toBeVisible();
		await b.waitForTimeout(PAST_DEBOUNCE);
		await expect(
			a.getByTestId('subpatch-breadcrumb'),
			'A must not be dragged into a sub-patch B entered'
		).toBeHidden();

		// Now A AUTHORS. The tab travels; B's own position inside the sub-patch survives it.
		await a.evaluate(() => (window as any).goofi.commands.addTab());
		await expect(tabs(b), 'A’s new tab arrived').toHaveCount(2);
		await expect(
			b.getByTestId('subpatch-breadcrumb'),
			'…and B is still where it was looking'
		).toBeVisible();
		await expect(tabs(b).first(), 'still on its own tab, too').toHaveAttribute(
			'aria-selected',
			'true'
		);
	} finally {
		// B climbs out first — its editor must not be sitting inside an instance about to dissolve —
		// then A hands the tab back LAST, so the arrangement the manager keeps is the pristine one.
		const crumb = b.getByTestId('subpatch-breadcrumb');
		if (await crumb.isVisible())
			await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
		await b.waitForTimeout(PAST_DEBOUNCE);
		if ((await tabs(a).count()) > 1) await closeAddedTab(a);
		if (made) {
			await a.evaluate((i) => (window as any).goofi.commands.expandInstance(i), made.inst);
			await a.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [
				made.osc,
				made.buf
			]);
			await waitForNoNode(a, made.osc);
			await waitForNoNode(a, made.buf);
		}
		await ctxA.close();
		await ctxB.close();
	}
});
