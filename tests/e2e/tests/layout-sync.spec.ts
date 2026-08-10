import { test, expect, type Page } from '@playwright/test';
import { closeAddedTab, closeSplit, splitRight, waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * Two clients on one patch — a desktop and the phone next to it.
 *
 * The panel arrangement is the FIFTH CRDT doc root. Every gesture is a layout command the manager
 * applies, mirrors and broadcasts as a delta, so a split on one screen appears on the other through
 * the very machinery a node add already uses. Before the cutover this did not happen at all: the
 * arrangement was the client's, pushed back as an opaque blob, and a peer merged what it could.
 *
 * The other half is what does NOT travel. Where a client is looking — its front tab, its focused
 * panel, how deep into a sub-patch each editor has gone — is viewpoint, never a doc root, so a
 * phone three levels in stays there while the desktop rearranges around it, and neither client
 * writes the arrangement in answer to the other.
 *
 * Two browser CONTEXTS, not two pages: the session id that scopes undo lives in `sessionStorage`,
 * which a second page in the same context would share. Both contexts reach the ONE backend this
 * worker owns — the two clients converging on one manager is the whole point — so every test here
 * hands the workspace back.
 */

/** Comfortably past a layout command and its delta coming back. */
const PAST_DEBOUNCE = 1200;

const tabs = (page: Page) => page.getByTestId('workspace-tabs').locator('.ui-tab');
const panels = (page: Page) => page.locator('.panel');

/** Count this page's OWN writes to the arrangement. Must be attached before `goto` — `websocket`
 * only fires for sockets opened after the listener. */
function countLayoutWrites(page: Page): () => number {
	let n = 0;
	page.on('websocket', (ws) => {
		if (!ws.url().endsWith('/control')) return;
		ws.on('framesent', (f) => {
			if (typeof f.payload !== 'string') return;
			if (/"(page|session)_[a-z_]+"/.test(f.payload)) n += 1;
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

test('a layout change in one tab converges in another', async ({ browser }) => {
	const ctxA = await browser.newContext();
	const ctxB = await browser.newContext();
	const a = await ctxA.newPage();
	const b = await ctxB.newPage();
	const writesB = countLayoutWrites(b);
	try {
		await a.goto('/');
		await waitForApp(a);
		await b.goto('/');
		await waitForApp(b);
		await expect(panels(b)).toHaveCount(1);
		const before = writesB();

		// A SPLIT, through the real header menu — the arrangement's own structure, not a tab. This is
		// the change that used to reach nobody.
		await splitRight(a);
		await expect(panels(b), 'the split A made arrived through the doc').toHaveCount(2);

		// A closes it again, and the close converges too — a delta, not a wholesale blob.
		await closeSplit(a);
		await expect(panels(b), '…and so does the close').toHaveCount(1);

		// A tab, too — and B stays on its OWN tab, because which one is in front is viewpoint.
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

		// No echo, no storm: converging on a peer's arrangement must write nothing back. Waited well
		// past the round trip, so a write that was merely slow would still be counted.
		await b.waitForTimeout(PAST_DEBOUNCE);
		expect(writesB() - before, 'B must not write the arrangement back at the manager').toBe(0);
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
