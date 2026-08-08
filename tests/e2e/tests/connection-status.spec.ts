import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * What the header says about the control connection — which, while it is healthy, is nothing.
 *
 * Phil's call: "the 'connected' state should not indicate anything, we don't need to communicate
 * 'everything is good'. Let's only communicate when something needs attention." So the badge that
 * used to sit in the status cluster at every moment of the app's life — restating the one fact that
 * is true in every screenshot of a working app — is gone, and the width it spent goes to the
 * filename and the tab strip that were fighting over it.
 *
 * The alarm it becomes is deliberately louder than the badge it replaces: the badge alone is 72px
 * of a 412px bar and easy to miss, so the whole header wears a thick warning outline as well.
 */

/** Record the app's control socket, and give the test a way to cut it the way a backend going away
 * does — a real `close` through the client's own handler, not a store poke.
 *
 * The reconnect is frozen deliberately. `api/control.ts` retries on a 250ms→5s backoff, so a bare
 * close would flicker (drop → reconnect → drop) under a running backend and every assertion below
 * would be racing it. Replacing the constructor with one that never fires `open` OR `close` leaves
 * the client exactly where a real unreachable backend leaves it: disconnected, one dangling attempt
 * outstanding, nothing scheduled. */
async function armControlCut(page: import('@playwright/test').Page): Promise<void> {
	await page.addInitScript(() => {
		const Native = window.WebSocket;
		const control: WebSocket[] = [];
		class Recorded extends Native {
			constructor(url: string | URL, protocols?: string | string[]) {
				super(url, protocols);
				if (String(url).includes('/control')) control.push(this);
			}
		}
		window.WebSocket = Recorded as unknown as typeof WebSocket;
		(window as any).__cutControl = () => {
			window.WebSocket = class {
				readyState = 0;
				addEventListener() {}
				removeEventListener() {}
				send() {}
				close() {}
			} as unknown as typeof WebSocket;
			for (const ws of control) ws.close();
		};
	});
}

test('says nothing at all while the connection is healthy', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await expect(page.getByTestId('topbar-connection'), 'no chip, and no width spent').toHaveCount(0);
	// Read off `textContent`, like `topbar.spec.ts` reads the brand: the claim is that the header
	// carries no connection state at all, not merely that it is painted out of sight.
	const text = ((await page.locator('.topbar').textContent()) ?? '').toLowerCase();
	expect(text, 'a healthy socket is not news').not.toContain('connect');
	await expect(page.locator('.topbar'), 'and the bar wears no alarm').toHaveCSS(
		'outline-style',
		'none'
	);
});

test('a LOST connection takes space in the bar and outlines it, without moving anything', async ({
	page
}) => {
	await armControlCut(page);
	await page.goto('/');
	await waitForApp(page);
	const bar = page.locator('.topbar');
	const before = await bar.boundingBox();

	await page.evaluate(() => (window as any).__cutControl());

	// (a) the chip, in the status cluster — where D-R6 keeps it out of the progressive overflow, so
	// a warning can never spill into a menu the user has to open to find it.
	const chip = page.locator('.topbar .status [data-testid="topbar-connection"]');
	await expect(chip).toHaveText(/disconnected/i);
	await expect(
		page.locator('.topbar .actions [data-testid="topbar-connection"]'),
		'it is not one of the spillable actions'
	).toHaveCount(0);

	// (b) the whole bar, outlined in the warning ink — thick enough to catch the eye that misses a
	// 72px chip. Retrying `toHaveCSS`, never a one-shot evaluate.
	await expect(bar).toHaveCSS('outline-style', 'solid');
	await expect(bar).toHaveCSS('outline-width', '3px');
	await expect(bar).toHaveCSS('outline-color', 'rgb(240, 192, 80)'); // --warning

	// …and nothing moved. An outline is painted outside the box model, so the bar keeps its height
	// and the workspace below it keeps its origin — the whole reason the alarm is not a border.
	expect(await bar.boundingBox(), 'the alarm costs no layout').toEqual(before);
});
