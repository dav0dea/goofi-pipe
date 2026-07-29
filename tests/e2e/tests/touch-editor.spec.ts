import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * The node editor's coarse-pointer door for adding a node (R spec §3.2b).
 *
 * Adding a node had four routes: a port click, a canvas double-click, Tab, and the app header's
 * ＋ Add node. None of the first three is reachable on a phone — no double-click, no keyboard, and
 * a port click needs a node to already exist — and the header button is panel-local behaviour that
 * this same change removed (it was clipped off-screen at 412px anyway). A long press on empty
 * canvas is now the touch door, and it is the ONLY one, so it earns a gesture-level guard on top
 * of the recognizer's unit tests.
 *
 * Driven through CDP touch events, not `page.mouse`: under `hasTouch` Playwright's mouse API still
 * dispatches MOUSE events, whose `pointerType` is `mouse` — exactly the input this door is closed
 * to, so a mouse-driven "long press" would prove nothing.
 */

interface Touch {
	x: number;
	y: number;
}

async function touchSession(page: Page) {
	const cdp = await page.context().newCDPSession(page);
	const send = (type: string, touchPoints: Touch[]) =>
		cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
	return {
		down: (p: Touch) => send('touchStart', [p]),
		moveTo: (p: Touch) => send('touchMove', [p]),
		up: () => send('touchEnd', [])
	};
}

/**
 * A point where the flow pane really is the topmost element. One backend serves the whole run, so
 * the patch may carry nodes left by an earlier spec — a hardcoded centre point could land on a node
 * card (which must NOT arm the door) and green the test for the wrong reason.
 */
async function emptySpot(page: Page): Promise<Touch> {
	const spot = await page.locator('.svelte-flow__pane').first().evaluate((pane) => {
		const r = pane.getBoundingClientRect();
		for (let fy = 0.25; fy <= 0.75; fy += 0.1) {
			for (let fx = 0.2; fx <= 0.8; fx += 0.1) {
				const x = Math.round(r.left + r.width * fx);
				const y = Math.round(r.top + r.height * fy);
				if (document.elementFromPoint(x, y) === pane) return { x, y };
			}
		}
		return null;
	});
	expect(spot, 'the canvas has some empty space to press on').not.toBeNull();
	return spot!;
}

test('a long press on empty canvas opens the add-node menu there', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const touch = await touchSession(page);
	const at = await emptySpot(page);
	const menu = page.getByTestId('add-node-menu-anchor');

	await touch.down(at);
	await expect(menu, 'the press opens the menu while the finger is still down').toBeVisible();

	// The touchend that ENDS the opening gesture must not read as a dismissal: it lands on the
	// menu's own click-catcher, which the press itself mounted mid-gesture.
	await touch.up();
	await expect(menu, 'and releasing that same press does not close it again').toBeVisible();

	const box = (await menu.boundingBox())!;
	const vp = page.viewportSize()!;
	// Anchored to the press vertically; horizontally the shared viewport clamp owns the answer at
	// 412px (a 320px menu cannot open at mid-screen and still fit), which is its whole job.
	expect(Math.abs(box.y - at.y), 'the menu opens at the finger, not at some centred fallback')
		.toBeLessThan(40);
	expect(box.x, 'and the clamp keeps it fully on screen').toBeGreaterThanOrEqual(0);
	expect(box.x + box.width).toBeLessThanOrEqual(vp.width);

	// Escape only reaches the menu through the editor's own keydown handler, which stands down
	// unless its panel is active — so this closing also proves the press left `Panel`'s
	// capture-phase `setActive` alone rather than swallowing the pointerdown that drives it.
	await page.keyboard.press('Escape');
	await expect(menu).toHaveCount(0);
});

test('a pan is not a press — dragging the canvas opens nothing', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const touch = await touchSession(page);
	const at = await emptySpot(page);

	await touch.down(at);
	await touch.moveTo({ x: at.x + 60, y: at.y + 20 });
	// Hold well past the recognizer's window with the finger parked at its new spot.
	await page.waitForTimeout(900);
	await expect(page.getByTestId('add-node-menu-anchor'), 'a drag never arms the door').toHaveCount(
		0
	);
	await touch.up();
});
