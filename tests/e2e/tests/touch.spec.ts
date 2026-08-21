// One patch, built with a finger.
//
// AGENTS.md makes it a hard constraint that no interaction lives solely behind hover, right-click
// or a keyboard chord, and that touch gets its own door "gated on the one spelling a test
// enforces". This is that test. It is a SESSION rather than a suite because a gesture only means
// something in sequence: a tap after a long press, a drag after a tap, a second tap that is either
// a double or a new gesture depending on what came before.
//
// What is NOT here: tap-target sizes, coarse font sizes, and everything else about how the app
// MEASURES under a finger. Those are invariants now, swept over the whole scene by
// `integrity.spec.ts` under this same Pixel 7 project — a rule that names no element beats fifty
// tests that each name one.

import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, tapNode, waitForNode } from '../lib/goofi';
import { emptySpot, touchSession } from '../lib/touch';
import { pane } from '../lib/inspector';

/** A long press at `p`, the coarse door onto everything hover and right-click own on a desktop. */
async function longPress(page: Page, p: { x: number; y: number }, ms = 600): Promise<void> {
	const touch = await touchSession(page);
	await touch.down(p);
	await page.waitForTimeout(ms);
	await touch.up();
}

/** A finger drag from `a` to `b`, coming to REST before it lifts so Chromium reads no fling. */
async function swipe(
	page: Page,
	a: { x: number; y: number },
	b: { x: number; y: number },
	steps = 8
): Promise<void> {
	const touch = await touchSession(page);
	await touch.down(a);
	for (let i = 1; i <= steps; i++) {
		await touch.moveTo({
			x: Math.round(a.x + ((b.x - a.x) * i) / steps),
			y: Math.round(a.y + ((b.y - a.y) * i) / steps)
		});
	}
	await page.waitForTimeout(150);
	await touch.up();
}

async function tearDown(page: Page): Promise<void> {
	await page.evaluate(async () => {
		const g = (window as any).goofi;
		const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
		if (uids.length) await g.commands.removeNodes(uids);
	});
	await expect
		.poll(() => page.evaluate(() => (window as any).goofi.query.graph().nodes.length))
		.toBe(0);
}

test('a patch authored with a finger, and every door hover owns on a desktop', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	try {
		await test.step('a long press on bare canvas opens the add-node menu, where the finger was', async () => {
			// The coarse door onto right-click. Without it the canvas has no way in at all on a phone.
			const spot = await emptySpot(page);
			const menu = page.getByTestId('add-node-menu-anchor');
			const touch = await touchSession(page);
			await touch.down(spot);
			await expect(menu, 'the press opens it while the finger is still down').toBeVisible();
			// The touchend ENDING the opening gesture must not read as a dismissal: it lands on the
			// click-catcher the press itself mounted mid-gesture.
			await touch.up();
			await expect(menu, 'and the release does not close what the press opened').toBeVisible();
			const box = (await menu.boundingBox())!;
			expect(
				Math.abs(box.y - spot.y),
				'anchored to the finger vertically; the viewport clamp owns the horizontal at 412px'
			).toBeLessThan(40);
			expect(box.x, 'and the clamp keeps it wholly on screen').toBeGreaterThanOrEqual(0);
			expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize()!.width);
			await page.keyboard.press('Escape');
			await expect(menu).toHaveCount(0);
		});

		let osc = '';
		await test.step('a tap selects a node, and the inspector arrives as a sheet', async () => {
			osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, osc);
			await tapNode(page, osc);
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
				.toBe(1);
			await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
		});

		await test.step('a long press on a control tells what it does, and does NOT do it', async () => {
			// The other half of the hover door: a desktop learns a control's name by hovering it, and a
			// finger has to be able to ask without also pressing. Named by test id rather than by
			// ordinal — the header's right end is a progressive overflow, so which control is first
			// there is a function of the panel's width.
			const tip = page.getByTestId('title-tip');
			const btn = page.getByTestId('panel-maximize').first();
			const b = (await btn.boundingBox())!;
			const touch = await touchSession(page);
			await touch.down({ x: Math.round(b.x + b.width / 2), y: Math.round(b.y + b.height / 2) });
			await expect(tip, 'the press surfaces the title the finger is on').toHaveText('Maximize');
			await touch.up();
			// `toggleMaximize` would have flipped this to "Restore" had the press gone through as a
			// click. Asking what a control does must never be the same as using it.
			await expect(tip, 'and releasing leaves the answer standing').toHaveText('Maximize');
		});

		await test.step('…while a TAP acts, and raises no tip', async () => {
			const tabs = page.getByTestId('param-tabs');
			const other = tabs.getByRole('tab', { selected: false }).first();
			const name = (await other.textContent())!.trim();
			await other.tap();
			await expect(tabs.getByRole('tab', { selected: true })).toHaveText(name);
			await expect(page.getByTestId('title-tip'), 'a tap is not a press').toHaveCount(0);
		});

		await test.step('the sheet resizes by dragging its own seam, and a finger cannot throw it away', async () => {
			// The one gesture the pane has, identical to the mouse's. Dragging it far past its floor
			// used to dismiss the pane, which made the only way out of the inspector a swipe nobody
			// documented; the ✕ is the whole of it now.
			const grip = page.getByTestId('panel-resize-handle').first();
			const g = (await grip.boundingBox())!;
            const from = { x: Math.round(g.x + g.width / 2), y: Math.round(g.y + g.height / 2) };
			const before = (await pane(page).boundingBox())!;
			await swipe(page, from, { x: from.x, y: from.y + 400 });
			await expect(pane(page), 'still open — only the ✕ closes it').toHaveClass(/open/);
			const after = (await pane(page).boundingBox())!;
			expect(after.height, 'and the drag resized it').toBeLessThan(before.height);
		});

		await test.step('the ✕ is the way out, and the next tap brings the pane back', async () => {
			await pane(page).getByTestId('inspector-close').tap();
			await expect(pane(page)).not.toHaveClass(/open/);
			// A NEW selection, not the same one again: the ✕ dismisses the pane and leaves the node
			// selected, so re-tapping it changes nothing. That is what "not an off-switch" means — the
			// dismissal lasts exactly until the next selection.
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
				.toBe(0);
			await tapNode(page, osc);
			await expect(pane(page), 'a dismiss is not an off-switch').toHaveClass(/open/);
		});
		await test.step('a drag on the canvas PANS, and opens nothing', async () => {
			// LAST, because it moves the viewport: every step above has to be able to reach the node it
			// put there. A pan must not read as the press that opens the menu, or the canvas becomes
			// unusable the moment anyone scrolls it.
			const from = await emptySpot(page);
			await swipe(page, from, { x: from.x - 90, y: from.y - 60 });
			await expect(
				page.getByTestId('add-node-menu-anchor'),
				'a pan is not a press'
			).toHaveCount(0);
		});
	} finally {
		await tearDown(page);
	}
});
