import { expect, type Page } from '@playwright/test';

/**
 * Real touch input, driven through CDP.
 *
 * Not `page.mouse`: under `hasTouch` Playwright's mouse API still dispatches MOUSE events, whose
 * `pointerType` is `mouse` — exactly the input the coarse doors are closed to, so a mouse-driven
 * "long press" would prove nothing. Playwright's own `tap()` is a tap and cannot hold, which is
 * what a long press is.
 *
 * Shared through `lib/` rather than by importing one spec from another, since loading a spec file
 * is how Playwright registers its tests.
 */
export interface TouchPoint {
	x: number;
	y: number;
}

export interface TouchSession {
	down(p: TouchPoint): Promise<unknown>;
	moveTo(p: TouchPoint): Promise<unknown>;
	up(): Promise<unknown>;
}

export async function touchSession(page: Page): Promise<TouchSession> {
	const cdp = await page.context().newCDPSession(page);
	const send = (type: string, touchPoints: TouchPoint[]) =>
		cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
	return {
		down: (p: TouchPoint) => send('touchStart', [p]),
		moveTo: (p: TouchPoint) => send('touchMove', [p]),
		up: () => send('touchEnd', [])
	};
}

/**
 * Fake a soft keyboard by shrinking `visualViewport.height` (an own property shadows the prototype
 * getter) and firing the resize the real keyboard would fire. `px = 0` restores it.
 *
 * Here rather than in a spec because two specs need the same fake: `device-stamp.spec.ts` proves
 * `--kb-inset` tracks it and that the two anchored-overlay clamps read it, and `touch-expr.spec.ts`
 * proves the completion popup does. A second copy is a second thing to keep true.
 */
export async function setKeyboardInset(page: Page, px: number): Promise<void> {
	await page.evaluate((n) => {
		const vv = window.visualViewport as VisualViewport & { height?: number };
		if (n > 0) {
			Object.defineProperty(vv, 'height', {
				configurable: true,
				get: () => window.innerHeight - n
			});
		} else {
			delete vv.height;
		}
		vv.dispatchEvent(new Event('resize'));
	}, px);
}

/** The published `--kb-inset`, as the app computed it. */
export const kbInset = (page: Page): Promise<string> =>
	page.evaluate(() =>
		getComputedStyle(document.documentElement).getPropertyValue('--kb-inset').trim()
	);

/**
 * A point where the flow pane really is the topmost element AND no node card is within a tap target
 * of it. One backend serves the whole run, so the patch may carry nodes left by an earlier spec — a
 * hardcoded centre point could land on a node card (which must NOT arm the long-press door) and
 * green a test for the wrong reason.
 *
 * `elementFromPoint` answers "is this bare canvas" exactly. A TOUCH does not ask it that way:
 * Chromium applies **touch adjustment**, snapping a tap within roughly a finger's radius onto a
 * nearby clickable box. Measured here at 10px — a tap 10px clear of a node card's right edge came
 * back as a `click` on the card's edge, so a tap meant to deselect re-selected the node it was
 * beside and the canvas looked dead when it was not. So the point must clear every rendered card by
 * a full `--hit` as well as being provably bare.
 */
export async function emptySpot(page: Page): Promise<TouchPoint> {
	const spot = await bareSpot(page, [0.25, 0.35, 0.45, 0.55, 0.65, 0.75], 0);
	expect(spot, 'the canvas has some empty space to press on').not.toBeNull();
	return spot!;
}

/**
 * The same bare point, but as LOW on the canvas as one can be found — at or below `minY` in viewport
 * coordinates. The add-node menu's clamp only engages when the press sits within a menu's height of
 * the bottom, and that is the geometry in which the menu covers the finger, so a spec about the
 * clamp has to be able to ask for a press point there rather than in the comfortable middle.
 */
export async function lowEmptySpot(page: Page, minY: number): Promise<TouchPoint> {
	const spot = await bareSpot(page, [0.98, 0.93, 0.88, 0.83, 0.78, 0.73, 0.68], minY);
	expect(spot, `the canvas has empty space below y=${minY} to press on`).not.toBeNull();
	return spot!;
}

/** Scan `fys` (fractions of the pane's height, in order) for the first bare, card-clear point at or
 * below `minY`. One copy of the bareness rule the doc comment above spells out. */
async function bareSpot(page: Page, fys: number[], minY: number): Promise<TouchPoint | null> {
	return page.locator('.svelte-flow__pane').first().evaluate(
		(pane, { fys, minY }) => {
			const hit = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit'));
			const cards = [...document.querySelectorAll('.svelte-flow__node')].map((n) =>
				n.getBoundingClientRect()
			);
			const clear = (x: number, y: number): boolean =>
				cards.every(
					(c) => x < c.left - hit || x > c.right + hit || y < c.top - hit || y > c.bottom + hit
				);
			const r = pane.getBoundingClientRect();
			for (const fy of fys) {
				for (let fx = 0.2; fx <= 0.8; fx += 0.1) {
					const x = Math.round(r.left + r.width * fx);
					const y = Math.round(r.top + r.height * fy);
					if (y < minY) continue;
					if (document.elementFromPoint(x, y) === pane && clear(x, y)) return { x, y };
				}
			}
			return null;
		},
		{ fys, minY }
	);
}
