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
 * A point where the flow pane really is the topmost element. One backend serves the whole run, so
 * the patch may carry nodes left by an earlier spec — a hardcoded centre point could land on a node
 * card (which must NOT arm the long-press door) and green a test for the wrong reason.
 */
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

export async function emptySpot(page: Page): Promise<TouchPoint> {
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
