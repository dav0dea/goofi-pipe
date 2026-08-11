import { test } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { barsMatchTheHeader, controlsSitAtOneGap } from '../lib/panelBar';

/* The coarse half of `panel-bar.spec.ts`, and the half that has to be measured rather than
   reasoned about: under a coarse pointer `--panel-header-h` IS `--hit`, so a 44px bar has exactly
   room for its 44px controls and nothing else. Every control in it that carries the tap floor as a
   BORDER box fits; one that carries it as a content box (a <select> did, at 44 + 2px of border)
   pushes the whole strip past the header. `touch-hit-floor.spec.ts` measures the other direction —
   that the floor survived the shortening. */
test('every panel’s toolbar is exactly as tall as the panel header above it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await barsMatchTheHeader(page);
});

/* The coarse half of the one-gap rule, and not a re-measure of a constant: under a coarse pointer
   the viewer bar's controls (215px) no longer fit the slack they are given (184px), so the strip's
   control host is a live horizontal scroller here and nowhere else. The gaps have to survive that. */
test('a panel’s toolbar spaces every control it holds by the strip’s own gap', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await controlsSitAtOneGap(page);
});
