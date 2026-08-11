import { test } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { barsMatchTheHeader, controlsSitAtOneGap } from '../lib/panelBar';

/* One height for every chrome strip. A panel's toolbar was its content's height plus its own
   vertical padding — 34.8px under a 26px header on a desktop, 53px under a 44px one on a phone —
   so the two strips a panel stacks read as two unrelated bands. `touch-panel-bar.spec.ts` is the
   same assertion under a coarse pointer, where the floor and the height pull hardest against each
   other and the bar has NO padding left to spend. */
test('every panel’s toolbar is exactly as tall as the panel header above it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await barsMatchTheHeader(page);
});

/* And one gap across that strip. The height rule above says the toolbar is one band; this says its
   contents read as one row — the node picker and the panel's own controls are siblings, not two
   clusters separated by a margin. */
test('a panel’s toolbar spaces every control it holds by the strip’s own gap', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await controlsSitAtOneGap(page);
});
