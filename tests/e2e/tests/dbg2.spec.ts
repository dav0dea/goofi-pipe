import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

test('dbg modal keys', async ({ page }) => {
	page.on('console', (m) => console.log('PAGE:', m.text()));
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await page.evaluate(() => { (window as any).__k = []; window.addEventListener('keydown', (e) => (window as any).__k.push({ key: e.key, tag: (e.target as HTMLElement).tagName, td: (e.target as HTMLElement).dataset?.testid, inDialog: !!(e.target as HTMLElement).closest?.('dialog[open]') })); });
	await page.getByTestId('topbar-load').click();
	await expect(page.getByTestId('fs-browser')).toBeVisible();
	console.log('active', await page.evaluate(() => { const el = document.activeElement as HTMLElement; return `${el.tagName}|${el.dataset?.testid}|${el.getAttribute('aria-label')}`; }));
	console.log('activePanelId', await page.evaluate(() => (window as any).goofi.query.panels().map((p:any)=>p.panelId)));
	await page.keyboard.press('Escape');
	await page.waitForTimeout(300);
	console.log('KEYS', JSON.stringify(await page.evaluate(() => (window as any).__k)));
	console.log('selection', JSON.stringify(await page.evaluate(() => (window as any).goofi.query.selection())));
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
});
