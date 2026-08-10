import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * R Task 5 (§3.1h) — dragging a node onto a panel is the ONLY way to populate a Viewer /
 * Parameters / Metadata / Console panel (`NodeLinkedPanel` literally says "Drag a node here"), and
 * it never worked under touch: `linkTargetAt` read `event.clientX`, which a `TouchEvent` does not
 * have, so it returned null on every touch drag and `ws.linkNodeToPanel` was unreachable.
 * `editor/eventPoint.ts` owns the extraction now (unit-tested); this is the wiring proof.
 */

interface Pt {
	x: number;
	y: number;
}

async function touchSession(page: Page) {
	const cdp = await page.context().newCDPSession(page);
	const send = (type: string, touchPoints: Pt[]) =>
		cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
	return {
		down: (p: Pt) => send('touchStart', [p]),
		moveTo: (p: Pt) => send('touchMove', [p]),
		up: (p: Pt) => send('touchEnd', [p])
	};
}

const panels = (page: Page) => page.locator('[data-panel-id]');

test('a node dragged onto a viewer panel binds it, under touch', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const before = await panels(page).count();
	const touch = await touchSession(page);

	await page.getByTestId('panel-header').first().click({ button: 'right' });
	await page.locator('.context-menu .item', { hasText: 'Split Right' }).first().click();
	await expect(panels(page)).toHaveCount(before + 1);

	const uid = await addNode(page, 'Oscillator', 'inputs', [10, 10]);
	try {
		await waitForNode(page, uid);
		const target = panels(page).nth(1);
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'viewer'),
			await target.getAttribute('data-panel-id')
		);

		const head = (await page.locator(`.goofi-node .header`).first().boundingBox())!;
		const dst = (await target.boundingBox())!;
		const from = { x: Math.round(head.x + head.width / 2), y: Math.round(head.y + head.height / 2) };
		const to = { x: Math.round(dst.x + dst.width / 2), y: Math.round(dst.y + dst.height / 2) };

		await touch.down(from);
		// Two steps: the first is what SvelteFlow reads as "this is a drag", the second lands it.
		await touch.moveTo({ x: from.x + 12, y: from.y + 12 });
		await touch.moveTo(to);
		await touch.up(to);

		// The bind is a `page_set_panel` command, so it shows up when the manager's delta does.
		await expect
			.poll(
				() =>
					page.evaluate(
						() =>
							(
								(window as any).goofi.query.panels() as Array<{ type: string; node: string | null }>
							).find((p) => p.type === 'viewer')?.node ?? null
					),
				{ message: 'the drop bound the node to the viewer panel' }
			)
			.toBe(uid);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await page
			.getByTestId('panel-header')
			.nth(1)
			.getByRole('button', { name: 'Close panel' })
			.click();
		await expect(panels(page)).toHaveCount(before);
	}
});
