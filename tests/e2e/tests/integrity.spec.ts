// One complex scene, built up in stages, swept for structural violations after each.
//
// This is the whole of the app's visual coverage, and deliberately so. It asserts NO design value:
// every rule in `lib/invariants` compares the app against itself or against its own `--hit` token,
// so a restyle is free and only the app falling apart is red — a page that scrolls, text clipped
// away, a control off screen or too small to hit.
//
// It runs in every viewport project, which is what makes it a responsive test: the same scene at
// 1280, at 412 portrait, at 863 landscape and on a tablet, judged by one rule set.

import { test, expect, type Page } from '@playwright/test';
import { closeAddedTab, closeSplit, splitRight, waitForApp } from '../lib/app';
import { expectIntact } from '../lib/invariants';
import { addNode, selectNode, waitForNode } from '../lib/goofi';

/** Open the panel header menu, reveal its content submenu, and answer every row it shows. */
async function panelTypeRows(page: Page): Promise<string[]> {
	await page.getByTestId('panel-header').first().click({ button: 'right' });
	await page.locator('.context-menu .item', { hasText: 'Change content' }).first().hover();
	await expect(page.locator('.context-menu .item').nth(6)).toBeVisible();
	const rows = (await page.locator('.context-menu .item').allTextContents()).map((t) => t.trim());
	await page.keyboard.press('Escape');
	return rows;
}

/** Switch the first panel to the type named in its own switcher. */
async function choosePanelType(page: Page, name: string): Promise<void> {
	await page.getByTestId('panel-header').first().click({ button: 'right' });
	await page.locator('.context-menu .item', { hasText: 'Change content' }).first().hover();
	await page.locator('.context-menu .item', { hasText: new RegExp(`^\\s*${name}\\s*$`) })
		.first()
		.click();
	await expect(page.locator('.context-menu')).toHaveCount(0);
}

/** Everything this scene put on the backend, so the next spec meets a pristine workspace. */
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

test('a patch under construction holds together at every stage', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const wide = (page.viewportSize()?.width ?? 0) >= 900;
	try {
		await test.step('an empty workspace', async () => {
			await expectIntact(page, 'the empty app');
		});

		let osc = '';
		let buf = '';
		await test.step('two nodes on the canvas, wired', async () => {
			osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, osc);
			buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
			await waitForNode(page, buf);
			await page.evaluate(
				([a, b]) =>
					(window as any).goofi.commands.addLink({
						node_out: a,
						slot_out: 'out',
						node_in: b,
						slot_in: 'data'
					}),
				[osc, buf]
			);
			await expectIntact(page, 'a wired pair');
		});

		await test.step('a viewer streaming real frames', async () => {
			// A node that is RUNNING, not merely present: a viewer sizes itself around live data, and
			// an empty one cannot overflow the way a full one can.
			await expect
				.poll(
					() => page.evaluate((u) => (window as any).goofi.query.frameSummary(u, 'out') !== null, buf),
					{ message: 'frames reached the tab', timeout: 30_000 }
				)
				.toBe(true);
			await expectIntact(page, 'a streaming viewer');
		});

		await test.step('the inspector open over it, with every param group rendered', async () => {
			await selectNode(page, osc);
			await expect(page.getByTestId('auto-side-panel')).toHaveClass(/open/);
			await expectIntact(page, 'the inspector open');
			const tabs = page.getByTestId('param-tabs');
			for (const name of await tabs.getByRole('tab').allTextContents()) {
				await tabs.getByRole('tab', { name }).click();
				await expectIntact(page, `the inspector on its ${name} group`);
			}
		});

		await test.step('a long name does not burst the boxes that carry it', async () => {
			// The scene's one adversarial input. Everything else here is well-behaved content, and a
			// layout only falls apart when something in it is bigger than the design imagined. Renamed
			// through the inspector's own header, because there is no rename on the command façade.
			const panel = page.getByTestId('auto-side-panel');
			await panel.getByTestId('node-name').click();
			const input = panel.getByTestId('node-name-input');
			await input.fill('a_node_name_far_longer_than_any_header_was_drawn_to_hold');
			await input.press('Enter');
			await expect
				.poll(() => page.evaluate((u) => (window as any).goofi.query.node(u)?.name, osc))
				.toContain('far_longer');
			await expectIntact(page, 'a very long node name');
		});

		if (wide) {
			await test.step('a second panel beside the first', async () => {
				// The inspector is parked first: it is an overlay ON the editor, and a split under an
				// open one puts the new panel's header beneath it.
				await page.getByTestId('auto-side-panel').getByTestId('inspector-close').click();
				await expect(page.getByTestId('auto-side-panel')).not.toHaveClass(/open/);
				await splitRight(page);
				await expectIntact(page, 'a split workspace');
				await closeSplit(page);
			});
		}

		await test.step('a second workspace tab', async () => {
			await page.evaluate(() => (window as any).goofi.commands.addTab());
			await expect(page.getByTestId('workspace-tabs').locator('.ui-tab')).toHaveCount(2);
			await expectIntact(page, 'a second tab');
			await closeAddedTab(page);
		});

		if (wide) {
			await test.step('every panel type the build offers, one after another', async () => {
				// The sweep names no element, so a panel type added later is covered the day it renders —
				// but only if the scene MOUNTS it. The list comes from the panel's own switcher rather
				// than from a literal here, so a new type joins this walk by existing.
				const CHROME = ['split', 'maximize', 'change content', 'close'];
				const names = (await panelTypeRows(page)).filter(
					(n) => !CHROME.some((c) => n.toLowerCase().includes(c))
				);
				expect(names.length, 'the switcher offers types to walk').toBeGreaterThan(3);
				for (const name of names) {
					await choosePanelType(page, name);
					await expectIntact(page, `the ${name} panel`);
				}
				await choosePanelType(page, 'Node Editor');
			});
		}
	} finally {
		await tearDown(page);
	}
});

test('the primitive gallery holds together', async ({ page }) => {
	// `/dev/ui` renders one sample of every primitive the `$lib/ui` barrel exports, so sweeping it
	// asks the integrity question of the whole component library at once — including the primitives
	// no screen in the app happens to be showing right now. It is served only under `--debug`, which
	// is how the fleet spawns.
	await page.goto('/dev/ui');
	await expect(page.getByTestId('ui-button-default-md')).toBeVisible();
	await expectIntact(page, 'the primitive gallery');
});
