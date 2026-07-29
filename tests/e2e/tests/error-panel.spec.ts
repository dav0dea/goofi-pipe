import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

// M-Task 2 gave the ErrorPanel the real dismissal it never had: the error dropdown used to only
// toggle from the chip — no Escape, no outside-click. It now delegates to the Popover primitive,
// so this asserts the NEW behaviour on the REAL panel, driven by a REAL node error.
//
// The error is reachable and environment-independent: the tier calls `node.process(**present)` with
// only the CONNECTED inputs as kwargs (goofi-pymod exec.rs `run_process`), and `LempelZiv.process`
// has a required `data` param — so a node added with nothing connected raises a missing-argument
// TypeError on its first tick. That is a per-tick Python-node error, exactly the "a node that raises"
// case the panel exists to surface (it sets the same `error` field that drives the floating chip).
// It does NOT depend on any missing dependency (numpy is present in both venvs) — it is the
// unconnected-required-input raise, which is permanent. The node is torn down after each test so the
// shared backend graph stays clean for later specs.
test.describe('ErrorPanel dismissal (delegated to Popover)', () => {
	let created: string[] = [];

	test.afterEach(async ({ page }) => {
		for (const uid of created) {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
		created = [];
	});

	async function summonErrorChip(page: Page) {
		await page.goto('/');
		await waitForApp(page);
		// A Python node whose process() needs a connected ARRAY input; added with none, it raises on
		// its first tick and the error surfaces as the floating chip.
		const uid = await addNode(page, 'LempelZiv', 'python');
		created.push(uid);
		await waitForNode(page, uid);
		const chipHost = page.getByTestId('error-chip');
		await expect(chipHost, 'a real node error raises the floating error chip').toBeVisible();
		return chipHost;
	}

	test('the chip opens the error popover, and Escape dismisses it', async ({ page }) => {
		const chipHost = await summonErrorChip(page);
		const popover = page.getByTestId('error-popover');
		const chip = chipHost.locator('button').first();

		await expect(popover, 'closed by default').toBeHidden();
		// The chip is a disclosure trigger, and says so — the surface it opens carries no popup role
		// by design (its children are plain buttons), so `aria-expanded` is the whole story it can tell.
		await expect(chip, 'the trigger reports its collapsed state').toHaveAttribute(
			'aria-expanded',
			'false'
		);
		await chip.click();
		await expect(popover, 'clicking the chip opens the popover').toBeVisible();
		await expect(chip, 'and its expanded one').toHaveAttribute('aria-expanded', 'true');

		await page.keyboard.press('Escape');
		await expect(popover, 'Escape dismisses the popover (new behaviour)').toBeHidden();
		await expect(chip, 'and back to collapsed').toHaveAttribute('aria-expanded', 'false');
	});

	test('an outside click dismisses the error popover', async ({ page }) => {
		const chipHost = await summonErrorChip(page);
		const popover = page.getByTestId('error-popover');

		await chipHost.locator('button').click();
		await expect(popover).toBeVisible();
		// Click the top-left of the viewport — neither the popover surface nor the chip anchor.
		await page.mouse.click(10, 10);
		await expect(popover, 'an outside click dismisses the popover (new behaviour)').toBeHidden();
	});

	// The chip is anchored to the BOTTOM of the editor (`bottom: 12px`), and the clamp only ever
	// shifted — never flipped — so the surface slid up until it fit and then ENCLOSED the chip on both
	// axes at a higher z-index. Two consequences: the chip's toggle-to-close (the pre-M panel's only
	// dismissal) became unreachable, and the click was not inert — it landed on the first error row and
	// focused a node the user never asked for. Pre-M this panel opened upward.
	test('the popover opens clear of its own chip, and the chip still toggles it closed', async ({
		page
	}) => {
		const chipHost = await summonErrorChip(page);
		const chip = chipHost.locator('button');
		const popover = page.getByTestId('error-popover');

		await chip.click();
		await expect(popover).toBeVisible();
		const chipBox = (await chip.boundingBox())!;
		// The Popover primitive positions the surface itself; measure the surface, not the list inside.
		const menuBox = (await page.locator('.ui-popover').boundingBox())!;
		const overlaps =
			chipBox.x < menuBox.x + menuBox.width &&
			menuBox.x < chipBox.x + chipBox.width &&
			chipBox.y < menuBox.y + menuBox.height &&
			menuBox.y < chipBox.y + chipBox.height;
		expect(overlaps, 'the popover does not cover the chip that opens it').toBe(false);

		const before = await page.evaluate(() => (window as any).goofi.query.selection());
		await chip.click(); // the toggle-to-close — must reach the chip, and must be inert otherwise
		await expect(popover, 'the chip toggles its own popover closed').toBeHidden();
		expect(
			await page.evaluate(() => (window as any).goofi.query.selection()),
			'closing the popover selected nothing behind it'
		).toEqual(before);
	});

	// The row inside the popover is the one M kept bespoke here (a stacked name-over-message list
	// row, not an action). M-Task 7 strips app.css's base `button` SKIN, so it must render from its
	// own rule alone — including the fade on its hover fill, which it was inheriting from that skin
	// until the strip made the dependency visible. This is the only place the row is reachable.
	test('the kept-bespoke error row renders from its own rule, not the base skin', async ({
		page
	}) => {
		const chipHost = await summonErrorChip(page);
		await chipHost.locator('button').click();
		const row = page.locator('.error-list .prow').first();
		await expect(row).toBeVisible();
		// Park the cursor away from the list and poll until the hover fill has faded back out (the
		// row's own transition), or the probe samples a mid-fade colour rather than the rest state.
		await page.mouse.move(5, 5);
		await expect
			.poll(() => row.evaluate((el) => getComputedStyle(el).backgroundColor), {
				message: 'the row is transparent at rest'
			})
			.toBe('rgba(0, 0, 0, 0)');
		const s = await row.evaluate((el) => {
			const cs = getComputedStyle(el);
			return {
				fontFamily: cs.fontFamily,
				fontSize: parseFloat(cs.fontSize),
				background: cs.backgroundColor,
				borderWidth: cs.borderTopWidth,
				radius: cs.borderTopLeftRadius,
				padTop: parseFloat(cs.paddingTop),
				padLeft: parseFloat(cs.paddingLeft),
				transition: cs.transitionProperty,
				rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
			};
		});
		expect(s.fontFamily, 'the error row renders in the app mono face').toContain('JetBrains Mono');
		expect(s.fontSize, 'it inherits the popover surface size (`font: inherit`)').toBeCloseTo(
			0.82 * s.rem,
			0
		);
		expect(s.background, 'transparent at rest').toBe('rgba(0, 0, 0, 0)');
		expect(s.borderWidth, 'borderless').toBe('0px');
		expect(s.radius, 'its hover fill is rounded (--radius-sm)').toBe('4px');
		expect(s.padTop).toBeCloseTo(0.375 * s.rem, 0);
		expect(s.padLeft).toBeCloseTo(0.625 * s.rem, 0);
		expect(s.transition, 'the hover fill fades in').toContain('background');
	});
});
