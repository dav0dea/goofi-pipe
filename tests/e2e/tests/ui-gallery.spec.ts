import { test, expect } from '@playwright/test';

// The /dev/ui primitive gallery is a static, backend-free showcase, so we do NOT wait on
// window.goofi (the AppShell/control-WS readiness gate the graph specs use) — we wait on the
// rendered samples. This spec is the "failing test first" for <Button>/<IconButton>: it fails
// before the route + primitives exist. Runs under the `default` (fine-pointer) project.
test.describe('UI primitives gallery', () => {
	test('renders every Button and IconButton variant/size sample', async ({ page }) => {
		await page.goto('/dev/ui');
		// A spread across variants + sizes for both primitives.
		await expect(page.getByTestId('ui-button-default-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-primary-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-ghost-sm')).toBeVisible();
		await expect(page.getByTestId('ui-button-danger-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-disabled')).toBeDisabled();
		await expect(page.getByTestId('ui-icon-primary-md')).toBeVisible();
		await expect(page.getByTestId('ui-icon-danger-sm')).toBeVisible();
	});

	test('a keyboard-focused Button shows the app accent focus ring', async ({ page }) => {
		await page.goto('/dev/ui');
		await page.getByTestId('ui-button-default-sm').waitFor();
		// Keyboard Tab (not programmatic focus) so :focus-visible engages; the first focusable
		// element on this static page is the first sample button.
		await page.keyboard.press('Tab');
		const ring = await page.evaluate(() => {
			const el = document.activeElement as HTMLElement | null;
			if (!el || el === document.body) return { tag: '', testid: '', outlineWidth: '', outlineColor: '' };
			const s = getComputedStyle(el);
			return {
				tag: el.tagName,
				testid: el.getAttribute('data-testid') ?? '',
				outlineWidth: s.outlineWidth,
				outlineColor: s.outlineColor
			};
		});
		expect(ring.tag, 'Tab moves focus onto a gallery button').toBe('BUTTON');
		expect(ring.testid, 'the focused element is a UI primitive sample').toMatch(/^ui-button-/);
		// Assert the app rule specifically (2px solid --accent = #50d0a0), not merely "some outline",
		// so a future UA-default outline could not pass this as a tautology.
		expect(ring.outlineWidth, 'the app :focus-visible ring is 2px').toBe('2px');
		expect(ring.outlineColor, 'the ring colour is --accent (#50d0a0)').toBe('rgb(80, 208, 160)');
	});
});

// The layout primitives (Task 2). Each assertion is behavioural, not "renders": the token gap must
// actually separate children, ScrollArea must actually scroll, and Bar must actually push its end
// group to the right. Runs under the `default` (fine-pointer) project like the rest of this file.
test.describe('UI layout primitives gallery', () => {
	// The px `var(--space-N)` resolves to at THIS viewport (the html clamp makes rem viewport-relative),
	// so assertions compare the measured spacing to the token's own resolved value — never a hardcoded px.
	async function spacePx(page: import('@playwright/test').Page, key: number): Promise<number> {
		return page.evaluate((k) => {
			const probe = document.createElement('div');
			probe.style.position = 'absolute';
			probe.style.width = `var(--space-${k})`;
			document.body.appendChild(probe);
			const px = parseFloat(getComputedStyle(probe).width);
			probe.remove();
			return px;
		}, key);
	}

	test('Row lays its children with the token gap between them', async ({ page }) => {
		await page.goto('/dev/ui');
		const row = page.getByTestId('ui-row-gap4');
		await row.waitFor();
		// The container's computed column-gap is the F --space-4 token — not a literal.
		const gap = parseFloat(await row.evaluate((el) => getComputedStyle(el).columnGap));
		expect(gap, 'Row column-gap equals the resolved --space-4 token').toBeCloseTo(await spacePx(page, 4), 1);
		expect(gap, 'the gap is non-zero').toBeGreaterThan(0);
		// And that gap actually separates adjacent children (the point of the primitive).
		const a = (await page.getByTestId('ui-row-child-a').boundingBox())!;
		const b = (await page.getByTestId('ui-row-child-b').boundingBox())!;
		expect(b.x - (a.x + a.width), 'child B starts one token-gap right of child A').toBeCloseTo(gap, 0);
	});

	test('Stack lays its children with the token gap between them', async ({ page }) => {
		await page.goto('/dev/ui');
		const stack = page.getByTestId('ui-stack-gap4');
		await stack.waitFor();
		const gap = parseFloat(await stack.evaluate((el) => getComputedStyle(el).rowGap));
		expect(gap, 'Stack row-gap equals the resolved --space-4 token').toBeCloseTo(await spacePx(page, 4), 1);
		const a = (await page.getByTestId('ui-stack-child-a').boundingBox())!;
		const b = (await page.getByTestId('ui-stack-child-b').boundingBox())!;
		expect(b.y - (a.y + a.height), 'child B starts one token-gap below child A').toBeCloseTo(gap, 0);
	});

	test('ScrollArea scrolls when its content overflows', async ({ page }) => {
		await page.goto('/dev/ui');
		const sc = page.getByTestId('ui-scrollarea');
		await sc.waitFor();
		const metrics = await sc.evaluate((el) => ({ scroll: el.scrollHeight, client: el.clientHeight }));
		expect(metrics.scroll, 'content overflows the box').toBeGreaterThan(metrics.client);
		// overflow-y:auto is really active: scrollTop can only move past 0 on a scrollable element.
		const moved = await sc.evaluate((el) => {
			el.scrollTop = 40;
			return el.scrollTop;
		});
		expect(moved, 'the ScrollArea actually scrolled').toBeGreaterThan(0);
	});

	test('Bar pushes its end group to the right edge (the pusher pattern)', async ({ page }) => {
		await page.goto('/dev/ui');
		const bar = page.getByTestId('ui-bar');
		await bar.waitFor();
		const barBox = (await bar.boundingBox())!;
		const start = (await page.getByTestId('ui-bar-start').boundingBox())!;
		const end = (await page.getByTestId('ui-bar-end').boundingBox())!;
		// start hugs the left, end is pushed to the right (a flex spacer between them).
		expect(start.x - barBox.x, 'start group hugs the left').toBeLessThan(barBox.width * 0.25);
		expect(end.x, 'end group sits in the right half of the bar').toBeGreaterThan(barBox.x + barBox.width * 0.5);
		expect(
			barBox.x + barBox.width - (end.x + end.width),
			'end group hugs the right edge (within the bar padding)'
		).toBeLessThan(barBox.width * 0.2);
	});
});
