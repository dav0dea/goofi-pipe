import { test, expect, type Locator, type Page } from '@playwright/test';
import { closeSplit, splitRight, waitForApp } from '../lib/app';

/**
 * The panel header's progressive overflow — D-R6's arithmetic (`editor/overflowFit.ts`) at its
 * second consumer.
 *
 * A panel header is not the app header: its width is the PANEL's, not the window's, so two panels
 * side by side on a laptop are already narrower than the app header ever gets on a phone. That is
 * why this is a `default`-project spec and not a touch one — the collapse keys on available width,
 * and this is the cheapest place to drive width across its whole range.
 *
 * Two things separate it from `topbar-overflow.spec.ts`:
 *   · the ✕ is NOT in the plan. It is the one control that must be reachable at every width, so it
 *     never spills, and the header must never push it out of its own panel either.
 *   · the ⋯ is NOT resident. Its menu holds the spilled actions and nothing else, so at a width
 *     where all three fit there is no menu to open and the trigger is not drawn.
 */

/** The header's overflow-able actions, in DOM order (which is also the order they are given up). */
const ACTIONS = ['panel-split-row', 'panel-split-column', 'panel-maximize'];

const hdr = (page: Page, n = 0): Locator => page.getByTestId('panel-header').nth(n);

/** Which of the three actions the header is currently RENDERING, in DOM order (a spilled one stays
 *  in the tree at `display: none`, so its width can be re-read when the root size moves). */
function inHeader(page: Page, n = 0): Promise<string[]> {
	return hdr(page, n).evaluate(
		(el, ids) =>
			[...el.querySelectorAll<HTMLElement>('button[data-testid]')]
				.filter((b) => ids.includes(b.dataset.testid!) && b.offsetParent !== null)
				.map((b) => b.dataset.testid!),
		ACTIONS
	);
}

/** Resize and let the ResizeObserver settle (it runs after layout, before the next paint). */
async function widthTo(page: Page, width: number): Promise<void> {
	await page.setViewportSize({ width, height: 720 });
	await page.evaluate(
		() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
	);
}

function menuRow(page: Page, label: string): Locator {
	return page
		.locator('.context-menu .item')
		.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
}

async function openOverflow(page: Page, n = 0): Promise<void> {
	await hdr(page, n).getByTestId('panel-overflow').click();
	await expect(page.locator('.context-menu').first()).toBeVisible();
}

const panels = (page: Page): Locator => page.locator('.panel');

/** Narrow the window until the first panel's header has given up all three actions, and answer the
 *  width that did it. Searched rather than hardcoded: how narrow "narrow" is depends on the panel
 *  type's own name, on the number of panels sharing the window, and on wherever the responsive
 *  root-size clamp lands — none of which is what these tests are about. */
async function collapseFully(page: Page): Promise<number> {
	for (const w of [900, 760, 640, 560, 480, 420, 360, 320]) {
		await widthTo(page, w);
		if ((await inHeader(page)).length === 0) return w;
	}
	return 0;
}

/** Hand a workspace of any size back as one panel.
 *  `closeSplit` asserts its way down to one in a single step, so it cannot unwind three. */
async function restoreSinglePanel(page: Page): Promise<void> {
	while ((await panels(page).count()) > 1) {
		await hdr(page, 1).getByRole('button', { name: 'Close panel' }).click();
		await page.waitForTimeout(100);
	}
	await expect(panels(page), 'the workspace is back to one panel').toHaveCount(1);
}

test('a wide panel keeps all three actions in its header, with no ⋯ beside them', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);

	expect(await inHeader(page), 'every action is in the header').toEqual(ACTIONS);
	await expect(
		hdr(page).getByTestId('panel-overflow'),
		'a trigger onto an empty menu is a door into an empty room'
	).toBeHidden();
});

test('a narrow panel gives its actions up, lowest priority first', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	await splitRight(page); // two panels: each header is half the window wide

	try {
		// The precondition, stated: a half of a 1400px window is still wide enough for all three.
		// Without it the walk below is satisfiable by a header that never had them.
		expect(await inHeader(page), 'a 700px panel keeps every action').toEqual(ACTIONS);

		const left: string[] = [];
		let prev = ACTIONS;
		for (let w = 1400; w >= 320; w -= 20) {
			await widthTo(page, w);
			const now = await inHeader(page);
			for (const id of now)
				expect(prev.includes(id), `${id} came BACK into the header at ${w}px`).toBe(true);
			for (const id of prev) if (!now.includes(id)) left.push(id);
			prev = now;
		}
		expect(left, 'the header gives its actions up in the declared priority order').toEqual(ACTIONS);
	} finally {
		await widthTo(page, 1280);
		await closeSplit(page);
	}
});

test('the ✕ is in the header at every width, and never outside its own panel', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	await splitRight(page);

	try {
		for (const w of [1400, 1000, 800, 640, 520, 440, 380, 320]) {
			await widthTo(page, w);
			const close = hdr(page).getByRole('button', { name: 'Close panel' });
			await expect(close, `the ✕ is rendered at ${w}px`).toBeVisible();

			// Visible is not enough: the panel clips its overflow, so a header whose line runs long
			// pushes the ✕ out of the box the user can see while it keeps a bounding rect.
			const box = (await close.boundingBox())!;
			const panel = (await panels(page).first().boundingBox())!;
			expect(box.x, `the ✕ starts inside its panel at ${w}px`).toBeGreaterThanOrEqual(panel.x - 1);
			expect(box.x + box.width, `and ends inside it at ${w}px`).toBeLessThanOrEqual(
				panel.x + panel.width + 1
			);
		}
	} finally {
		await widthTo(page, 1280);
		await closeSplit(page);
	}
});

test('the header actions act: Split Right, Split Down and Maximize', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	expect(await inHeader(page), 'all three are inline at this width').toEqual(ACTIONS);

	// Maximize — `maximizedPanelId` lives outside `WorkspaceState`, so this provably cannot reach
	// the arrangement or the `.gfi`, and it is read back through the button's own label flip.
	await hdr(page).getByRole('button', { name: 'Maximize panel' }).click();
	await expect(hdr(page).getByRole('button', { name: 'Restore panel' })).toBeVisible();
	await hdr(page).getByRole('button', { name: 'Restore panel' }).click();
	await expect(hdr(page).getByRole('button', { name: 'Maximize panel' })).toBeVisible();

	// Split Right — a row split, so the new panel lands beside this one.
	await hdr(page).getByTestId('panel-split-row').click();
	try {
		await expect(panels(page), 'Split Right added a panel').toHaveCount(2);
		const [a, b] = await panels(page).evaluateAll((els) =>
			els.map((e) => e.getBoundingClientRect())
		);
		expect(b.left, 'and it sits beside the original, not under it').toBeGreaterThan(a.left);
	} finally {
		await closeSplit(page);
	}

	// Split Down — a column split: the new panel lands under this one.
	await hdr(page).getByTestId('panel-split-column').click();
	try {
		await expect(panels(page), 'Split Down added a panel').toHaveCount(2);
		const [a, b] = await panels(page).evaluateAll((els) =>
			els.map((e) => e.getBoundingClientRect())
		);
		expect(b.top, 'and it sits under the original, not beside it').toBeGreaterThan(a.top);
	} finally {
		await closeSplit(page);
	}
});

test('a spilled action is reachable as a row in the ⋯ menu — and only then', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	await splitRight(page);

	/** Each action's row label in the overflow menu — the SAME wording the right-click structural
	 *  menu uses, because the two are one command with two representations (D-R2). */
	const AS_ROW: Record<string, string> = {
		'panel-split-row': 'Split Right',
		'panel-split-column': 'Split Down',
		'panel-maximize': 'Maximize'
	};

	try {
		// Narrow until the header has given every action up, so all three rows are in the menu.
		expect(
			await collapseFully(page),
			'a panel narrow enough to give up all three exists in range'
		).toBeGreaterThan(0);

		await openOverflow(page);
		for (const id of ACTIONS) await expect(menuRow(page, AS_ROW[id]), AS_ROW[id]).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);

		// …and a row that is still a button in the header must NOT also be a row: two doors onto one
		// action is how the two representations drift apart.
		await widthTo(page, 1400);
		expect(await inHeader(page), 'the header took its actions back').toEqual(ACTIONS);
		await expect(hdr(page).getByTestId('panel-overflow')).toBeHidden();
	} finally {
		await widthTo(page, 1280);
		await closeSplit(page);
	}
});

test('a menu row acts: Split Down from the ⋯ menu splits the panel', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await splitRight(page);

	try {
		expect(
			await collapseFully(page),
			'a width at which every action is behind the ⋯ exists in range'
		).toBeGreaterThan(0);

		await openOverflow(page);
		await menuRow(page, 'Split Down').click();
		await expect(panels(page), 'the row really split the panel').toHaveCount(3);
		await expect(page.locator('.context-menu')).toHaveCount(0);
	} finally {
		await widthTo(page, 1280);
		await restoreSinglePanel(page);
	}
});
