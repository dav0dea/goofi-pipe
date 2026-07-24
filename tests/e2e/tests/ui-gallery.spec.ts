import { test, expect } from '@playwright/test';
import { exportedPrimitives, SAMPLES } from '../lib/uiSweep';

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

// The Field family + `useLiveValue` (Task 3) — the north-star core. Each assertion is behavioural,
// not "renders": the real <label> must focus its control, a typed NumberInput must commit on blur
// (not per keystroke), the inputmode variant must set the right keyboard, and a Field must compose a
// Slider + NumberInput onto one shared value. Runs under the `default` (fine-pointer) project.
test.describe('UI Field family', () => {
	test('renders each Field-family control sample', async ({ page }) => {
		await page.goto('/dev/ui');
		await expect(page.getByTestId('ui-slider')).toBeVisible();
		await expect(page.getByTestId('ui-select')).toBeVisible();
		await expect(page.getByTestId('ui-trigger')).toBeVisible();
		await expect(page.getByTestId('ui-toggle')).toBeVisible();
		await expect(page.getByTestId('ui-field-number')).toBeVisible();
	});

	test("clicking a Field's label focuses its control (real <label for>)", async ({ page }) => {
		await page.goto('/dev/ui');
		const field = page.getByTestId('ui-field-single');
		await field.waitFor();
		const input = page.getByTestId('ui-field-number');
		await expect(input).not.toBeFocused();
		// Click the label text (NOT the input) — the native for= linkage must forward focus.
		await field.locator('label.ui-field-label').click();
		await expect(input, 'clicking the Field label focuses the wrapped control').toBeFocused();
	});

	test('NumberInput commits on blur/Enter, not per keystroke', async ({ page }) => {
		await page.goto('/dev/ui');
		const committed = page.getByTestId('ui-field-committed');
		await expect(committed, 'seed value shown').toHaveText('1');
		const input = page.getByTestId('ui-field-number');
		await input.focus();
		await input.fill('7'); // buffers the edit + fires input — but must NOT commit yet
		await expect(committed, 'a typed value does not echo-commit per keystroke').toHaveText('1');
		await input.press('Enter'); // Enter commits (blurs)
		await expect(committed, 'Enter commits the buffered value').toHaveText('7');
	});

	test('TextInput sets the inputmode + enterkeyhint per variant', async ({ page }) => {
		await page.goto('/dev/ui');
		await expect(page.getByTestId('ui-text-text')).toHaveAttribute('inputmode', 'text');
		await expect(page.getByTestId('ui-text-decimal')).toHaveAttribute('inputmode', 'decimal');
		await expect(page.getByTestId('ui-text-search')).toHaveAttribute('inputmode', 'search');
		// `path` maps to the `url` virtual keyboard (surfaces `/` + `.`, drops space) — a distinct
		// inputmode per variant so this mapping is unambiguous.
		await expect(page.getByTestId('ui-text-path')).toHaveAttribute('inputmode', 'url');
		await expect(page.getByTestId('ui-text-search')).toHaveAttribute('enterkeyhint', 'search');
	});

	test('a Field composes a Slider + NumberInput onto one shared value', async ({ page }) => {
		await page.goto('/dev/ui');
		const value = page.getByTestId('ui-compose-value');
		const number = page.getByTestId('ui-compose-number');
		await number.focus();
		await number.fill('0.5');
		await number.press('Enter');
		// The shared state updated, and the sibling Slider (bound to the SAME value) followed.
		await expect(value, 'the shared value committed').toHaveText('0.5');
		const range = page.getByTestId('ui-compose-slider').locator('input[type=range]');
		await expect(range, 'the paired Slider tracks the same value').toHaveValue('0.5');
	});

	test('Trigger fires its action on click', async ({ page }) => {
		await page.goto('/dev/ui');
		const count = page.getByTestId('ui-trigger-count');
		await expect(count).toHaveText('0');
		await page.getByTestId('ui-trigger').click();
		await expect(count).toHaveText('1');
	});

	test('Toggle flips its value on click', async ({ page }) => {
		await page.goto('/dev/ui');
		const val = page.getByTestId('ui-toggle-value');
		await expect(val).toHaveText('off');
		await page.getByTestId('ui-toggle').click();
		await expect(val).toHaveText('on');
	});

	test('Select fires onRefresh from the ⟳ button', async ({ page }) => {
		await page.goto('/dev/ui');
		const refreshes = page.getByTestId('ui-select-refreshes');
		await expect(refreshes).toHaveText('0');
		await page.getByTestId('ui-select').getByRole('button', { name: /re-scan/i }).click();
		await expect(refreshes, 'the refresh affordance re-scans even from a live list').toHaveText('1');
	});
});

// NumberInput drag-to-scrub. The scrub interaction (the 3px click-vs-scrub threshold, the dx→value
// math, the live per-move commit, the pointer lifecycle) had ZERO pointer-driven coverage: these drive
// `ui-compose-number` with a REAL pointer sequence off the paired `ui-compose-slider`/`ui-compose-value`.
// The first two are the characterization safety net (the normal path, green on any correct build); the
// last two reproduce the pointer-lifecycle + latch bugs and go RED on the pre-fix code. Runs under the
// `default` (fine-pointer) project like the rest of this file.
test.describe('UI NumberInput drag-to-scrub', () => {
	const readValue = (page: import('@playwright/test').Page) =>
		page.getByTestId('ui-compose-value').textContent().then((t) => Number(t));

	// The scrub input's centre, so a press lands on it and a drag starts from a known origin.
	async function centre(page: import('@playwright/test').Page): Promise<{ x: number; y: number }> {
		const box = (await page.getByTestId('ui-compose-number').boundingBox())!;
		return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
	}

	// A full press → drag-by-dx → release, from the input's centre.
	async function scrub(page: import('@playwright/test').Page, dx: number): Promise<void> {
		const c = await centre(page);
		await page.mouse.move(c.x, c.y);
		await page.mouse.down();
		await page.mouse.move(c.x + dx, c.y); // past the 3px click-vs-scrub threshold
		await page.mouse.up();
	}

	test('a horizontal drag scrubs the value and the paired Slider follows', async ({ page }) => {
		await page.goto('/dev/ui');
		await expect(page.getByTestId('ui-compose-value'), 'seed value').toHaveText('0.3');
		await scrub(page, 40); // +40px right → the value climbs
		const after = await readValue(page);
		expect(after, 'the drag scrubbed the value upward').toBeGreaterThan(0.3);
		// The paired Slider (bound to the SAME cutoff) tracked the scrub.
		const range = page.getByTestId('ui-compose-slider').locator('input[type=range]');
		expect(Number(await range.inputValue()), 'the paired Slider followed the scrub').toBeCloseTo(after, 5);
	});

	test('a bare click (no drag) focuses to type and does NOT commit', async ({ page }) => {
		await page.goto('/dev/ui');
		const number = page.getByTestId('ui-compose-number');
		const c = await centre(page);
		await page.mouse.move(c.x, c.y);
		await page.mouse.down();
		await page.mouse.up(); // no movement → a click, not a scrub
		await expect(number, 'a click focuses the input to type').toBeFocused();
		await expect(page.getByTestId('ui-compose-value'), 'a click commits nothing').toHaveText('0.3');
	});

	// Finding 2: a cancelled gesture (a touch scroll reclaims the pointer → the browser fires
	// `pointercancel`) must tear the scrub down. Pre-fix there is no cancel handler and the listeners
	// live on `window`, so the scrub keeps committing under the cancelled pointer — the value tracks the
	// cursor after the cancel. RED here; GREEN once pointer capture + the cancel teardown land.
	test('a pointercancel tears the scrub down (no tracking under a cancelled pointer)', async ({ page }) => {
		await page.goto('/dev/ui');
		const number = page.getByTestId('ui-compose-number');
		const c = await centre(page);
		await page.mouse.move(c.x, c.y);
		await page.mouse.down();
		await page.mouse.move(c.x + 40, c.y); // latch the scrub
		const atCancel = await readValue(page);
		expect(atCancel, 'the scrub latched and moved the value').toBeGreaterThan(0.3);

		await number.dispatchEvent('pointercancel'); // abort the gesture, the way a touch-scroll steal does
		await page.mouse.move(c.x + 80, c.y); // keep moving — a torn-down scrub must NOT follow
		expect(await readValue(page), 'the value is frozen after the cancel').toBeCloseTo(atCancel, 5);
		await page.mouse.up(); // release Playwright's tracked button state

		// The control still works afterwards — the latch self-healed, it was not left stranded.
		await scrub(page, 30);
		expect(await readValue(page), 'a fresh scrub works after the cancel').toBeGreaterThan(atCancel);
	});
});

// Tabs (the connected bar) + Disclosure (Task 4). The Tabs assertions verify the REAL connected-look
// mechanism (the active tab's background equals the panel body's, and differs from an inactive tab's)
// and arrow-key navigation — not merely "renders". Disclosure toggles children visibility + aria.
// Runs under the `default` (fine-pointer) project like the rest of this file.
test.describe('UI Tabs + Disclosure', () => {
	test('the active tab drops to the body surface (the connected look)', async ({ page }) => {
		await page.goto('/dev/ui');
		const tabs = page.getByTestId('ui-tabs');
		await tabs.waitFor();
		const bg = (loc: import('@playwright/test').Locator) =>
			loc.evaluate((el) => getComputedStyle(el).backgroundColor);

		const activeTab = tabs.getByRole('tab', { selected: true });
		const inactiveTab = tabs.getByRole('tab', { selected: false }).first();
		const bodyBg = await bg(page.getByTestId('ui-tabs-body'));
		const activeBg = await bg(activeTab);
		const inactiveBg = await bg(inactiveTab);

		// Connected: the active tab paints the SAME surface as the body flush beneath it (two
		// independent elements resolving to the same token — not a tautology).
		expect(activeBg, 'active tab background equals the body surface (merged)').toBe(bodyBg);
		// And the drop actually happened — an inactive tab sits at the header surface, a different colour.
		expect(inactiveBg, 'an inactive tab sits at the header surface, not the body').not.toBe(activeBg);
	});

	test('arrow keys move the active tab (roving tablist)', async ({ page }) => {
		await page.goto('/dev/ui');
		const tabs = page.getByTestId('ui-tabs');
		await tabs.waitFor();
		const active = page.getByTestId('ui-tabs-active');
		await expect(active, 'first tab active initially').toHaveText('signal');

		// Focus the active tab, then ArrowRight advances the selection (automatic activation).
		await tabs.getByRole('tab', { selected: true }).focus();
		await page.keyboard.press('ArrowRight');
		await expect(active, 'ArrowRight selects the next tab').toHaveText('audio');
		// aria-selected followed the selection onto the newly active tab.
		await expect(tabs.getByRole('tab', { name: 'Audio' })).toHaveAttribute('aria-selected', 'true');
		// ArrowLeft from the first wraps to the last.
		await page.keyboard.press('ArrowLeft');
		await expect(active, 'ArrowLeft retreats').toHaveText('signal');
		await page.keyboard.press('ArrowLeft');
		await expect(active, 'ArrowLeft wraps past the start to the last tab').toHaveText('video');
	});

	test('Disclosure toggles its children + aria-expanded', async ({ page }) => {
		await page.goto('/dev/ui');
		const summary = page.getByTestId('ui-disclosure').getByRole('button');
		await summary.waitFor();
		const content = page.getByTestId('ui-disclosure-content');
		const toggles = page.getByTestId('ui-disclosure-toggles');

		// Collapsed by default: content is not in the DOM, aria-expanded=false.
		await expect(content, 'children hidden when closed').toBeHidden();
		await expect(summary).toHaveAttribute('aria-expanded', 'false');

		await summary.click();
		await expect(content, 'children revealed on open').toBeVisible();
		await expect(summary).toHaveAttribute('aria-expanded', 'true');
		await expect(toggles, 'onToggle fired once').toHaveText('1');

		await summary.click();
		await expect(content, 'children hidden again on close').toBeHidden();
		await expect(summary).toHaveAttribute('aria-expanded', 'false');
		await expect(toggles, 'onToggle fired twice').toHaveText('2');
	});
});

// Surfaces (Task 5): Popover, Dialog, PanelShell. Each assertion is behavioural — the Popover really
// dismisses on Escape + outside-click and its box is really clamped on-screen; the Dialog really
// traps focus and closes on Escape + backdrop; the PanelShell really renders exactly ONE header row
// composing title+toolbar+actions. Runs under the `default` (fine-pointer) project.
test.describe('UI surfaces', () => {
	test('Popover opens from its trigger and shows its content', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-popover-content');
		await expect(content, 'closed by default (not in the DOM)').toBeHidden();
		await page.getByTestId('ui-popover-trigger').click();
		await expect(content, 'clicking the trigger opens the popover').toBeVisible();
	});

	test('Popover dismisses on Escape', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-popover-content');
		await page.getByTestId('ui-popover-trigger').click();
		await expect(content).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(content, 'Escape dismisses the popover').toBeHidden();
	});

	test('Popover dismisses on an outside pointerdown', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-popover-content');
		await page.getByTestId('ui-popover-trigger').click();
		await expect(content).toBeVisible();
		// Click the top-left of the viewport — neither the popover surface nor its anchor.
		await page.mouse.click(10, 10);
		await expect(content, 'an outside click dismisses the popover').toBeHidden();
	});

	test("Popover's box is clamped on-screen even when its trigger hugs the right edge", async ({
		page
	}) => {
		await page.goto('/dev/ui');
		await page.getByTestId('ui-popover-trigger').click();
		const pop = page.getByTestId('ui-popover');
		await expect(pop).toBeVisible();
		const box = (await pop.boundingBox())!;
		const vp = await page.evaluate(() => ({ w: window.innerWidth, h: window.innerHeight }));
		// The whole popover box sits within the viewport — the clamp shifted it back from the edge.
		expect(box.x, 'left edge on-screen').toBeGreaterThanOrEqual(0);
		expect(box.y, 'top edge on-screen').toBeGreaterThanOrEqual(0);
		expect(box.x + box.width, 'right edge within the viewport').toBeLessThanOrEqual(vp.w + 1);
		expect(box.y + box.height, 'bottom edge within the viewport').toBeLessThanOrEqual(vp.h + 1);
	});

	test('Dialog opens and traps focus inside itself', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-dialog-content');
		await expect(content, 'closed by default').toBeHidden();
		await page.getByTestId('ui-dialog-trigger').click();
		await expect(content, 'clicking the trigger opens the dialog').toBeVisible();
		// showModal() moves focus into the dialog (its first focusable descendant).
		const focusInside = await page.evaluate(() => {
			const d = document.querySelector('[data-testid="ui-dialog"]');
			const a = document.activeElement;
			return !!d && !!a && (d === a || d.contains(a));
		});
		expect(focusInside, 'focus is inside the dialog after open').toBe(true);
	});

	test('Dialog closes on Escape', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-dialog-content');
		await page.getByTestId('ui-dialog-trigger').click();
		await expect(content).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(content, 'Escape closes the dialog').toBeHidden();
	});

	test('Dialog closes on a backdrop click', async ({ page }) => {
		await page.goto('/dev/ui');
		const content = page.getByTestId('ui-dialog-content');
		await page.getByTestId('ui-dialog-trigger').click();
		await expect(content).toBeVisible();
		// The modal dialog is centered; a click near the corner lands on the backdrop (target === the
		// dialog element), which routes to onClose.
		await page.mouse.click(8, 8);
		await expect(content, 'a backdrop click closes the dialog').toBeHidden();
	});

	test('PanelShell renders exactly ONE header row composing title + toolbar + actions', async ({
		page
	}) => {
		await page.goto('/dev/ui');
		const shell = page.getByTestId('ui-panelshell');
		await shell.waitFor();
		const header = shell.locator('.ui-panel-shell-header');
		await expect(header, 'exactly one header row').toHaveCount(1);
		await expect(header, 'the header carries the title').toContainText('Parameters');
		await expect(
			header.getByTestId('ui-panelshell-toolbar'),
			'the toolbar group lives in the header'
		).toBeVisible();
		await expect(
			header.getByTestId('ui-panelshell-actions'),
			'the actions group lives in the header'
		).toBeVisible();
		// The body composes a ScrollArea that actually scrolls its overflowing rows.
		const body = page.getByTestId('ui-panelshell-body');
		const metrics = await body.evaluate((el) => ({ scroll: el.scrollHeight, client: el.clientHeight }));
		expect(metrics.scroll, 'the shell body overflows and scrolls').toBeGreaterThan(metrics.client);
	});
});

// Display primitives (Task 6): Badge, Chip, StatusDot, Spinner, EmptyState. Each assertion is
// behavioural, not "renders": tones resolve to distinct colours, the pressable Chip fires its click,
// the StatusDot carries NO glow (the named health-dot regression guard), and the EmptyState centres
// on both axes. Runs under the `default` (fine-pointer) project like the rest of this file.
test.describe('UI display primitives', () => {
	const cssColor = (loc: import('@playwright/test').Locator, prop: 'color' | 'backgroundColor') =>
		loc.evaluate((el, p) => getComputedStyle(el)[p], prop);

	test('renders every Badge tone, and the meaningful tones are visually distinct', async ({ page }) => {
		await page.goto('/dev/ui');
		for (const tone of ['neutral', 'accent', 'success', 'warning', 'danger']) {
			await expect(page.getByTestId(`ui-badge-${tone}`)).toBeVisible();
		}
		// success shares the accent token by design, so it is excluded — the four semantically-distinct
		// tones must resolve to four distinct text colours (not all defaulting to one).
		const colors = await Promise.all(
			['neutral', 'accent', 'warning', 'danger'].map((t) => cssColor(page.getByTestId(`ui-badge-${t}`), 'color'))
		);
		expect(new Set(colors).size, 'neutral/accent/warning/danger badges are visually distinct').toBe(4);
	});

	test('renders every Chip tone', async ({ page }) => {
		await page.goto('/dev/ui');
		for (const tone of ['neutral', 'accent', 'success', 'warning', 'danger']) {
			await expect(page.getByTestId(`ui-chip-${tone}`)).toBeVisible();
		}
	});

	test('Chip is a real <button> and fires its onclick', async ({ page }) => {
		await page.goto('/dev/ui');
		const chip = page.getByTestId('ui-chip');
		await expect(chip).toHaveJSProperty('tagName', 'BUTTON');
		const count = page.getByTestId('ui-chip-count');
		await expect(count).toHaveText('0');
		await chip.click();
		await expect(count, 'clicking the Chip fired its onclick').toHaveText('1');
	});

	test('StatusDot has NO glow (box-shadow none) — the health-dot regression guard', async ({ page }) => {
		await page.goto('/dev/ui');
		for (const tone of ['ok', 'error', 'warn']) {
			const dot = page.getByTestId(`ui-statusdot-${tone}`);
			await expect(dot).toBeVisible();
			const shadow = await dot.evaluate((el) => getComputedStyle(el).boxShadow);
			expect(shadow, `StatusDot ${tone} must carry no glow`).toBe('none');
		}
		// ok/error/warn are semantically different states → three distinct fill colours.
		const fills = await Promise.all(
			['ok', 'error', 'warn'].map((t) => cssColor(page.getByTestId(`ui-statusdot-${t}`), 'backgroundColor'))
		);
		expect(new Set(fills).size, 'ok/error/warn dots are distinct colours').toBe(3);
	});

	test('renders the Spinner in each size', async ({ page }) => {
		await page.goto('/dev/ui');
		await expect(page.getByTestId('ui-spinner-sm')).toBeVisible();
		await expect(page.getByTestId('ui-spinner-md')).toBeVisible();
	});

	test('EmptyState centres its content on both axes and renders bare without snippets', async ({ page }) => {
		await page.goto('/dev/ui');
		const empty = page.getByTestId('ui-emptystate');
		await expect(empty).toBeVisible();
		const layout = await empty.evaluate((el) => {
			const s = getComputedStyle(el);
			return { display: s.display, direction: s.flexDirection, align: s.alignItems, justify: s.justifyContent };
		});
		expect(layout.display, 'a flex column').toBe('flex');
		expect(layout.direction, 'stacked vertically').toBe('column');
		expect(layout.align, 'cross-axis centred').toBe('center');
		expect(layout.justify, 'main-axis centred').toBe('center');
		// The content actually sits centred within the frame (not hugging the left edge).
		const frameBox = (await empty.boundingBox())!;
		const titleBox = (await page.getByTestId('ui-emptystate').getByText('No nodes yet').boundingBox())!;
		const titleMid = titleBox.x + titleBox.width / 2;
		const frameMid = frameBox.x + frameBox.width / 2;
		expect(Math.abs(titleMid - frameMid), 'the title is horizontally centred in the frame').toBeLessThan(2);
		// A bare EmptyState (no icon/title/hint snippets) still renders as a valid centred box.
		await expect(page.getByTestId('ui-emptystate-bare')).toBeVisible();
	});
});

// The `@container` enablement (Task 7). A Field's control row must stack to a single column when its
// query container is narrower than the threshold and stay a row when wide — the FIRST consumer of the
// `container-type: inline-size` the panel body now establishes. The assertion is geometric AND on the
// computed flex-direction, so it fails hard before the Field `@container` rule exists. Runs under the
// `default` (fine-pointer) project like the rest of this file.
test.describe('UI @container responsiveness', () => {
	const controlDirection = (field: import('@playwright/test').Locator) =>
		field.locator('.ui-field-control').evaluate((el) => getComputedStyle(el).flexDirection);

	test('a Field control row stacks to one column in a narrow container and stays a row when wide', async ({
		page
	}) => {
		await page.goto('/dev/ui');
		const narrow = page.getByTestId('ui-cq-narrow-field');
		const wide = page.getByTestId('ui-cq-wide-field');
		await narrow.waitFor();

		// The control row's computed flex-direction responds to its query container's width.
		expect(await controlDirection(narrow), 'narrow container stacks the controls').toBe('column');
		expect(await controlDirection(wide), 'wide container keeps the controls in a row').toBe('row');

		// And the geometry follows (a real layout assertion, not just the computed property): in the
		// narrow container the NumberInput sits BELOW the Slider; in the wide one they sit side by side.
		const nSlider = (await page.getByTestId('ui-cq-narrow-slider').boundingBox())!;
		const nNumber = (await page.getByTestId('ui-cq-narrow-number').boundingBox())!;
		expect(nNumber.y, 'narrow: the NumberInput is below the Slider (stacked)').toBeGreaterThanOrEqual(
			nSlider.y + nSlider.height - 1
		);

		const wSlider = (await page.getByTestId('ui-cq-wide-slider').boundingBox())!;
		const wNumber = (await page.getByTestId('ui-cq-wide-number').boundingBox())!;
		expect(wNumber.x, 'wide: the NumberInput is right of the Slider (a row)').toBeGreaterThan(
			wSlider.x + wSlider.width - 1
		);
		expect(wNumber.y, 'wide: they share the same row (vertical overlap)').toBeLessThan(
			wSlider.y + wSlider.height
		);
	});
});

// The final whole-library sweep (Task 8). Earlier describes assert each primitive's BEHAVIOUR; this
// one asserts the library as a SURFACE: the sample registry is pinned to `$lib/ui`'s export barrel,
// every exported primitive has a visible gallery sample, and a keyboard-focused interactive control
// rings. Its point is the enumeration — a future `export { default as Foo }` with no gallery entry
// fails HERE, so the sweep can never silently skip a new primitive. The coarse touch-target roll-up
// lives in `touch-ui-gallery.spec.ts` (only that file runs under the `touch` project). Default project.
test.describe('UI library sweep (the whole export surface)', () => {
	test('the sample registry covers exactly the exported primitive surface', () => {
		// The SSOT is index.ts. If a primitive is exported without a registry entry (hence no gallery
		// sample), or a stale entry points at a deleted primitive, these two sets diverge and this fails
		// RED — the guard that keeps the render/touch roll-ups below honest about the full surface.
		expect(Object.keys(SAMPLES).sort(), 'the registry mirrors the export barrel, 1:1').toEqual(
			exportedPrimitives()
		);
	});

	test('every exported primitive renders a representative sample in the gallery', async ({ page }) => {
		await page.goto('/dev/ui');
		for (const [name, sample] of Object.entries(SAMPLES)) {
			await expect(page.getByTestId(sample.testid), `${name} has a visible gallery sample`).toBeVisible();
		}
	});

	test('a keyboard-focused interactive primitive shows the app accent focus ring', async ({ page }) => {
		await page.goto('/dev/ui');
		await page.getByTestId(SAMPLES.Button.testid).waitFor(); // the page has rendered
		// A real keyboard Tab (not programmatic focus) so :focus-visible engages; the first focusable
		// element on this static page is a gallery interactive-control sample. The sweep asserts the
		// invariant "an interactive primitive rings" against whatever control Tab reaches, rather than
		// re-pinning one blessed control.
		await page.keyboard.press('Tab');
		const ring = await page.evaluate(() => {
			const el = document.activeElement as HTMLElement | null;
			if (!el || el === document.body) return { tag: '', testid: '', outlineWidth: '', outlineColor: '' };
			const s = getComputedStyle(el);
			return {
				tag: el.tagName,
				testid: el.closest('[data-testid]')?.getAttribute('data-testid') ?? '',
				outlineWidth: s.outlineWidth,
				outlineColor: s.outlineColor
			};
		});
		expect(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'], 'Tab lands on a real interactive control').toContain(
			ring.tag
		);
		expect(ring.testid, 'the focused control is a gallery UI-primitive sample').toMatch(/^ui-/);
		// The app rule specifically (2px solid --accent = #50d0a0), not merely "some outline", so a
		// future UA-default outline could not pass this as a tautology.
		expect(ring.outlineWidth, 'the app :focus-visible ring is 2px').toBe('2px');
		expect(ring.outlineColor, 'the ring colour is --accent (#50d0a0)').toBe('rgb(80, 208, 160)');
	});
});
