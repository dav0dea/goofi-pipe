import { test, expect } from '@playwright/test';

// The /dev/inspector gallery is a static, backend-free showcase of `<ParamField>` (spec §2, N-Task 2):
// one field per non-expression control kind against synthetic fx-OFF descriptors, each with a
// committed-value read-out. Like the /dev/ui gallery it needs no window.goofi readiness gate — it waits
// on the rendered samples. This spec is the "failing test first": it fails before the route +
// ParamField exist. Runs under the `default` (fine-pointer) project.
test.describe('Inspector field gallery', () => {
	// The D-N2 regression guard (the Critical the spec review caught): vmin/vmax are SOFT bounds — the
	// engine does not clamp on set — so the NumberInput must stay UNBOUNDED. Typing 5 into a [0,1] float
	// commits 5, NOT 1; and the Slider's track auto-extends to span the out-of-range live value.
	test('numeric commits an out-of-[vmin,vmax] value UNCLAMPED and the slider track auto-extends (D-N2)', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-float');
		const number = field.getByTestId('param-number');
		const range = field.getByTestId('param-slider').locator('input[type=range]');
		const readout = page.getByTestId('inspector-float-value');
		await expect(readout, 'seeded 0.3').toHaveText('0.3');
		await expect(range, 'the track is seeded at its soft upper bound').toHaveAttribute('max', '1');
		// Type 5 — well outside the soft [0,1] — and commit it (blur via Enter).
		await number.fill('5');
		await number.press('Enter');
		await expect(readout, 'the out-of-bounds value commits UNCLAMPED (not clipped to vmax)').toHaveText('5');
		await expect(range, 'the slider max auto-extends to span the live value').toHaveAttribute('max', '5');
	});

	test('numeric (int) commits a typed integer', async ({ page }) => {
		await page.goto('/dev/inspector');
		const number = page.getByTestId('inspector-int').getByTestId('param-number');
		const readout = page.getByTestId('inspector-int-value');
		await expect(readout).toHaveText('5');
		await number.fill('9');
		await number.press('Enter');
		await expect(readout, 'the typed integer commits').toHaveText('9');
	});

	test('toggle flips and commits the boolean', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-bool-value');
		await expect(readout).toHaveText('false');
		await page.getByTestId('inspector-bool').getByTestId('param-toggle').click();
		await expect(readout, 'the flipped value commits').toHaveText('true');
	});

	test('trigger commits true on click', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-trigger-value');
		await expect(readout, 'no fire yet').toHaveText('0');
		await page.getByTestId('inspector-trigger').getByTestId('param-trigger').click();
		await expect(readout, 'the trigger commits true (the read-out counts a fire)').toHaveText('1');
	});

	test('text commits on Enter (blur), not per keystroke', async ({ page }) => {
		await page.goto('/dev/inspector');
		const input = page.getByTestId('inspector-text').getByTestId('param-text');
		const readout = page.getByTestId('inspector-text-value');
		await expect(readout).toHaveText('hello');
		await input.fill('world');
		// The buffered value has NOT committed while typing — commit-on-blur is inherited from the P control.
		await expect(readout, 'typing buffers; it has not committed yet').toHaveText('hello');
		await input.press('Enter');
		await expect(readout, 'Enter blurs and commits the string').toHaveText('world');
	});

	test('select commits the chosen option through onChange', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-options-value');
		await expect(readout, 'seed value shown').toHaveText('sine');
		await page.getByTestId('inspector-options').getByTestId('param-select').locator('select').selectOption('square');
		await expect(readout, 'choosing an option commits it').toHaveText('square');
	});

	// The refreshable select: a stale-but-live current value absent from the options stays selected (P
	// Select prepends it), and the ⟳ button (param-refresh) fires onRefresh, which re-scans the options.
	test('a refreshable select keeps a stale value selected and ⟳ re-scans the options', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-device');
		const select = field.getByTestId('param-select').locator('select');
		const refreshes = page.getByTestId('inspector-device-refreshes');
		// 'mic-1' is not among the seeded options, so its presence as the selection proves the prepend.
		await expect(select, 'the stale-but-live value is still selected').toHaveValue('mic-1');
		await expect(select.locator('option').first(), 'it is prepended as the first option').toHaveText('mic-1');
		await expect(refreshes).toHaveText('0');
		await field.getByTestId('param-refresh').click();
		await expect(refreshes, 'the ⟳ button fired onRefresh').toHaveText('1');
		// The re-scan lands a fresh option the seed list did not have.
		await expect(select.locator('option'), 'the option list was refreshed').toContainText(['scanned-1']);
	});

	test('unknown renders a read-only JSON display of the value', async ({ page }) => {
		await page.goto('/dev/inspector');
		const display = page.getByTestId('inspector-unknown').getByTestId('param-unknown');
		// The raw JSON of the synthetic descriptor value — a read-only fallback, no editable control.
		await expect(display).toContainText('"channels"');
		await expect(display).toContainText('Fz');
	});
});
