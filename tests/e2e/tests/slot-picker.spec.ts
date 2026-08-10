import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The product-layer guard for M's one known user-visible regression: a sub-patch's renamed port
 * names vanished from the Viewer and Metadata slot pickers, because the `Select` migration dropped
 * the option-text/option-value split those two panels depend on. `5f445e0` fixed it with
 * `Select.labels` — but the guard went into the ui-gallery, which proves only that the PRIMITIVE
 * renders a labels map it is handed. Deleting `labels={…}` from either panel left the whole suite
 * green, so the exact regression could re-ship.
 *
 * The split is not cosmetic. A portal's boundary id (`out0`) is the routing key external wires and
 * every read resolve through; its NAME is a free-text label a rename changes without re-keying
 * anything. A picker that commits the label breaks the wire; one that shows the id shows the user
 * an implementation detail instead of the port they named.
 */
test('the slot pickers show a sub-patch port’s name and commit its boundary id', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);

	// A sub-patch whose one exposed output is deliberately renamed, so the label and the routing
	// key differ — the whole point. Grouping alone would not do: a fresh port's name DEFAULTS to
	// its id, which is precisely the case a broken picker still passes.
	const inst: string = await page.evaluate(
		(u) => (window as any).goofi.commands.groupNodes([u], [40, 40]),
		osc
	);
	// A collapsed instance is SYNTHESIZED per scene, so it never appears in the flat `graph().nodes`
	// list — `query.node`/`query.instance` are what resolve it.
	await page.waitForFunction((i) => (window as any).goofi.query.instance(i) !== null, inst);
	const bnd: string = await page.evaluate(
		(i) => (window as any).goofi.commands.addBoundary(i, 'out', 'ARRAY', [200, 40]),
		inst
	);
	await page.evaluate(
		([i, b, u]) => (window as any).goofi.commands.wireBoundary(i, b, u, 'out'),
		[inst, bnd, osc] as const
	);
	await page.evaluate(
		([i, b]) => (window as any).goofi.commands.renameBoundary(i, b, 'envelope'),
		[inst, bnd] as const
	);
	await page.waitForFunction(
		([i, b]) => (window as any).goofi.query.node(i)?.slot_labels?.[b] === 'envelope',
		[inst, bnd] as const
	);

	// Borrow the default editor panel as each picker's host in turn, and give it back after.
	// Order matters: switching a panel's type discards the old type's state, the binding included.
	const host = async (type: string) => {
		await page.evaluate(
			([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
			[panelId, type] as const
		);
		await page.evaluate(
			([id, u]) => (window as any).goofi.commands.bindNodeToPanel(id, u),
			[panelId, inst] as const
		);
	};

	try {
		await host('viewer');
		const viewerSlot = page.getByTestId('viewer-slot').locator('select');
		await expect(viewerSlot.locator('option')).toHaveText([`envelope · array`]);
		await viewerSlot.selectOption({ label: 'envelope · array' });
		await expect(viewerSlot, 'the viewer picker commits the routing key').toHaveValue(bnd);

		await host('metadata');
		const metaSlot = page.getByTestId('metadata-slot').locator('select');
		await expect(metaSlot.locator('option')).toHaveText(['envelope']);
		await metaSlot.selectOption({ label: 'envelope' });
		await expect(metaSlot, 'the metadata picker commits the routing key').toHaveValue(bnd);
	} finally {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		// RemoveNode captures the whole subtree, so deleting the collapsed instance takes its member
		// and stubs with it.
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), inst);
		await page
			.waitForFunction((u) => (window as any).goofi.query.node(u) === null, osc)
			.catch(() => {});
		// The restore is a command against the RUNNING PATCH, which outlives this page — and the
		// assertion above only passes once the manager's delta drew it back, so it has landed.
	}
});
