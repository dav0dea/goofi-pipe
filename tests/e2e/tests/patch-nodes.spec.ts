import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { waitForApp } from '../lib/app';

/**
 * A patch's own `nodes/` directory, end to end against the REAL discovery seam — the probe, the
 * tier routing and the registry the shipped `nodes/` tree goes through, not a stub. That is the
 * whole claim of W1 ("loaded and managed in the exact same way, using the same code"), and nothing
 * below the e2e can prove it: every Rust test injects a stub scan precisely so it needs no
 * interpreter.
 *
 * The file is written straight into the live workspace mount, which is what an agent in the harness
 * will do. Its path is not derivable — a per-run temp directory under a random nonce — so the
 * façade's `openWorkspace` is the only way in, and that is the same door the agent gets.
 *
 * Hermeticity: the node file is removed and a final rescan re-derives the registry, so the shared
 * backend hands back the palette it started with.
 */

const SOURCE = `import goofi


class E2ePatchNode(goofi.Node):
    """A node that came with the patch."""

    def config_output_slots(self):
        return {"out": goofi.DataType.ARRAY}

    def process(self):
        return {"out": ([1.0], {})}
`;

/** The palette row for `type`, as the add menu reads it. */
function paletteRow(page: Page, type: string): Promise<{ type: string; source: string } | null> {
	return page.evaluate((t) => {
		const types = (window as any).goofi.query.nodeTypes() ?? [];
		return types.find((x: { type: string }) => x.type === t) ?? null;
	}, type);
}

function rescan(page: Page): Promise<{ added: string[]; changed: string[]; removed: string[] }> {
	return page.evaluate(() => (window as any).goofi.commands.rescanNodes());
}

test('a node file written into the patch workspace joins the palette, marked as the patch’s own', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	const workspace: string = await page.evaluate(() =>
		(window as any).goofi.commands.openWorkspace()
	);
	const dir = path.join(workspace, 'nodes');
	const file = path.join(dir, 'e2e_patch_node.py');
	fs.mkdirSync(dir, { recursive: true });

	try {
		expect(await paletteRow(page, 'E2ePatchNode'), 'not there before it is written').toBeNull();

		fs.writeFileSync(file, SOURCE);
		const diff = await rescan(page);
		// ONLY the new file — the shipped `nodes/` tree was indexed by the boot scan, which runs
		// this very function, so a refresh reports what changed rather than re-announcing goofi's
		// own nodes as new.
		expect(diff.added, 'the rescan reports exactly what it found').toEqual(['E2ePatchNode']);

		// The catalog reaches this tab through the `node_types` broadcast, not a reply, so wait for
		// the store rather than asserting on the round trip that produced it.
		await page.waitForFunction(
			() =>
				((window as any).goofi.query.nodeTypes() ?? []).some(
					(t: { type: string }) => t.type === 'E2ePatchNode'
				),
			undefined,
			{ timeout: 20_000 }
		);
		const row = await paletteRow(page, 'E2ePatchNode');
		expect(row?.source, 'and the palette says where it came from').toBe('patch');

		// A shipped node is the control: same palette, other provenance.
		expect((await paletteRow(page, 'Oscillator'))?.source).toBe('builtin');

		// It is a real type, not merely a row: it instantiates through the ordinary add path.
		const uid: string = await page.evaluate(() =>
			(window as any).goofi.commands.addNode('E2ePatchNode', 'python', [0, 0])
		);
		expect(uid).toBeTruthy();
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	} finally {
		fs.rmSync(file, { force: true });
		const gone = await rescan(page);
		expect(gone.removed, 'the palette is handed back as it was found').toContain('E2ePatchNode');
	}
});
