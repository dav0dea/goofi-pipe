import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { closeSplit, resetPatch, splitRight, waitForApp } from '../lib/app';
import { dismiss, spawnSh, stateOf } from '../lib/harness';

/**
 * The agent panel, driven end to end against a real PTY.
 *
 * **Nothing here needs an agent harness installed.** The harness spawned is `_sh` — the hidden
 * test adapter Task 1 registered, a plain `/bin/sh`, the same `_`-prefixed idiom the node catalog
 * uses — so the terminal, the transcript and the whole detach/kill lifecycle are driven by a
 * binary every machine has. It is spawned over a raw `/control` socket because the panel's
 * launcher deliberately lists only DETECTED harnesses (`_`-prefixed ones are hidden from it), and
 * a client attaching to an instance it did not spawn is the production path a second tab takes.
 *
 * The markers are spelled `goofi''markN`, which the shell's own echo shows verbatim and only the
 * CHILD prints joined — so no assertion below can pass on the line discipline repeating the input.
 */

/** Make the last panel an agent panel and wait for it to attach. A fresh agent panel claims a
 * running instance no other panel is showing — which is what makes closing and reopening one a
 * re-attach rather than a relaunch. */
async function openAgentPanel(page: Page): Promise<void> {
	const panels = await page.evaluate(() => (window as any).goofi.query.panels());
	await page.evaluate(
		(id) => (window as any).goofi.commands.setPanelType(id, 'agent'),
		panels[panels.length - 1].panelId
	);
	await expect(page.getByTestId('agent-terminal')).toBeVisible();
}

/** Type a line into the terminal and wait for the CHILD's answer to it. */
async function say(page: Page, marker: string): Promise<void> {
	await page.getByTestId('agent-terminal').click();
	await page.keyboard.type(`echo goofi''${marker}`);
	await page.keyboard.press('Enter');
	await expect(page.getByTestId('agent-terminal')).toContainText(`goofi${marker}`, {
		timeout: 15_000
	});
}

/** Hand the instance back, and wait for the CLIENT to agree: a panel still bound to it would keep
 * a ✕ that asks instead of closing, which is how this raced the first time. */
async function sweep(page: Page, id: string): Promise<void> {
	await dismiss(page, id);
	await expect(page.getByTestId('agent-launcher')).toBeVisible();
}

/** Hand the shared backend back, from a `finally`. Its OWN failure must not replace the failure
 * that sent us here — masking one cost two rounds of debugging the wrong thing — but a run whose
 * body passed still fails on a leak, which is what the contract in `lib/app.ts` is for. */
async function handBack(page: Page, id: string, threw: boolean): Promise<void> {
	try {
		if (id) await sweep(page, id);
		if ((await page.locator('.panel').count()) > 1) await closeSplit(page);
		await resetPatch(page);
	} catch (e) {
		if (!threw) throw e;
		console.error('agent-panel: cleanup after a failed body', e);
	}
}

const unsaved = (page: Page): Promise<boolean> =>
	page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges);

let scratch = '';
test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-agent-')));
});
test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

test('a harness runs in a panel, and its transcript survives closing that panel', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	let id = '';
	let threw = true;
	try {
		await splitRight(page);
		await openAgentPanelLater(page);

		// Named, and therefore clean — so everything after this point that does NOT move the dot is
		// proven to be viewpoint rather than authoring.
		await page.evaluate(
			(p) => (window as any).goofi.commands.save(p),
			path.join(scratch, 'agent.gfi')
		);
		await expect.poll(() => unsaved(page)).toBe(false);

		id = await spawnSh(page);
		await expect(page.getByTestId('agent-terminal')).toBeVisible();
		await say(page, 'mark42');

		// Attaching a panel to an instance, and the instance itself, are both viewpoint: the choice
		// never leaves this client, and the harness config lands beside the workspace, not in it.
		expect(await unsaved(page), 'attaching an agent dirtied the patch').toBe(false);
		await expect(page.getByTestId('topbar-agents')).toContainText('1');

		// Close the VIEW: the ✕ asks rather than killing, and Detach leaves the agent running.
		await page.getByTestId('panel-header').nth(1).getByRole('button', { name: 'Close panel' }).click();
		await expect(page.getByTestId('agent-close-dialog')).toBeVisible();
		await page.getByTestId('agent-detach').click();
		await expect(page.locator('.panel')).toHaveCount(1);
		expect(await stateOf(page, id), 'a detach killed the harness').toBe('running');

		// Name it again: closing that panel really WAS authoring (the layout changed), so the dot it
		// left is correct and has to be cleared before the next question can be asked of it.
		await page.evaluate(
			(p) => (window as any).goofi.commands.save(p),
			path.join(scratch, 'agent.gfi')
		);
		await expect.poll(() => unsaved(page)).toBe(false);

		// The badge is the door onto an agent from outside its panel — including now, with NO agent
		// panel open. That is where it used to open a TAB to hold the question: `add_tab` +
		// `set_panel`, an unsaved dot and two undo steps for asking. Asking is not authoring.
		await page.getByTestId('topbar-agents').click();
		await page.locator('.context-menu .item').first().click();
		await expect(page.getByTestId('agent-close-dialog')).toBeVisible();
		expect(await unsaved(page), 'asking about an agent dirtied the patch').toBe(false);
		await expect(page.getByTestId('workspace-tabs').locator('.pt-chip')).toHaveCount(1);
		await page.keyboard.press('Escape');
		await expect(page.getByTestId('agent-close-dialog')).toBeHidden();

		// …and reopening finds the transcript where it was. Nothing was replayed: the manager keeps
		// no grid, so this can only be the Terminal object that outlived the panel.
		await splitRight(page);
		await openAgentPanel(page);
		await expect(page.getByTestId('agent-terminal')).toContainText('goofimark42');

		// The TopBar chip is the door from outside the panel: it lists what is running, and choosing
		// one raises the same question — this time answered with Kill.
		await page.getByTestId('topbar-agents').click();
		await page.locator('.context-menu .item').first().click();
		await expect(page.getByTestId('agent-close-dialog')).toBeVisible();
		await page.getByTestId('agent-kill').click();
		// `gone`, not `exited`: a dead harness is dropped rather than kept, and the app asks the
		// manager to drop it as soon as it sees the exit.
		await expect.poll(() => stateOf(page, id), { timeout: 15_000 }).toMatch(/exited|gone/);
		threw = false;
	} finally {
		await handBack(page, id, threw);
	}
});

/** As `openAgentPanel`, but before any instance exists: the panel shows its launcher instead. */
async function openAgentPanelLater(page: Page): Promise<void> {
	const panels = await page.evaluate(() => (window as any).goofi.query.panels());
	await page.evaluate(
		(id) => (window as any).goofi.commands.setPanelType(id, 'agent'),
		panels[panels.length - 1].panelId
	);
	await expect(page.getByTestId('agent-launcher')).toBeVisible();
}

/** The terminal's own grid box, and the box the panel gives it — both in CSS pixels, the host's
 *  measured as a CONTENT box so its padding is not counted as unused screen. */
const boxes = (page: Page): Promise<{ grid: [number, number]; host: [number, number] }> =>
	page.evaluate(() => {
		const host = document.querySelector('[data-testid="agent-terminal"]') as HTMLElement;
		const hs = getComputedStyle(host);
		const screen = host.querySelector('.xterm-screen') as HTMLElement;
		return {
			grid: [screen.getBoundingClientRect().width, screen.getBoundingClientRect().height],
			host: [
				parseFloat(hs.width) - parseFloat(hs.paddingLeft) - parseFloat(hs.paddingRight),
				parseFloat(hs.height) - parseFloat(hs.paddingTop) - parseFloat(hs.paddingBottom)
			]
		};
	});

test('a freshly launched harness is sized to its panel, with no manual resize', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	let id = '';
	let threw = true;
	try {
		await splitRight(page);
		await openAgentPanelLater(page);
		id = await spawnSh(page);
		await expect(page.getByTestId('agent-terminal')).toBeVisible();

		// What the CHILD was told — the only reading that says whether the size reached the PTY, and
		// the thing a full-screen TUI lays itself out against.
		await page.getByTestId('agent-terminal').click();
		await page.keyboard.type('stty size');
		await page.keyboard.press('Enter');
		// `innerText`, not `toContainText`: the assertion normalizes whitespace away and the answer
		// is a LINE of its own — the only thing that tells it apart from the command that asked.
		const answer = async (): Promise<string> =>
			(await page.getByTestId('agent-terminal').innerText())
				.split('\n')
				.map((l) => l.trim())
				.filter((l) => /^\d+ \d+$/.test(l))
				.pop() ?? '';
		await expect.poll(answer, { timeout: 15_000 }).toMatch(/^\d+ \d+$/);
		const [rows, cols] = (await answer()).split(' ').map(Number);

		// The child's grid IS the drawn grid (one cell each way), so the cell size follows from it —
		// no font arithmetic, and nothing to drift when the face changes.
		const { grid, host } = await boxes(page);
		const cell = [grid[0] / cols, grid[1] / rows];
		// …and that grid fills the panel: what is left over is less than one more column (plus
		// xterm's own scrollbar) and less than one more row. A launch whose proposal never landed
		// leaves the child at the PTY's 80×24 default, which in this panel is ~65px of dead space.
		expect(host[0] - grid[0], `left ${host[0] - grid[0]}px unused across a ${cols}-col grid`)
			.toBeLessThan(cell[0] + 20);
		expect(host[1] - grid[1], `left ${host[1] - grid[1]}px unused down a ${rows}-row grid`)
			.toBeLessThan(cell[1] + 1);
		threw = false;
	} finally {
		await handBack(page, id, threw);
	}
});

test('a harness that dies hands its panel back to the launcher, and says why', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	let id = '';
	let threw = true;
	try {
		await splitRight(page);
		await openAgentPanelLater(page);
		id = await spawnSh(page);
		await expect(page.getByTestId('agent-terminal')).toBeVisible();

		// A crash: nobody asked this child to stop and it did not end well.
		await page.getByTestId('agent-terminal').click();
		await page.keyboard.type('exit 7');
		await page.keyboard.press('Enter');

		// No frozen last screen and no dismiss button to press — the panel offers a new harness.
		await expect(page.getByTestId('agent-launcher')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByTestId('toast')).toContainText('exited unexpectedly');
		// …and nothing dead is left behind on the manager's roster either.
		await expect.poll(() => stateOf(page, id), { timeout: 15_000 }).toBe('gone');
		id = '';

		// A CLEAN exit is the ordinary end of a shell session, and gets no alarm at all.
		await page.getByTestId('toast').click();
		id = await spawnSh(page);
		await expect(page.getByTestId('agent-terminal')).toBeVisible();
		await page.getByTestId('agent-terminal').click();
		await page.keyboard.type('exit');
		await page.keyboard.press('Enter');
		await expect(page.getByTestId('agent-launcher')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByTestId('toast'), 'a clean exit is not an alarm').toBeHidden();
		id = '';
		threw = false;
	} finally {
		await handBack(page, id, threw);
	}
});

test('a second view of one harness is live, and both views see the same stream', async ({
	page,
	context
}) => {
	await page.goto('/');
	await waitForApp(page);
	let id = '';
	let threw = true;
	const second = await context.newPage();
	try {
		await splitRight(page);
		id = await spawnSh(page);
		await openAgentPanel(page);
		await say(page, 'first');

		// A second tab attaches to the SAME instance. Its terminal starts empty — history is
		// deliberately per-tab (there is no server-side scrollback) — but it is live…
		await second.goto('/');
		await second.waitForFunction(() => (window as any).goofi?.query.nodeTypes()?.length > 0);
		await expect(second.getByTestId('agent-terminal')).toBeVisible();
		await say(second, 'second');

		// …and what it typed reaches the first view too, because one PTY has one stream and the
		// manager fans it out rather than owning a screen per viewer.
		await expect(page.getByTestId('agent-terminal')).toContainText('goofisecond');
		threw = false;
	} finally {
		await second.close();
		await handBack(page, id, threw);
	}
});
