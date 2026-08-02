import { test, expect } from '@playwright/test';
import { closeSplit, splitRight, waitForApp } from '../lib/app';
import { settledBox } from '../lib/geometry';
import { addNode, waitForNode } from '../lib/goofi';
import {
	dropNode as drop,
	grip,
	openInspector as addAndSelect,
	pane,
	paneAxis,
	paneBound,
	rootRem as rem,
	settledGrabber,
	type Grabber
} from '../lib/inspector';

/**
 * The orientation-aware inspector, on the FINE pointer — because the anchor rule contains no device
 * class at all (D-I2). It asks its host panel whether it is taller than it is wide, through
 * `@container (orientation: portrait)`, and a desktop panel can answer either way.
 *
 * That is the point of this file being a `default`-project spec: everything here runs on a mouse,
 * at a landscape desktop window, and one of the cases still gets the bottom sheet. `touch-sheet` and
 * `touch-reflow` cover the phone geometries, and `touch-modality` runs the coarse half of the very
 * assertions below — off `lib/inspector.ts`, which both files read the pane through, so the mouse
 * and the finger are measured with one ruler.
 *
 * Sizes are measured against the HOST or against what the pane itself publishes, never against a
 * literal — except where the literal IS the claim (§4.3: the desktop resting width does not move
 * off 420px). In particular no test here restates the small-screen guard's PERCENTAGE: which of the
 * ceiling's two halves bound is asserted by measurement, so D-I6 can retune that number without
 * this file having an opinion about it.
 */

test('the desktop resting width does not move off 420px (§4.3)', async ({ page }) => {
	// 1600 wide, so the ceiling resolves to its REM half: the root clamp saturates at 14px anywhere
	// near this size, so 30rem is exactly 420 — the width the pane has always rested at. On a
	// narrower host the small-screen guard binds instead, which is the next case.
	await page.setViewportSize({ width: 1600, height: 900 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	expect(await rem(page), 'the root clamp is saturated at this size').toBeCloseTo(14, 5);
	const uid = await addAndSelect(page);
	try {
		const p = await settledBox(pane(page));
		expect(p.width, 'the resting pane is 30rem = 420px').toBeCloseTo(420, 0);
		// …and 30rem is genuinely the binding half here, MEASURED rather than arithmetic: a wider
		// host cannot make the pane wider. Stated this way it names no percentage, so it holds for
		// whatever the small-screen guard is set to — which is the number D-I6 leaves live.
		await page.setViewportSize({ width: 1900, height: 900 });
		const wider = await settledBox(pane(page));
		expect(wider.width, 'and a wider host cannot make it wider').toBeCloseTo(420, 0);
	} finally {
		await drop(page, uid);
	}
});

/**
 * The other half of D-I6's resting size, and the half the small-screen guard exists for: on a host
 * too narrow for the 30rem comfort cap, it is the HOST that decides. Before D-I6 the pane sat at a
 * flat 420px here and its only host clamp was `100% - --hit`, which reserved 28px of canvas out of
 * 1272.
 *
 * No fraction is restated: what is asserted is which of the two halves bound, and that the one that
 * bound still left the pane room. That second assertion is the whole of the floor/ceiling fix — a
 * host-relative size is free to resolve BELOW the floor it is measured against, which is exactly
 * what a landscape phone got (256px under a 260px floor), and stating the floor as the SAME
 * `clamp()`'s lower bound is what stops it. A guard tight enough to cross the floor on a narrow
 * desktop fails right here.
 */
test('a narrower desktop editor sizes the pane against its HOST, and still leaves it room', async ({
	page
}) => {
	// 800 wide: narrow enough that the small-screen guard is the tighter half at this root size.
	await page.setViewportSize({ width: 800, height: 720 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const p = await settledBox(pane(page));
		expect(p.width, 'the 30rem comfort cap is not what bound it — the host did').toBeLessThan(
			30 * (await rem(page))
		);
		expect(
			p.width,
			'…and the host-relative size did not land on or under the floor beside it'
		).toBeGreaterThan(await paneBound(page, 'floor'));
	} finally {
		await drop(page, uid);
	}
});

/**
 * The consequence the spec flagged for the user rather than choosing silently (§6), pinned so it
 * reads as deliberate.
 *
 * The WINDOW is landscape here — `matchMedia` says so — and the pane is a bottom sheet anyway,
 * because the panel it lives in is a narrow tall column. That is the whole argument for D-I2 over
 * `@media (orientation)`: this docked editor has exactly the phone's problem (a right-hand pane
 * would eat the canvas), and a viewport query would answer "landscape" and leave it broken. It is
 * also the one line that would be edited to take it back — see `InspectorOverlay.svelte`'s
 * `@media all` prelude.
 */
test('a narrow, tall DOCKED editor gets the sheet, in a landscape window (spec §6)', async ({
	page
}) => {
	await page.setViewportSize({ width: 1280, height: 900 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	expect(
		await page.evaluate(() => matchMedia('(orientation: landscape)').matches),
		'the WINDOW is landscape — the sheet below is the panel’s shape, not the screen’s'
	).toBe(true);
	await splitRight(page);
	let uid = '';
	try {
		// Both halves are node editors and each mounts its own inspector, so everything below is
		// scoped to the LEFT one — and its selection is addressed by panel id rather than through the
		// active editor, which the split just moved to the new panel.
		const left = page.locator('.panel').first();
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		await page.evaluate(
			([u, id]) => (window as any).goofi.commands.select([u], id),
			[uid, panelId] as const
		);
		const sheet = left.getByTestId('auto-side-panel');
		await expect(sheet, 'the left editor inspects its selection').toHaveClass(/open/);

		const host = (await left.locator('.editor-panel').boundingBox())!;
		expect(host.height, 'the split left the editor taller than it is wide').toBeGreaterThan(
			host.width
		);

		const p = await settledBox(sheet);
		expect(p.width, 'the sheet spans its host').toBeCloseTo(host.width, 0);
		expect(
			host.y + host.height - (p.y + p.height),
			'and is flush with the bottom it slid from'
		).toBeLessThan(2);
		expect(p.height, 'at the 60% D-I6 allows').toBeCloseTo(host.height * 0.6, 0);
	} finally {
		if (uid) await drop(page, uid);
		await closeSplit(page);
	}
});

/**
 * D-I3/D-I4: the edge drag is not a touch affordance. It is THE resize, identical on both inputs,
 * and this is the mouse half — driven with `page.mouse`, whose `pointerType` is `mouse`, i.e. the
 * input every coarse door in the app is closed to.
 */
test('an edge drag resizes the pane with a MOUSE, and the size outlives the reload', async ({
	page
}) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const before = await settledBox(pane(page));
		const g = (await grip(page).boundingBox())!;
		const y = g.y + g.height / 2;
		// Rightward, i.e. INTO the pane, which shrinks it — the same direction of travel that shrinks
		// the sheet when it is pushed down. Away from the pane the ceiling would bind and the
		// measurement would be of `max-width`, not of the drag.
		await page.mouse.move(g.x + g.width / 2, y);
		await page.mouse.down();
		await page.mouse.move(g.x + g.width / 2 + 60, y, { steps: 8 });
		await page.mouse.up();

		const after = await settledBox(pane(page));
		expect(after.width, 'the pane shrank by exactly the drag').toBeCloseTo(before.width - 60, 0);
		expect(
			await page.evaluate(() => localStorage.getItem('goofi.panelWidth')),
			'and the RENDERED size is what was stored, so a reload agrees with the screen'
		).toBe(String(Math.round(after.width)));

		await page.reload();
		await waitForApp(page);
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		await expect(pane(page)).toHaveClass(/open/);
		const restored = await settledBox(pane(page));
		expect(restored.width, 'and it comes back at that width').toBeCloseTo(after.width, 0);
	} finally {
		await drop(page, uid);
	}
});

/**
 * The mouse half of THE GESTURE IS UNIFORM: the drag `touch-modality.spec.ts` carries past the floor
 * with a finger, carried just as far with a MOUSE. Both resize to the floor and leave the pane open,
 * which is the whole claim — there is no pointer type for which this gesture means something else.
 */
test('a drag carried far past the floor clamps there, on a mouse too (D-I4)', async ({ page }) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const before = await settledBox(pane(page));
		// The floor is the STYLESHEET's — the lower bound of the pane's own `clamp()`, asked of CSS
		// rather than a `260` this file would hold its own copy of.
		const floor = await paneBound(page, 'floor');
		const g = (await grip(page).boundingBox())!;
		const y = g.y + g.height / 2;
		await page.mouse.move(g.x + g.width / 2, y);
		await page.mouse.down();
		// Past the floor by the pane's whole width again, so nothing about this ends AT the bound.
		await page.mouse.move(g.x + g.width / 2 + before.width + 100, y, { steps: 10 });
		await page.mouse.up();

		await expect(pane(page), 'the pane is still there').toHaveClass(/open/);
		const after = await settledBox(pane(page));
		expect(after.width, 'clamped at the floor, not closed').toBeCloseTo(floor, 0);
	} finally {
		await drop(page, uid);
	}
});

/**
 * D-I9, and the rule it is one instance of: ORIENTATION decides the anchored AXIS; INPUT MODALITY
 * decides only the resting AFFORDANCE. The two are independent — so under one modality the grabber
 * must be the SAME in either anchor, and this reads it in both and compares them against each other
 * rather than against a number either could drift away from alone.
 *
 * It was not. The portrait branch declared a resting pill unconditionally, which is an affordance
 * chosen by orientation: a phone got a chunky pill standing up and a thin line lying down (the same
 * finger, two affordances), and a narrow docked desktop column got the touch grabber under a mouse.
 * The pill belongs to the coarse block — `touch-modality.spec.ts` proves it there, in BOTH anchors —
 * and what portrait carries is geometry alone.
 *
 * What a fine pointer gets is therefore the transparent-until-hover seam in either anchor. That is
 * not an affordance living solely behind `:hover` in CLAUDE.md's sense: hover is the door this
 * modality HAS, which is exactly why the door a finger needs is gated on the finger and not on
 * which way the phone is held.
 */
test('one modality, one grabber — the fine seam is identical in either anchor', async ({ page }) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const seen: Partial<Record<'portrait' | 'landscape', Grabber>> = {};
	for (const [name, size, axis] of [
		['portrait', { width: 600, height: 1000 }, 'y'],
		['landscape', { width: 1280, height: 800 }, 'x']
	] as const) {
		await page.setViewportSize(size);
		const uid = await addAndSelect(page);
		try {
			// The pointer is parked off the pane — no hover anywhere near the grip.
			await page.mouse.move(5, 5);
			await expect
				.poll(() => paneAxis(page), { message: `${name} anchors as its host panel is shaped` })
				.toBe(axis);
			seen[name] = await settledGrabber(page);
			// …and the seam is not merely invisible, it is hover-REVEALED, in this anchor too. That
			// door has to exist wherever the transparent resting state does, or the affordance really
			// would be nowhere on a fine pointer.
			await grip(page).hover();
			await expect
				.poll(() => settledGrabber(page).then((g) => g.painted), {
					message: `hovering the ${name} seam lights it`
				})
				.toBe(true);
		} finally {
			await drop(page, uid);
		}
	}
	expect(seen.portrait, 'one modality, one grabber — whichever axis it is on').toEqual(
		seen.landscape
	);
	expect(
		seen.portrait!.painted,
		'a fine pointer rests on the transparent seam, having hover to find it with'
	).toBe(false);
	expect(seen.portrait!.length, 'a hairline down the whole seam, never a pill').toBe('seam');
	expect(seen.portrait!.rounded, 'and uncapped — the pill is the coarse pointer’s').toBe(false);
});

test('the ✕ dismisses the pane in either anchor', async ({ page }) => {
	// `inspector-dismiss.spec.ts` proves the ✕ on the right-hand pane; this is the same door on the
	// SHEET, and D-I4 is why it must exist there too — it is the ONLY way out, in either anchor.
	await page.setViewportSize({ width: 600, height: 1000 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		await expect
			.poll(() =>
				pane(page).evaluate((el) => getComputedStyle(el).getPropertyValue('--pane-axis').trim())
			)
			.toBe('y');
		await pane(page).getByTestId('inspector-close').click();
		await expect(pane(page), 'the sheet closes').toHaveCount(0);
	} finally {
		await drop(page, uid);
	}
});
