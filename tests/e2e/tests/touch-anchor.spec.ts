/**
 * What must survive the orientation change, and so runs in BOTH phone anchors.
 *
 * Orientation picks only the ANCHOR — a side pane in landscape, a bottom sheet in portrait — while
 * INPUT MODALITY picks the gesture and the affordance. So the same assertions have to come back the
 * same answer in either one, and running one file in two projects is what makes a re-coupling fail
 * by name instead of going unnoticed.
 */

import { test, expect, type Locator, type Page } from '@playwright/test';
import {
	dragGripToTheFloor,
	dropNode,
	editorHost,
	expectDragPastTheFloorStillResizes,
	expectEdgeDragFollowsThePointer,
	expectRestingPill,
	openInspector,
	pane,
	paneAxis,
	paneBound,
	rootRem,
	sizeOn
} from '../lib/inspector';
import { settledBox } from '../lib/geometry';
import { waitForApp } from '../lib/app';
import { nodes } from '../lib/goofi';
import { openGhost } from '../lib/placement';
import { touchSession, type TouchPoint } from '../lib/touch';

test.describe('one modality, one gesture, in either anchor', () => {
	/**
	 * The inspector's INPUT-MODALITY behaviour, run in BOTH orientations off ONE set of assertions.
	 *
	 * `playwright.config.ts` points `touch` (Pixel 7 portrait → the bottom sheet) and `touch-landscape`
	 * (→ the right-hand pane) at this file. Everything asserted below lives in `lib/inspector.ts`,
	 * normalised to whichever axis the container query anchored the pane on, so the two projects run
	 * the same bytes rather than two copies that can drift apart.
	 *
	 * That IS the guard. The rule is that ORIENTATION decides the anchored AXIS, INPUT MODALITY decides
	 * only the resting AFFORDANCE (the grabber pill and the coarse hit band), THE GESTURE IS UNIFORM —
	 * edge-drag, identical on a mouse and a finger — and closing is the explicit ✕ alone. A change that
	 * re-couples any of those makes exactly one of these projects disagree, immediately and by name. It
	 * is what caught the defect this file was written for: a resting grabber declared by the portrait
	 * branch, so one finger got a chunky pill standing up and a thin line lying down.
	 *
	 * The rule used to end "…modality decides the GESTURE and its affordance", because a touch swipe
	 * past the floor closed the pane. That was the only gesture modality ever reached, and it is gone.
	 *
	 * Only two projects, not three: `tablet` is portrait as well, so it would re-measure `touch`'s
	 * answer. What has to be re-run is the OTHER anchor, and there is exactly one of those.
	 */

	test.beforeEach(async ({ page }) => {
		// The pane slides over `--dur-slow` and nearly everything here is a POSITION; the app's own
		// reduced-motion rule collapses the transition, so a read describes the layout and not a frame.
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
	});

	/* The anti-vacuity guard, and it comes first: "the same assertions in both orientations" says
	   nothing unless the two projects really are in different anchors. The anchor is asked of the HOST
	   PANEL's shape (D-I2) — not the viewport, not the device class — so this asserts the pane against
	   the very box the container query measures. */
	test('this project’s host panel picks the anchor, and the pane agrees with it', async ({ page }) => {
		const uid = await openInspector(page);
		try {
			const host = (await editorHost(page).boundingBox())!;
			expect(await paneAxis(page), 'the anchor is the host panel’s shape and nothing else').toBe(
				host.height >= host.width ? 'y' : 'x'
			);
		} finally {
			await dropNode(page, uid);
		}
	});

	/* The limits, in whichever anchor: a TENTH of the host at the bottom, NINE TENTHS at the top.
	   Measured off the pane rather than restated, and measured against the HOST panel rather than the
	   viewport — the pane answers to the surface it lives in (D-I2), and on this phone the two are
	   different numbers. One `clamp()` per axis states both, so the floor is below the ceiling by
	   construction and the pair that crossed (a 256px ceiling under a 260px floor, an EMPTY range) is
	   no longer spellable. */
	test('the pane’s floor and its ceiling are a tenth and nine tenths of its HOST', async ({ page }) => {
		const uid = await openInspector(page);
		try {
			const axis = await paneAxis(page);
			const host = sizeOn(axis, (await editorHost(page).boundingBox())!);
			expect(await paneBound(page, 'floor'), `${axis}: a tenth of a ${host}px host`).toBeCloseTo(
				host * 0.1,
				0
			);
			expect(
				await paneBound(page, 'ceiling'),
				`${axis}: nine tenths of a ${host}px host`
			).toBeCloseTo(host * 0.9, 0);
		} finally {
			await dropNode(page, uid);
		}
	});

	/* …and widening them MOVED NOTHING. The resting size used to EMERGE from the old ceiling clamping
	   a larger fallback, so raising the ceiling is precisely the edit that could have dragged it along;
	   it is stated outright now, and this is what says it did not move.

	   Sitting strictly INSIDE both limits is the other half of the same claim, and the ANTI-VACUITY
	   guard the rest of this file rests on: two anchors running one set of assertions prove nothing
	   about an anchor where there is nothing left to assert. On THIS project's 854px host the pane's
	   two bounds once crossed — a 256px ceiling under a 260px floor — so the landscape pane had
	   NEGATIVE room and the edge drag below was dragging zero pixels and passing. */
	test('the resting default did not move, and now sits strictly inside those limits', async ({
		page
	}) => {
		const uid = await openInspector(page);
		try {
			const axis = await paneAxis(page);
			const host = (await editorHost(page).boundingBox())!;
			const resting = sizeOn(axis, await settledBox(pane(page)));
			expect(
				resting,
				axis === 'y' ? '60% of the host' : 'the lesser of 40% of the host and 30rem'
			).toBeCloseTo(
				axis === 'y' ? host.height * 0.6 : Math.min(host.width * 0.4, 30 * (await rootRem(page))),
				0
			);
			expect(resting, 'above the floor, so it can be dragged smaller').toBeGreaterThan(
				await paneBound(page, 'floor')
			);
			expect(resting, 'and below the ceiling, so it can be dragged larger').toBeLessThan(
				await paneBound(page, 'ceiling')
			);
		} finally {
			await dropNode(page, uid);
		}
	});

	test('a coarse pointer rests the SAME grabber pill in either anchor (D-I9)', async ({ page }) => {
		const uid = await openInspector(page);
		try {
			await expectRestingPill(page);
		} finally {
			await dropNode(page, uid);
		}
	});

	/* The other half of D-I4's "edge drag is for BOTH", and the half that was only ever proved on the Y
	   axis: the same grip, the same module, the same persistence idiom, a finger instead of a mouse.
	   `inspector-orientation.spec.ts` runs the mouse in both anchors; this runs the finger in both. */
	test('an edge drag by TOUCH moves the pane to where the pointer asked (D-I3/D-I4)', async ({
		page
	}) => {
		const uid = await openInspector(page);
		try {
			// 60px: inside the room the TIGHTER anchor has (landscape's 82px above its floor), because
			// the helper now asserts that room rather than clamping the drag down into it.
			await expectEdgeDragFollowsThePointer(page, 60);
		} finally {
			await dropNode(page, uid);
		}
	});

	test('a drag carried past the floor is still a resize, never a dismiss (D-I4)', async ({ page }) => {
		const uid = await openInspector(page);
		try {
			await expectDragPastTheFloorStillResizes(page);
		} finally {
			await dropNode(page, uid);
		}
	});

	/* THE ONE GUARD THE DELETION MADE NECESSARY. The ✕ is the only way out of the pane now, so it has to
	   survive the pane's smallest legal size — and the floor moved down to 10% of the host, which is
	   ~85px of an 854px landscape host and a ~74px sheet in portrait. Neither is a geometry the ✕ has
	   ever had to work in, and the pane is DRAGGED there rather than seeded there, because a stored size
	   restores a pane that was never squeezed.

	   Present is not the claim; HITTABLE is. The coarse hit band leans INWARD over the pane's own rows
	   in landscape, and `.ins-head` is a row it could reach: in portrait that reach once landed on this
	   very ✕ (measured then: band bottom 418px, ✕ centre 416px) and swallowed the pane's one pointer
	   door. A tap alone would not notice — Playwright taps the element's box whether or not anything is
	   laid over it — so the topmost element at the ✕'s centre is asserted to be the ✕ itself. */
	test('the ✕ is still there, still hittable, and still closes the pane AT the floor', async ({
		page
	}) => {
		const uid = await openInspector(page);
		try {
			await dragGripToTheFloor(page);

			const close = pane(page).getByTestId('inspector-close');
			await expect(close, 'the pane keeps its one door at its smallest size').toBeVisible();
			const b = (await close.boundingBox())!;
			// Inside the PANE, not merely on-screen: the pane hugs the screen edge in this anchor, so a
			// ✕ pushed past its edge is a ✕ pushed off the device. It happened by flex arithmetic — the
			// identity Bar's end group could shrink below its unshrinkable button, and the overflow
			// walked right — so the box is asserted against the box that must contain it.
			const pb = (await pane(page).boundingBox())!;
			expect(b.x + b.width, 'the ✕ stays inside the pane').toBeLessThanOrEqual(pb.x + pb.width + 0.5);
			expect(
				await page.evaluate(
					(p) =>
						document.elementFromPoint(p.x, p.y)?.closest('[data-testid]')?.getAttribute('data-testid'),
					{ x: b.x + b.width / 2, y: b.y + b.height / 2 }
				),
				'nothing is laid over it — a tap at its centre reaches the ✕ itself'
			).toBe('inspector-close');

			await close.tap();
			await expect(pane(page), 'and it closes the pane').not.toHaveClass(/open/);
			await expect(pane(page), 'the outro finishes hidden').toHaveCSS('visibility', 'hidden');
		} finally {
			await dropNode(page, uid);
		}
	});
});
test.describe('placing a node', () => {
	/**
	 * Placing a new node WITH A FINGER — the gesture, and the two halves of it that are easy to confuse.
	 *
	 * Adding a node spawns a ghost that follows the pointer until it is put down. On a phone that was
	 * broken in both directions at once: `PlacementPreview` listened for `mousemove`/`mousedown`/`click`
	 * and nothing else, so no finger ever moved the ghost, and the drag it did not consume reached
	 * SvelteFlow's panner instead — the canvas slid out from under a ghost frozen where the palette row
	 * had been, and the tap that followed committed the node there.
	 *
	 * SO BOTH HALVES ARE ASSERTED, SEPARATELY AND ON PURPOSE. The ghost renders inside a
	 * `ViewportPortal`, i.e. in FLOW coordinates, so "the ghost moved" and "the canvas panned" produce
	 * the SAME screen-space delta and a spec that measures only the ghost passes on the old behaviour.
	 * What pins it is the pair: the ghost's screen box tracks the finger, AND `.svelte-flow__viewport`'s
	 * transform is byte-identical before and after.
	 *
	 * AND THE GHOST IS CARRIED BY ITS MIDDLE, which is the other thing a finger changes: a cursor is a
	 * visible point, a finger is an opaque disc over the corner it is on, so the mouse ghost hangs off
	 * its top-left and the touch ghost is centred. Both the DRAWN box and the COMMITTED position are
	 * measured against that centre, because centring only the CSS transform would satisfy the first and
	 * drop the node half a card away from the second.
	 *
	 * MODALITY, NOT ORIENTATION — the rule `panels/paneDrag.ts` and `touch-modality.spec.ts` are built
	 * on: orientation decides an anchored axis, input modality decides a gesture. This gesture is
	 * modality-gated (`pointerType === 'touch'`, asked per event so a hybrid device stays right on both
	 * of its inputs), which means it must answer identically in either anchor — so `playwright.config.ts`
	 * runs these same bytes in `touch` AND `touch-landscape`. A re-coupling to orientation fails one
	 * project by name instead of going unnoticed.
	 */

	test.beforeEach(async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await clearGraph(page);
	});

	/** Hand the shared backend back empty even on failure — a leftover card changes where the NEXT
	 *  spec's `emptySpot` may press, and adds a snap target that moves this one's arithmetic. */
	test.afterEach(async ({ page }) => {
		await clearGraph(page).catch(() => {});
	});

	async function clearGraph(page: Page): Promise<void> {
		await page.evaluate(() => {
			const g = (window as any).goofi;
			const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
			if (uids.length) return g.commands.removeNodes(uids);
		});
		await expect.poll(async () => (await nodes(page)).length).toBe(0);
	}

	/** Two points inside the flow pane, far enough apart to be a drag in EITHER anchor — read off the
	 *  pane itself rather than hardcoded, since portrait and landscape share these bytes. */
	async function dragPoints(page: Page): Promise<{ from: TouchPoint; to: TouchPoint }> {
		const p = (await page.locator('.svelte-flow__pane').first().boundingBox())!;
		const at = (fx: number, fy: number): TouchPoint => ({
			x: Math.round(p.x + p.width * fx),
			y: Math.round(p.y + p.height * fy)
		});
		return { from: at(0.25, 0.3), to: at(0.6, 0.6) };
	}

	/** The pan/zoom matrix every flow-space thing is drawn through. */
	const viewport = (page: Page): Locator => page.locator('.svelte-flow__viewport');

	interface Box {
		x: number;
		y: number;
		width: number;
		height: number;
	}

	/** A rendered box, in SCREEN px — i.e. already through the viewport's zoom. */
	async function boxAt(loc: Locator): Promise<Box> {
		return (await loc.boundingBox())!;
	}

	const topLeft = (b: Box): TouchPoint => ({ x: b.x, y: b.y });
	const centre = (b: Box): TouchPoint => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 });

	/**
	 * THE FINGER IS OVER THE MIDDLE OF THE GHOST, not over its corner.
	 *
	 * A cursor is a point the user can see, so the mouse ghost hangs off its top-left exactly where the
	 * arrow is — that path is the reference and is unchanged, and `default`'s placement specs still
	 * measure it. A finger COVERS the corner it is on: dragging a card by an edge you cannot see is the
	 * bug this centring fixes, so on touch the anchor moves to the ghost's centre.
	 *
	 * The graph is emptied first, so there are no snap targets and the position is the finger's
	 * unmodified — the few px of slack are the commit's rounding to whole flow units, taken back
	 * through a 0.85 zoom.
	 */
	const TOL = 4;
	function expectAt(p: TouchPoint, at: TouchPoint, what: string): void {
		expect(Math.abs(p.x - at.x), `${what}: horizontally (at ${p.x}, wanted ${at.x})`).toBeLessThanOrEqual(TOL);
		expect(Math.abs(p.y - at.y), `${what}: vertically (at ${p.y}, wanted ${at.y})`).toBeLessThanOrEqual(TOL);
	}

	/**
	 * The ghost is centred on the finger IN FLOW UNITS — one offset, applied to the anchored flow
	 * position, so the snap arithmetic, the drawing and the committed position are all the same number.
	 * Measured here at the editor's initial 0.85 zoom, which is what makes this an assertion about
	 * UNITS: a half-size subtracted in SCREEN px instead of flow units lands the centre
	 * `(1 - 0.85) * half` off — ~17px horizontally, four times this tolerance.
	 */
	const expectCentredOn = (b: Box, at: TouchPoint, what: string): void => expectAt(centre(b), at, what);

	test('a touch drag moves the GHOST and leaves the canvas exactly where it was', async ({ page }) => {
		const ghost = await openGhost(page, 'Oscillator');
		const { from, to } = await dragPoints(page);
		const touch = await touchSession(page);

		// The matrix BEFORE anything is touched. Read as a string and compared as one: a pan of a single
		// px changes it, and nothing else on this page writes it.
		const before = await viewport(page).evaluate((el) => getComputedStyle(el).transform);

		await touch.down(from);
		expectCentredOn(await boxAt(ghost), from, 'the press put the ghost under the finger');

		for (let i = 1; i <= 5; i++) {
			await touch.moveTo({
				x: Math.round(from.x + ((to.x - from.x) * i) / 5),
				y: Math.round(from.y + ((to.y - from.y) * i) / 5)
			});
		}
		const held = await boxAt(ghost);
		expectCentredOn(held, to, 'and it tracked the finger across the canvas');

		// THE HALF THAT REGRESSES SILENTLY. Both a moving ghost and a panning canvas move the ghost's
		// screen box by the drag delta, so the assertion above cannot tell them apart — this one can.
		await expect(
			viewport(page),
			'the drag moved the ghost, NOT the canvas — SvelteFlow never saw it'
		).toHaveCSS('transform', before);

		// Chromium reads back-to-back synthetic moves as a FLING, which then eats the next tap anywhere
		// on the page. Letting the finger come to rest before it lifts is what keeps this spec from
		// poisoning the one after it.
		await page.waitForTimeout(150);
		await touch.up();

		await expect(ghost, 'lifting the finger committed the placement').toHaveCount(0);
		await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);
		const uid = (await nodes(page))[0].uid;
		const card = page.locator(`.svelte-flow__node[data-id="${uid}"]`);
		// THE HALF-NODE-OFF BUG, stated as a measurement. Centring only the ghost's CSS transform would
		// draw exactly the box asserted above and still COMMIT the un-offset position, dropping the node
		// half a card away from where it was being carried. Comparing the card's top-left against the
		// ghost's own — the last one it drew before the lift — is what tells those two apart.
		// (And that is the whole chain: the ghost was centred on the finger, the card is where the ghost
		// was, so the card is centred on the finger — without this spec having to assume a placed card
		// and its ghost are the same height.)
		expectAt(topLeft(await boxAt(card)), topLeft(held), 'the node landed where the ghost was');
		await expect(viewport(page), 'and the canvas still has not moved').toHaveCSS('transform', before);
	});

	test('a plain tap places the node at the tap, on the same path as the drag', async ({ page }) => {
		const ghost = await openGhost(page, 'Oscillator');
		const { to } = await dragPoints(page);
		// The ghost's rendered size, read while it still exists — the tap is what destroys it, and the
		// committed corner is half of this up and left of where the finger went down.
		const g = await boxAt(ghost);

		// No move between down and up at all: a tap is a drag of zero length, which is the whole reason
		// there is no tap-vs-drag threshold for it to fall on the wrong side of.
		await page.touchscreen.tap(to.x, to.y);

		await expect(ghost, 'the tap committed the placement').toHaveCount(0);
		await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);
		const uid = (await nodes(page))[0].uid;
		expectAt(
			topLeft(await boxAt(page.locator(`.svelte-flow__node[data-id="${uid}"]`))),
			{ x: to.x - g.width / 2, y: to.y - g.height / 2 },
			'placed centred on the tap'
		);
	});

	/* The comparison is the SAME NODE before and after it is put down, which is the only way to state
	   "this one is not placed yet" as a measurement rather than as a constant. A second card measured
	   beside the ghost would say the same thing and would also have to be manoeuvred out of the way of
	   the next `emptySpot`. */
	test('the pending ghost reads as a placeholder, and the node it becomes does not', async ({
		page
	}) => {
		const opacity = (loc: Locator): Promise<number> =>
			loc.evaluate((el) => parseFloat(getComputedStyle(el).opacity));

		const ghost = await openGhost(page, 'Oscillator');
		const pending = await opacity(ghost);

		const { to } = await dragPoints(page);
		await page.touchscreen.tap(to.x, to.y);
		await expect(ghost).toHaveCount(0);
		await expect.poll(async () => (await nodes(page)).length).toBe(1);
		const card = page.locator(`.svelte-flow__node[data-id="${(await nodes(page))[0].uid}"]`);

		expect(await opacity(card), 'a placed node is drawn solid').toBe(1);
		expect(
			pending,
			'a coarse pointer has no cursor to say "this is stuck to me", so the ghost must say it itself'
		).toBeLessThanOrEqual(0.7);
	});
});
