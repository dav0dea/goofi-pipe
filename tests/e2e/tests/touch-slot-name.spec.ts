import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { touchSession } from '../lib/touch';
import {
	bareSpot,
	clearStage,
	inConn,
	inputLabel,
	outHandle,
	proximityStage
} from '../lib/cableDrag';

/**
 * The input-name tag under a coarse pointer, after the always-open rule was replaced.
 *
 * `.conn-label` is the ONLY rendering of an input slot's name. A mouse hovers the pill to ask for
 * it; a finger cannot, so R rested every one of them open under the coarse idiom — which put a name
 * tag on every input on the whole canvas, permanently, for a question nobody was asking. The door is
 * a PROXIMITY reveal now, scoped to a cable actually being in flight, and it is the same door on
 * both modalities rather than a phone-only rule.
 *
 * Both halves are pinned here because both can regress independently: the tag resting hidden (which
 * the deleted rule would restore) and the tag appearing for the input the cable is closing on but
 * not for a distant one (which a missing or oversized radius would break).
 */

test('with no cable in flight, an input names nothing', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const stage = await proximityStage(page);
	try {
		await expect(
			inputLabel(page, stage.near),
			'the always-open coarse rule is gone — a name tag is an answer to a question'
		).toHaveCSS('opacity', '0');
		await expect(inputLabel(page, stage.far)).toHaveCSS('opacity', '0');
	} finally {
		await clearStage(page, stage);
	}
});

test('a cable in flight names the input it is closing on, and not a distant one', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const stage = await proximityStage(page);
	const touch = await touchSession(page);
	let down = false;
	try {
		const from = await outHandle(page, stage.src);
		const to = await inConn(page, stage.near);

		await touch.down(from);
		down = true;
		// Walk the cable across, ending ON the target input. Stepped, and with the finger let come to
		// rest before anything is read: Chromium holds the first touchmoves back behind its touch slop,
		// and reads back-to-back synthetic moves as a fling.
		for (const f of [0.3, 0.6, 0.85, 1]) {
			await touch.moveTo({
				x: Math.round(from.x + (to.x - from.x) * f),
				y: Math.round(from.y + (to.y - from.y) * f)
			});
			await page.waitForTimeout(30);
		}
		await page.waitForTimeout(180);

		await expect(
			inputLabel(page, stage.near),
			'the input the cable is closing on names itself'
		).toHaveCSS('opacity', '1');
		await expect(
			inputLabel(page, stage.far),
			'…and one a screenful away stays quiet — a reveal, not a floodlight'
		).toHaveCSS('opacity', '0');

		// Let go over bare canvas, so this measures the reveal and never wires anything.
		const away = await bareSpot(page);
		await touch.moveTo(away);
		await page.waitForTimeout(180);
		await touch.up();
		down = false;
		await expect(
			inputLabel(page, stage.near),
			'and the tag goes away with the cable that summoned it'
		).toHaveCSS('opacity', '0');
	} finally {
		if (down) await touch.up();
		await clearStage(page, stage);
	}
});
