import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import {
	bareSpot,
	clearStage,
	inConn,
	inputLabel,
	outHandle,
	proximityStage
} from '../lib/cableDrag';

/**
 * The fine-pointer half of `touch-slot-name.spec.ts`.
 *
 * The proximity reveal is ADDITIVE here: a mouse keeps the hover reveal it always had (and the
 * `:focus-visible` one beside it), and gains the same in-flight proximity reveal touch now depends
 * on. Both are asserted, because "the existing specs still pass" cannot prove the first — nothing
 * else in the suite reads this label's opacity on a mouse.
 */

test('hover still reveals an input’s name, and a cable in flight reveals it too', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const stage = await proximityStage(page);
	try {
		const label = inputLabel(page, stage.near);
		await expect(label, 'hidden at rest, as it always was on a mouse').toHaveCSS('opacity', '0');

		// The reveal the mouse already had, and must keep.
		const pill = await inConn(page, stage.near);
		await page.mouse.move(pill.x, pill.y);
		await expect(label, 'hovering the connector still names it').toHaveCSS('opacity', '1');
		await page.mouse.move(2, 2);
		await expect(label).toHaveCSS('opacity', '0');

		// …and the one it gains: while a cable is in flight, proximity alone is enough — the pointer
		// stops short of the pill, so nothing here can be the hover rule answering in disguise.
		const from = await outHandle(page, stage.src);
		await page.mouse.move(from.x, from.y);
		await page.mouse.down();
		try {
			for (const f of [0.4, 0.75, 1]) {
				await page.mouse.move(
					Math.round(from.x + (pill.x - 20 - from.x) * f),
					Math.round(from.y + (pill.y - from.y) * f)
				);
				await page.waitForTimeout(30);
			}
			await expect(
				label,
				'the input the cable is closing on names itself, 20px short of the pill'
			).toHaveCSS('opacity', '1');
			await expect(
				inputLabel(page, stage.far),
				'…and a distant one stays quiet'
			).toHaveCSS('opacity', '0');
		} finally {
			// Release over bare canvas, so this measures the reveal and never wires anything.
			const away = await bareSpot(page);
			await page.mouse.move(away.x, away.y);
			await page.mouse.up();
		}
		await expect(label, 'the tag goes away with the cable').toHaveCSS('opacity', '0');
	} finally {
		await clearStage(page, stage);
	}
});
