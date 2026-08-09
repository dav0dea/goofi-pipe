import type { Page } from '@playwright/test';

/**
 * Where a label's visible INK sits, as one in-page probe.
 *
 * Several pins ask the same question of different labels — a Badge/Chip against its pill, the
 * layout tab's label against the ＋ — and each one owns a different REFERENCE to compare against.
 * So the probe answers only about the ink and leaves the comparison to the pin: the caller reads
 * its own reference box and asserts its own tolerance, exactly as it did when the measurement was
 * inlined. What is shared is the arithmetic, which is the part that was drifting.
 *
 * The ink EXTENT is derived at a 400px reference size and scaled back, never read at render size:
 * Chrome rounds `actualBoundingBox*` to whole device pixels, so at a ~10px label a 7.2px cap band
 * is reported as 8 and half that error (0.4px) lands straight in the delta — measurement noise, not
 * ink position (DPR-8 screenshots of the pills put the real offset at 0.54px while the raw metrics
 * claimed 0.92px). `fontBoundingBoxAscent` deliberately stays at RENDER size: the LAYOUT font box
 * is built from those same rounded integers, so a reference-scaled ascent would disagree with the
 * box the baseline is measured from — which is exactly what `fontBoxCheck` asserts.
 */
export interface InkMetrics {
	/** Viewport y of the centre of the visible glyph run. */
	inkCenter: number;
	/** Canvas font box minus the layout font box; |·| ≤ 1 says the baseline estimate is sound. */
	fontBoxCheck: number;
}

declare global {
	interface Window {
		__inkMetrics: (el: Element) => InkMetrics;
	}
}

/** Install `window.__inkMetrics` for this page. Call it BEFORE `page.goto`, like any init script. */
export async function installInkProbe(page: Page): Promise<void> {
	await page.addInitScript(() => {
		window.__inkMetrics = (el: Element): InkMetrics => {
			const cs = getComputedStyle(el);
			// The Range rect is the LINE BOX of the run itself — the element's own border box would
			// include whatever padding/centring the component put around it.
			const range = document.createRange();
			range.selectNodeContents(el);
			const box = range.getBoundingClientRect();
			// Canvas measures the string it is GIVEN, so a CSS text-transform has to be applied by
			// hand or the pills would be measured in the lowercase they are authored in.
			const written = (el.textContent ?? '').trim();
			const text = cs.textTransform === 'uppercase' ? written.toUpperCase() : written;
			const size = parseFloat(cs.fontSize);
			const cv = document.createElement('canvas').getContext('2d')!;
			cv.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
			const met = cv.measureText(text);
			const REF = 400;
			cv.font = `${cs.fontWeight} ${REF}px ${cs.fontFamily}`;
			const ref = cv.measureText(text);
			const extent = ((ref.actualBoundingBoxAscent - ref.actualBoundingBoxDescent) / REF) * size;
			const baseline = box.top + met.fontBoundingBoxAscent;
			return {
				inkCenter: baseline - extent / 2,
				fontBoxCheck: met.fontBoundingBoxAscent + met.fontBoundingBoxDescent - box.height
			};
		};
	});
}
