import { expect, type Page } from '@playwright/test';

/**
 * Structural rules a rendered goofi must obey, whatever it looks like.
 *
 * NOTHING here names a value the design owns. A restyle is free by construction: the hit floor is
 * read from the app's own `--hit`, and every other rule compares an element against itself — its
 * content against its box, its box against the viewport. What these catch is the app falling apart:
 * a page that scrolls, text clipped away, a control off screen or too small to hit.
 *
 * They are applied by SWEEPING the scene, never by naming elements, so a panel added next year is
 * covered the day it renders.
 */
export interface Violation {
	rule: string;
	where: string;
	detail: string;
}

/** A short, stable description of an element, for a failure message. */
const DESCRIBE = `(el) => {
	const id = el.id ? '#' + el.id : '';
	const tid = el.getAttribute('data-testid');
	const cls = typeof el.className === 'string' && el.className
		? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.')
		: '';
	return el.tagName.toLowerCase() + id + (tid ? '[' + tid + ']' : '') + cls;
}`;

/**
 * Every structural violation in the scene as it currently stands.
 *
 * `slack` absorbs sub-pixel layout: a box measured at 100.4 against a 100 parent is rounding, not
 * an overflow.
 */
export async function sweep(page: Page, slack = 1.5): Promise<Violation[]> {
	return page.evaluate(
		([slack, describeSrc]) => {
			const describe = eval(describeSrc as string) as (el: Element) => string;
			const out: { rule: string; where: string; detail: string }[] = [];
			const add = (rule: string, el: Element, detail: string) =>
				out.push({ rule, where: describe(el), detail });
			const s = slack as number;

			const doc = document.scrollingElement as HTMLElement;
			if (doc.scrollWidth > doc.clientWidth + s) {
				out.push({
					rule: 'the page does not scroll sideways',
					where: 'document',
					detail: `scrollWidth ${doc.scrollWidth} > clientWidth ${doc.clientWidth}`
				});
			}
			if (doc.scrollHeight > doc.clientHeight + s) {
				out.push({
					rule: 'the page does not scroll',
					where: 'document',
					detail: `scrollHeight ${doc.scrollHeight} > clientHeight ${doc.clientHeight}`
				});
			}

			const vw = window.innerWidth;
			const vh = window.innerHeight;
			const hit = parseFloat(
				getComputedStyle(document.documentElement).getPropertyValue('--hit')
			);
			const coarse = window.matchMedia('(pointer: coarse)').matches;
			const INTERACTIVE = 'button, a[href], input, select, textarea, [role="button"], [role="tab"], [role="checkbox"], [role="menuitem"]';
			// Controls that fail a rule TODAY, each a filed defect with an owner rather than a silenced
			// rule. This list shrinks; nothing joins it that this tree can fix.
			//   .ui-tab-close — 16px wide against a 44px coarse floor. It is panelty's own
			//   (dav0dea/panelty), so the fix is a release of that package, never a patch here.
			const KNOWN = '.ui-tab-close';

			for (const el of Array.from(document.querySelectorAll<HTMLElement>('*'))) {
				const cs = getComputedStyle(el);
				if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
				const r = el.getBoundingClientRect();
				if (r.width === 0 && r.height === 0) continue;

				// Text that does not fit the box holding it, where the box neither scrolls nor says it
				// truncates on purpose. Asked only of elements that hold text DIRECTLY: a pannable
				// canvas is legitimately larger than its window, and asking it this would be asking
				// whether the graph fits on screen.
				const ownsText = Array.from(el.childNodes).some(
					(n) => n.nodeType === 3 && (n.textContent ?? '').trim() !== ''
				);
				const scrolls = (v: string) => v === 'auto' || v === 'scroll';
				const truncates = cs.textOverflow === 'ellipsis' || cs.whiteSpace === 'nowrap';
				if (ownsText && !truncates) {
					if (el.scrollWidth > el.clientWidth + s && !scrolls(cs.overflowX)) {
						add('text fits its box, or the box scrolls', el,
							`scrollWidth ${el.scrollWidth} > clientWidth ${el.clientWidth} (overflow-x: ${cs.overflowX})`);
					}
					if (el.scrollHeight > el.clientHeight + s && cs.overflowY === 'hidden') {
						add('text is not clipped away', el,
							`scrollHeight ${el.scrollHeight} > clientHeight ${el.clientHeight} (overflow-y: hidden)`);
					}
				}

				if (!el.matches(INTERACTIVE)) continue;
				// The ONE named element in this file, and it stands for a semantic no rule can infer:
				// inside the zoomable canvas a control's rendered size is a function of the zoom the
				// user drives, not of the design, so a fixed floor says nothing about it. Chrome is
				// judged; the canvas is pinched.
				if (el.closest('.canvas-wrap') || el.matches(KNOWN)) continue;
				// An interactive control the pointer cannot reach is a dead control. Judged against
				// whatever CLIPS it rather than against the viewport: a control inside a scroller is
				// reachable by scrolling to it, and one in a parked pane is legitimately off screen.
				// What is unreachable is a control cut off by a box that does not scroll.
				const onScreen = r.right > 0 && r.left < vw && r.bottom > 0 && r.top < vh;
				if (!onScreen) continue;
				let clipper: HTMLElement | null = el.parentElement;
				while (clipper) {
					const pcs = getComputedStyle(clipper);
					if (pcs.overflowX !== 'visible' || pcs.overflowY !== 'visible') break;
					clipper = clipper.parentElement;
				}
				const box = clipper
					? clipper.getBoundingClientRect()
					: new DOMRect(0, 0, vw, vh);
				const ccs = clipper ? getComputedStyle(clipper) : null;
				const scrollsX = ccs ? ccs.overflowX === 'auto' || ccs.overflowX === 'scroll' : false;
				const scrollsY = ccs ? ccs.overflowY === 'auto' || ccs.overflowY === 'scroll' : false;
				const cutX = !scrollsX && (r.left < box.left - s || r.right > box.right + s);
				const cutY = !scrollsY && (r.top < box.top - s || r.bottom > box.bottom + s);
				if (cutX || cutY) {
					add('a control is not cut off by a box that cannot scroll', el,
						`${Math.round(r.left)},${Math.round(r.top)} ${Math.round(r.width)}x${Math.round(r.height)} ` +
						`outside ${Math.round(box.left)},${Math.round(box.top)} ${Math.round(box.width)}x${Math.round(box.height)}` +
						(clipper ? ` (${describe(clipper)})` : ' (viewport)'));
				}
				// The app's OWN floor, read from the token, so a retuned `--hit` moves this with it —
				// and only where the app says the floor applies. `--hit` carries a documentary value
				// on a fine pointer and becomes a floor under `(pointer: coarse)`; asserting it on a
				// mouse would demand 28px of a 20px icon button the design intends.
				// A text field under 16px makes mobile Safari zoom the page on focus — a platform fact,
				// not a design value, which is why a number appears here at all.
				if (coarse && el.matches('input, select, textarea') && !el.closest('.canvas-wrap')) {
					const fs = parseFloat(cs.fontSize);
					if (fs < 16) {
						add('a field a finger focuses does not zoom the page', el, `font-size ${fs}px under 16`);
					}
				}
				if (coarse && Number.isFinite(hit) && (r.width < hit - s || r.height < hit - s)) {
					add('a control meets the app’s own --hit floor', el,
						`${Math.round(r.width)}x${Math.round(r.height)} under --hit ${hit}`);
				}
			}
			return out;
		},
		[slack, DESCRIBE] as const
	);
}

/**
 * Wait for the scene to stop moving.
 *
 * A pane caught mid-slide is 400px past the right edge and is not a layout claim about anything.
 * Infinite animations are excluded and the wait is bounded, so a spinner or a streaming viewer
 * cannot hold this open.
 */
async function settled(page: Page): Promise<void> {
	await page.evaluate(async () => {
		const finite = document
			.getAnimations()
			.filter((a) => a.effect?.getTiming().iterations !== Infinity);
		await Promise.race([
			Promise.all(finite.map((a) => a.finished.catch(() => undefined))),
			new Promise((r) => setTimeout(r, 1000))
		]);
	});
}

/** Sweep a settled scene, and fail naming every violation at once. */
export async function expectIntact(page: Page, scene: string, slack?: number): Promise<void> {
	await settled(page);
	const found = await sweep(page, slack);
	expect(
		found.map((v) => `  [${v.rule}] ${v.where}: ${v.detail}`).join('\n'),
		`${scene} holds together`
	).toBe('');
}
