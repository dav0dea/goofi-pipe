import { describe, it, expect } from 'vitest';
import { createWidthCache, planOverflow, type OverflowItem } from './overflowFit';

/* The three traps D-R6 names, pinned as arithmetic.
 *
 * The whole non-oscillation argument lives in `planOverflow` being PURE: its answer depends on
 * the measured intrinsic widths and the container's budget, never on what is currently rendered.
 * Moving an item into the overflow cannot change the cached widths, and it hands the freed space
 * to the tab strip rather than to either term of the budget — so the observer that re-fires after
 * the write computes the same answer and writes nothing. (Only where the header's flex line is
 * ALREADY overflowing do the status cluster and the strip shrink together, which leaves a narrow
 * band of widths with hysteresis rather than oscillation; TopBar's own note carries that trade.)
 * The two properties below — idempotence and monotonicity in the budget — are what the claim
 * reduces to.
 */

const ITEMS: OverflowItem[] = [
	{ id: 'undo', width: 44 },
	{ id: 'redo', width: 44 },
	{ id: 'save', width: 60 },
	{ id: 'caret', width: 34 },
	{ id: 'load', width: 58 }
];
/** Lowest priority first — the order the bar gives items up (D-R6). */
const SPILL = ['caret', 'load', 'save', 'redo', 'undo'];
const OPTS = { gap: 4, trigger: 44 };

/** What `planOverflow` measures a visible set at: n items, n gaps (one before the trigger), trigger. */
function used(ids: string[]): number {
	const w = ids.reduce((a, id) => a + ITEMS.find((i) => i.id === id)!.width, 0);
	return w + OPTS.gap * ids.length + OPTS.trigger;
}
const ALL = ITEMS.map((i) => i.id);

describe('planOverflow', () => {
	it('spills nothing when the whole bar fits', () => {
		expect(planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(ALL) })).toEqual(new Set());
		expect(planOverflow(ITEMS, SPILL, { ...OPTS, budget: 2000 })).toEqual(new Set());
	});

	it('gives up the lowest-priority item first', () => {
		const spilled = planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(ALL) - 1 });
		expect(spilled).toEqual(new Set(['caret']));
	});

	it('walks down the priority order until the remainder fits', () => {
		// Room for undo + redo only.
		const spilled = planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(['undo', 'redo']) });
		expect(spilled).toEqual(new Set(['caret', 'load', 'save']));
	});

	it('spills everything when not even one item fits beside the trigger', () => {
		expect(planOverflow(ITEMS, SPILL, { ...OPTS, budget: OPTS.trigger })).toEqual(new Set(ALL));
	});

	/* Trap 2: the overflow button occupies space. It is RESIDENT chrome here — it carries the
	   canvas commands at every width — so its width is unconditionally in the budget and there is
	   no "the button just appeared" boundary at which the last item spills forever. */
	it('charges the resident trigger against the budget', () => {
		const bareItems = used(ALL) - OPTS.trigger; // exactly the items, no trigger
		expect(planOverflow(ITEMS, SPILL, { ...OPTS, budget: bareItems })).not.toEqual(new Set());
	});

	/* Trap 1: oscillation. */
	it('is idempotent at every width, including the boundary', () => {
		for (let budget = 0; budget <= used(ALL) + 20; budget++) {
			const a = planOverflow(ITEMS, SPILL, { ...OPTS, budget });
			const b = planOverflow(ITEMS, SPILL, { ...OPTS, budget });
			expect([...b]).toEqual([...a]);
		}
	});

	it('only ever grows the overflow as the budget shrinks', () => {
		let prev = new Set<string>();
		for (let budget = used(ALL) + 20; budget >= 0; budget--) {
			const now = planOverflow(ITEMS, SPILL, { ...OPTS, budget });
			for (const id of prev) expect(now.has(id), `budget ${budget} un-spilled ${id}`).toBe(true);
			prev = now;
		}
	});

	it('returns to the same set after crossing the boundary and back', () => {
		const wide = planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(ALL) });
		planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(ALL) - 1 });
		const back = planOverflow(ITEMS, SPILL, { ...OPTS, budget: used(ALL) });
		expect([...back]).toEqual([...wide]);
	});

	it('ignores an id in the spill order that is not in the bar', () => {
		const spilled = planOverflow(ITEMS, ['ghost', ...SPILL], { ...OPTS, budget: used(ALL) - 1 });
		expect(spilled).toEqual(new Set(['caret']));
	});
});

/* The second consumer's one real difference (the panel header). Its `⋯` is NOT resident chrome:
   the menu holds the spilled items and nothing else, so at a width where everything fits there is
   no menu to open and no trigger drawn. `residentTrigger: false` says so, and the plan then charges
   the trigger exactly where it is rendered.
   That reintroduces trap 2 — a trigger that appears and pushes an item out — as a real case rather
   than a designed-away one, so it is the property below that has to hold instead: the answer stays
   MONOTONE in the budget, because the plan returns the first prefix of the spill order that fits
   and "fits" only ever gets easier as the budget grows. */
describe('a non-resident trigger', () => {
	/** What a bar with no trigger measures a visible set at: n boxes, n − 1 gaps. */
	function bare(ids: string[]): number {
		const w = ids.reduce((a, id) => a + ITEMS.find((i) => i.id === id)!.width, 0);
		return w + OPTS.gap * (ids.length - 1);
	}

	it('does not charge a trigger that is not on screen', () => {
		expect(
			planOverflow(ITEMS, SPILL, { ...OPTS, budget: bare(ALL), residentTrigger: false })
		).toEqual(new Set());
		// …and the resident default still charges it at that very width, unchanged.
		expect(planOverflow(ITEMS, SPILL, { ...OPTS, budget: bare(ALL) })).not.toEqual(new Set());
	});

	it('charges the trigger from the moment the first item spills', () => {
		// One pixel short of the bare fit. The caret leaves — and the `⋯` it leaves for is now on
		// screen and costs more (44) than the caret freed (34 + a gap), so a second item goes with
		// it. Two at once is the honest answer, not an oscillation: it is what one evaluation of the
		// plan returns, and re-running it returns the same.
		const spilled = planOverflow(ITEMS, SPILL, {
			...OPTS,
			budget: bare(ALL) - 1,
			residentTrigger: false
		});
		expect(spilled).toEqual(new Set(['caret', 'load']));
	});

	it('spills everything when only the trigger fits', () => {
		expect(
			planOverflow(ITEMS, SPILL, { ...OPTS, budget: OPTS.trigger, residentTrigger: false })
		).toEqual(new Set(ALL));
	});

	it('only ever grows the overflow as the budget shrinks, trigger appearing and all', () => {
		let prev = new Set<string>();
		for (let budget = used(ALL) + 20; budget >= 0; budget--) {
			const now = planOverflow(ITEMS, SPILL, { ...OPTS, budget, residentTrigger: false });
			for (const id of prev) expect(now.has(id), `budget ${budget} un-spilled ${id}`).toBe(true);
			prev = now;
		}
	});
});

/* Trap 3: cached intrinsic widths go stale — the `html` base is a responsive clamp(), so the same
   button is a different number of pixels at a different root size. */
describe('createWidthCache', () => {
	it('measures once while the root size holds', () => {
		let calls = 0;
		const cache = createWidthCache(() => {
			calls++;
			return [1, 2, 3];
		});
		expect(cache.widths(14)).toEqual([1, 2, 3]);
		cache.widths(14);
		cache.widths(14);
		expect(calls).toBe(1);
	});

	it('re-measures when the root font size moves', () => {
		let rem = 14;
		let calls = 0;
		const cache = createWidthCache(() => {
			calls++;
			return [rem];
		});
		expect(cache.widths(14)).toEqual([14]);
		rem = 16;
		expect(cache.widths(16)).toEqual([16]);
		expect(calls).toBe(2);
		cache.widths(16);
		expect(calls).toBe(2);
	});

	it('re-measures after an explicit invalidate', () => {
		let calls = 0;
		const cache = createWidthCache(() => {
			calls++;
			return [calls];
		});
		expect(cache.widths(14)).toEqual([1]);
		cache.invalidate();
		expect(cache.widths(14)).toEqual([2]);
	});
});
