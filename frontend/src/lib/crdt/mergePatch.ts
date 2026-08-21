/** RFC 7386 JSON merge patch — the browser half of `goofi_bridge::doc`. */

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Apply `patch` into `target`, in place: an object merges, `null` removes, anything else replaces whole. */
export function applyMerge(target: Obj, patch: unknown): void {
	if (!isObj(patch)) return;
	for (const [k, pv] of Object.entries(patch)) {
		if (pv === null) {
			delete target[k];
		} else if (isObj(pv)) {
			if (!isObj(target[k])) target[k] = {};
			applyMerge(target[k] as Obj, pv);
		} else {
			target[k] = pv;
		}
	}
}
