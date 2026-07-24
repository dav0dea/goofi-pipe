/**
 * The declarative field-dependency predicate + evaluator (spec §5, D-N5).
 *
 * `show_when` is a frontend-only mechanism: no backend/descriptor/manifest field carries a param
 * dependency today. A field with no predicate always shows; a field whose predicate evaluates
 * `false` is hidden. Kept pure + unit-tested so `ParamForm` is a thin visibility filter and a
 * future backend-populated `show_when` flows through this same evaluator with no rework.
 *
 * `values` is the node's live param values across ALL groups (dependencies may cross groups).
 * Fail-closed: if the controlling `param` key is ABSENT from `values`, the predicate is `false`
 * (hide) — a dependent whose controller isn't present cannot be shown. A key present with value
 * `undefined` is a real value and follows the same rule as any other. The function escape hatch
 * owns its own absence semantics (it is called unconditionally).
 */
export type ShowWhenPredicate =
	| { param: string; equals: unknown }
	| { param: string; in: unknown[] }
	| { param: string; truthy: boolean }
	| ((values: Record<string, unknown>) => boolean);

export function evalShowWhen(pred: ShowWhenPredicate, values: Record<string, unknown>): boolean {
	if (typeof pred === 'function') return pred(values);
	// Fail-closed on an absent controlling key (present-but-undefined is a real value, so we test
	// key membership with `in`, not `values[pred.param] === undefined`).
	if (!(pred.param in values)) return false;
	if ('equals' in pred) return values[pred.param] === pred.equals;
	if ('in' in pred) return pred.in.includes(values[pred.param]);
	return Boolean(values[pred.param]) === pred.truthy;
}
