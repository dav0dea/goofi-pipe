/**
 * `use:portal` — relocate an element to a target (default `document.body`) so
 * popups and overlays escape their parent's stacking/clipping context. Panels
 * host transformed subtrees (SvelteFlow viewports), so menus must portal out to
 * stack predictably against the documented z-index scale.
 */
import type { Action } from 'svelte/action';

export const portal: Action<HTMLElement, HTMLElement | string | undefined> = (node, target) => {
	const resolve = (t?: HTMLElement | string): HTMLElement => {
		if (t instanceof HTMLElement) return t;
		if (typeof t === 'string') return document.querySelector<HTMLElement>(t) ?? document.body;
		return document.body;
	};
	let host = resolve(target);
	host.appendChild(node);
	return {
		update(next) {
			host = resolve(next);
			host.appendChild(node);
		},
		destroy() {
			node.parentNode?.removeChild(node);
		}
	};
};
