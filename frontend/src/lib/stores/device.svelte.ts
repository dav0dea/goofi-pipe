/** Device store — the soft keyboard's overlap of the layout viewport, and nothing else. A rune
 * singleton (like ui()). It owns the `visualViewport` subscription and publishes the measurement as
 * the `--kb-inset` custom property on <html>.
 *
 * One measurement, ONE representation: the custom property. It is published that way precisely
 * because `$lib/ui` must stay a leaf layer — the anchored-overlay clamps that need the number are
 * JS (`ui/clampToViewport.ts`'s `overlayViewport()`, which `Popover`, `ContextMenu` and their
 * callers all measure through), and a primitive that imported this file would pull the store layer
 * into the primitives' bundle chunk and cost a measured first-paint FOUC. `dd8cc4c` promoted that
 * layering rule to a CLAUDE.md hard constraint; `ui/clampToViewport.test.ts` scans for it. Keeping
 * the value in a `$state` field as well would be a second copy with no reader.
 *
 * This is the whole device seam. `data-pointer` / `data-size` / `data-short` were deleted with
 * `classify()` (D-R8): they had no reader, and every question they encoded is one `@media` query
 * away. The keyboard overlap is the one that isn't — it is observable only here. */
import { kbInset } from './deviceClassify';

class DeviceStore {
	private started = false;

	/** Wire up browser listeners and stamp <html>. Call once, on mount, in the browser. */
	init(): void {
		if (this.started || typeof window === 'undefined') return;
		this.started = true;

		const vv = window.visualViewport;
		const measure = (): void => {
			// Pixels the soft keyboard currently covers, measured from the bottom; 0 when it is down.
			const inset = vv ? kbInset(vv.height, window.innerHeight) : 0;
			document.documentElement.style.setProperty('--kb-inset', `${inset}px`);
		};
		// Both viewports move it: the visual one shrinks under the keyboard, and the layout one is
		// the reference the overlap is measured against (it changes on rotate / window resize).
		vv?.addEventListener('resize', measure);
		window.addEventListener('resize', measure);
		measure();
	}
}

let _device: DeviceStore | null = null;
export function device(): DeviceStore {
	if (!_device) _device = new DeviceStore();
	return _device;
}
