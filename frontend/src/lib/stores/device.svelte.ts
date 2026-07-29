/** Device store — the soft keyboard's overlap of the layout viewport, and nothing else. A rune
 * singleton (like ui()). It owns the `visualViewport` subscription and publishes the measurement
 * twice, because it has two kinds of consumer: as the `--kb-inset` custom property on <html> for
 * CSS (app/Toast.svelte), and as a `$state` field for the two anchored-overlay clamps, which are
 * JS and must re-place when the keyboard opens (ui/Popover.svelte, workspace/ContextMenu.svelte).
 * One measurement, two representations — never two sources of truth.
 *
 * This is the whole device seam. `data-pointer` / `data-size` / `data-short` were deleted with
 * `classify()` (D-R8): they had no reader, and every question they encoded is one `@media` query
 * away. The keyboard overlap is the one that isn't — it is observable only here. */
import { kbInset } from './deviceClassify';

class DeviceStore {
	/** Pixels the soft keyboard currently covers, measured from the bottom. 0 when it is down. */
	kbInset = $state(0);
	private started = false;

	/** Wire up browser listeners and stamp <html>. Call once, on mount, in the browser. */
	init(): void {
		if (this.started || typeof window === 'undefined') return;
		this.started = true;

		const vv = window.visualViewport;
		const measure = (): void => {
			this.kbInset = vv ? kbInset(vv.height, window.innerHeight) : 0;
			document.documentElement.style.setProperty('--kb-inset', `${this.kbInset}px`);
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
