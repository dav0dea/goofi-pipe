/** Device store — the soft keyboard's overlap of the layout viewport, published as the `--kb-inset`
 * custom property so `$lib/ui` primitives can read it without importing a store. */
import { kbInset } from './deviceClassify';

class DeviceStore {
	private started = false;

	/** Wire up browser listeners and stamp <html>. Call once, on mount. */
	init(): void {
		if (this.started || typeof window === 'undefined') return;
		this.started = true;

		const vv = window.visualViewport;
		const measure = (): void => {
			const inset = vv ? kbInset(vv.height, window.innerHeight) : 0;
			document.documentElement.style.setProperty('--kb-inset', `${inset}px`);
		};
		// The layout viewport is the reference the overlap is measured against, so it moves it too.
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
