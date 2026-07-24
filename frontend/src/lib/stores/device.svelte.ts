/** Device-class store — the app's ONLY classifier. A rune singleton (like ui()). It owns the
 * matchMedia + visualViewport subscriptions and stamps document.documentElement so both
 * consumers see one truth: components read the rune, global CSS reads the data-* attributes
 * (mirrors data-theme). All pure arithmetic is in deviceClassify.ts. */
import { classify, kbInset, type SizeClass } from './deviceClassify';

class DeviceStore {
	pointer = $state<'coarse' | 'fine'>('fine');
	size = $state<SizeClass>('full');
	short = $state(false);
	private started = false;

	/** Wire up browser listeners and stamp <html>. Call once, on mount, in the browser. */
	init(): void {
		if (this.started || typeof window === 'undefined') return;
		this.started = true;

		const coarse = window.matchMedia('(pointer: coarse)');
		const onPointer = () => {
			this.pointer = coarse.matches ? 'coarse' : 'fine';
			this.stamp();
		};
		coarse.addEventListener('change', onPointer);

		const onResize = () => {
			const c = classify(window.innerWidth, window.innerHeight, { coarse: coarse.matches });
			this.size = c.size;
			this.short = c.short;
			this.stamp();
		};
		window.addEventListener('resize', onResize);

		const vv = window.visualViewport;
		if (vv) vv.addEventListener('resize', () => this.stamp());

		onPointer();
		onResize();
	}

	private stamp(): void {
		const el = document.documentElement;
		el.setAttribute('data-pointer', this.pointer);
		el.setAttribute('data-size', this.size);
		el.toggleAttribute('data-short', this.short);
		const vv = window.visualViewport;
		const inset = vv ? kbInset(vv.height, window.innerHeight) : 0;
		el.style.setProperty('--kb-inset', `${inset}px`);
	}
}

let _device: DeviceStore | null = null;
export function device(): DeviceStore {
	if (!_device) _device = new DeviceStore();
	return _device;
}
