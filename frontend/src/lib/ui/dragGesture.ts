/**
 * `beginDrag` — the one pointer-drag lifecycle the shell's drag handles share (R spec §3.1g).
 *
 * Three of them (the panel corner grips, the split seam, the inspector's resize handle) each hand-
 * rolled the same `pointerdown → window move/up` pattern, and all three got the same thing wrong:
 * they detached on `pointerup` ALONE. A touch pointer is CANCELLED, not released, whenever the
 * browser reclaims the gesture — a pan takes over, a system gesture starts, a second finger lands —
 * so the listeners survived it, the preview ghost stayed painted, and the next release anywhere in
 * the app ran the leaked handler and committed the abandoned intent. For a corner grip that intent
 * is a `ws.split()` or a `ws.close()`: an abandoned gesture restructuring the workspace one tap
 * later (H6). `ui/NumberInput.svelte` was the only handler in the tree that already got this right;
 * this is its shape, extracted so the three cannot drift apart again.
 *
 * The pointer is CAPTURED rather than the window listened to, which is the other half of the same
 * correctness: capture guarantees the resolution lands on the element that started the gesture even
 * when the pointer has left it — including a mouse released outside the window, where a window
 * listener hears nothing at all.
 *
 * Pure enough to unit-test without a DOM: a real `HTMLElement` satisfies `DragPointerTarget`
 * structurally, and nothing here reads the document.
 */

/** The three events a drag needs, plus the capture. A real `HTMLElement` satisfies this. */
export interface DragPointerTarget {
	setPointerCapture(pointerId: number): void;
	addEventListener(type: DragEventName, fn: (e: PointerEvent) => void): void;
	removeEventListener(type: DragEventName, fn: (e: PointerEvent) => void): void;
}

type DragEventName = 'pointermove' | 'pointerup' | 'pointercancel';

export interface DragHandlers {
	/** The pointer moved while down. */
	move(e: PointerEvent): void;
	/** The gesture ended under the user's control — apply whatever it built. */
	commit(): void;
	/** The gesture was ABANDONED (the pointer was cancelled, or the owner unmounted mid-drag).
	 * Drop any preview; commit nothing. */
	cancel(): void;
}

/**
 * Arm a drag on `el` for `pointerId`. Returns the teardown for the owner's unmount — calling it
 * after the gesture has already resolved is a no-op, so it is safe to run unconditionally.
 */
export function beginDrag(
	el: DragPointerTarget,
	pointerId: number,
	handlers: DragHandlers
): () => void {
	let live = true;
	const detach = (): void => {
		live = false;
		el.removeEventListener('pointermove', onMove);
		el.removeEventListener('pointerup', onUp);
		el.removeEventListener('pointercancel', onCancel);
	};
	const onMove = (e: PointerEvent): void => handlers.move(e);
	const onUp = (): void => {
		detach();
		handlers.commit();
	};
	const onCancel = (): void => {
		detach();
		handlers.cancel();
	};

	el.setPointerCapture(pointerId);
	el.addEventListener('pointermove', onMove);
	el.addEventListener('pointerup', onUp);
	el.addEventListener('pointercancel', onCancel);

	return () => {
		if (!live) return;
		detach();
		handlers.cancel();
	};
}
