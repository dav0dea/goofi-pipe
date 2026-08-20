/**
 * Long-press recognition — the coarse-pointer door onto a command that a fine pointer reaches by
 * double-click, chord or right-click (R spec §3.2b). The node editor uses it for "add a node
 * here", which on touch has no other pointer route.
 *
 * A press fires once it has been held for `HOLD_MS` within `MOVE_TOLERANCE_PX` of where it landed;
 * anything that ends or moves the gesture disarms it, which is what keeps it clear of a pan (a
 * drag moves, a press does not). Distance is measured from the ORIGIN, so a slow walk across the
 * canvas cannot creep past the tolerance one sub-tolerance step at a time.
 *
 * Deliberately event-free: it takes points, not `PointerEvent`s, and never calls `preventDefault`
 * or `stopPropagation`. The gesture it watches therefore still reaches everything else listening
 * for it — SvelteFlow's pan, and `workspace/Panel.svelte`'s capture-phase `setActive`.
 */

/** How long a press must be held. Long enough not to fire on a tap, short enough to feel deliberate. */
export const HOLD_MS = 500;
/** How far a held finger may drift and still count as still. */
export const MOVE_TOLERANCE_PX = 10;

/** The only thing the recognizer needs from a pointer event. */
export interface PressPoint {
	clientX: number;
	clientY: number;
}

export interface LongPress {
	/** Arm at this point, replacing any press already in flight. */
	start(p: PressPoint): void;
	/** Feed a move; disarms once the pointer has left the tolerance around the origin. */
	move(p: PressPoint): void;
	/** Disarm — a release, a cancel, or anything else that ends the gesture. */
	cancel(): void;
}

export function createLongPress(
	onFire: (at: PressPoint) => void,
	opts: { holdMs?: number; tolerancePx?: number } = {}
): LongPress {
	const holdMs = opts.holdMs ?? HOLD_MS;
	const tolerance = opts.tolerancePx ?? MOVE_TOLERANCE_PX;

	let timer: ReturnType<typeof setTimeout> | null = null;
	let origin: PressPoint | null = null;

	function cancel(): void {
		if (timer !== null) clearTimeout(timer);
		timer = null;
		origin = null;
	}

	return {
		start(p: PressPoint): void {
			cancel();
			origin = { clientX: p.clientX, clientY: p.clientY };
			timer = setTimeout(() => {
				const at = origin;
				cancel();
				if (at) onFire(at);
			}, holdMs);
		},
		move(p: PressPoint): void {
			if (!origin) return;
			const dx = p.clientX - origin.clientX;
			const dy = p.clientY - origin.clientY;
			if (dx * dx + dy * dy > tolerance * tolerance) cancel();
		},
		cancel
	};
}
