import { describe, it, expect } from 'vitest';
import { displayValue } from './liveValue.svelte';

// `displayValue` is the ONE echo-suppression decision that every live control shares
// (spec §3): given whether the user is actively editing, the latest backend `source`, and
// the in-progress `local` edit — which value should the control show? Extracted pure so the
// "value jumps under the cursor" fix (ParamField:29-37, generalised) is unit-tested once and
// every control opts into it via `useLiveValue` rather than re-deriving (and re-breaking) it.
describe('displayValue', () => {
	it('follows the backend source while idle (not editing)', () => {
		// Idle → the control tracks the live value; a fresh backend echo is shown.
		expect(displayValue(false, 7, 3)).toBe(7);
	});

	it('suppresses the backend echo while editing (shows the local edit)', () => {
		// Editing → the user's in-progress value wins, so a backend echo cannot yank the
		// value out from under the cursor.
		expect(displayValue(true, 7, 3)).toBe(3);
	});

	it('works across the value kinds a control carries', () => {
		expect(displayValue(false, 'live', 'typed')).toBe('live');
		expect(displayValue(true, 'live', 'typed')).toBe('typed');
		expect(displayValue(false, true, false)).toBe(true);
		expect(displayValue(true, true, false)).toBe(false);
	});

	it('a mid-edit source change is ignored, then followed once the edit ends (the bug it fixes)', () => {
		// Start idle, showing the backend value.
		let source = 5;
		let local = source;
		expect(displayValue(false, source, local)).toBe(5);

		// User begins editing and types 6; meanwhile the backend echoes 7.
		local = 6;
		source = 7;
		// The display stays on the user's 6 — it does NOT jump to the echoed 7.
		expect(displayValue(true, source, local)).toBe(6);

		// On commit/blur the latch releases; the display resumes following the source.
		expect(displayValue(false, source, local)).toBe(7);
	});

	it('is pure — identical inputs yield identical output', () => {
		expect(displayValue(true, 2, 9)).toBe(displayValue(true, 2, 9));
	});
});
