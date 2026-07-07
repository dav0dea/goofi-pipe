/**
 * The options to render for a string dropdown: the node-declared options, with
 * the current value prepended when it isn't among them.
 *
 * A StringParam's `options` are structural config (never persisted), so a saved
 * value can be absent from the reloaded option list — a chosen LSL source, or an
 * audio device that's since unplugged. A `<select value={x}>` whose `<option>`s
 * don't include `x` renders BLANK (selectedIndex = -1), so the dropdown would
 * look unset while the node is actually bound to `x`. Keeping the active value
 * renderable (⟳ then refreshes to the live list) mirrors the node-side
 * REFRESH_PARAM handler's "keep the current value selectable" rule.
 */
export function selectOptions(options: string[], value: string): string[] {
	if (value && !options.includes(value)) return [value, ...options];
	return options;
}
