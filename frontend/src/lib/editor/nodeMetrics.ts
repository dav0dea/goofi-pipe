/**
 * Node geometry constants — these mirror the `--node-*` CSS custom properties in
 * `app.css`. They live here in JS too so the connector overlay can compute each
 * output port's y-position (which depends on how the output slots stack)
 * deterministically, without measuring the DOM.
 *
 * A node is drawn as two layers: a clipped visual `surface` (rounds the corners)
 * and an unclipped `ports` overlay (the connector pills, which overhang the
 * edges). The output handles can't be inside the clipped surface, so the overlay
 * positions them here using the same rhythm the SlotViewers lay out with:
 * header, then each slot — one unit tall collapsed, `unit + viewer` open.
 */
export const NODE = {
	width: 233,
	header: 36,
	unit: 24, // collapsed slot height === input connector pitch
	viewer: 144, // open viewer plot height
	border: 1
} as const;
