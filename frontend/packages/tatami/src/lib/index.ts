/**
 * tatami — a tab + panel workspace for Svelte.
 *
 * Modular mats laid out in fixed arrangements. Three components compose: `<Tabs>` alone is a tab
 * strip, `<Panels>` alone is a splittable panel tree, and `<Panels>` inside `<Tabs>` is a workspace
 * — the cross-boundary drags exist there and are UNEXPRESSIBLE anywhere else, because the drag bus
 * one publishes is what the other looks for.
 *
 * The package holds no tree and writes nothing. It draws the arrangement it is handed, recognises
 * the gestures over it, and raises each one through a [`LayoutHost`] — so whoever implements that
 * owns persistence, concurrency, undo and what a tab is CALLED. The panels themselves are the
 * consumer's too: `registerPanel` takes a component and an id, and nothing in here knows what any
 * of them draw.
 *
 * Styling is a namespaced custom-property contract — import `tatami/tokens.css` for the look it
 * ships with, then override `--tatami-*` to wear your own. See that file.
 */

// --- the workspace ------------------------------------------------------------------------------
export { default as Panels } from './WorkspaceView.svelte';
export { default as Tabs } from './WorkspaceTabs.svelte';
export { workspace, type DragRef, type LayoutIntent, type Viewpoint } from './workspace.svelte';

// --- the host: every gesture leaves through here -------------------------------------------------
export type { LayoutHost, TabHost, PanelHost, TabRef, Landing, TabsIn } from './host';
// …and one that needs nothing behind it, for a workspace with no persistence to answer to.
export { memoryHost, type MemoryHostOptions } from './memoryHost';

// --- the arrangement, as a consumer hands it over and reads it back ------------------------------
export {
	MIN_FRACTION,
	MIN_PANEL_PX,
	collectPanels,
	countPanels,
	findNode,
	findPanel,
	findParent,
	firstPanelId,
	firstPanelIn,
	resizeFractions,
	type Direction,
	type LayoutNode,
	type PanelNode,
	type SplitNode,
	type Workspace,
	type WorkspaceState
} from './model';
export { arrayToPath, asStateObject, linkedNodeName, pathToArray } from './panelState';

// --- the panel registry: the moddability seam ----------------------------------------------------
export {
	getPanelType,
	listPanelTypes,
	registerPanel,
	resolvePanelType,
	type PanelProps,
	type PanelType
} from './registry';

// --- the chrome ----------------------------------------------------------------------------------
// Exported because a consumer's panel CONTENT sits inside this chrome: a button in a panel toolbar
// that is not the same button as the one in the panel header reads as a foreign control two pixels
// away from a native one.
export { default as ContextMenu } from './ContextMenu.svelte';
export type { MenuItem } from './menu';
export {
	Button,
	IconButton,
	Icon,
	// `Tabs` is the WORKSPACE strip above — the one wired to the drag bus. This is the bare tab bar
	// it is built on, which a consumer also uses wherever it wants tabs that are not a workspace.
	Tabs as TabStrip,
	CHROME_ICONS,
	registerIcons,
	iconGeometry,
	type ButtonVariant,
	type ButtonSize,
	type ButtonDensity,
	type ChromeIconName,
	type TabItem
} from './ui';

// --- the leaf layer, exported for the same reason the chrome is ----------------------------------
export {
	portal,
	beginDrag,
	clampToViewport,
	overlayViewport,
	MARGIN,
	createLongPress,
	createWidthCache,
	planOverflow,
	HOLD_MS,
	MOVE_TOLERANCE_PX,
	type OverflowItem
} from './gesture';
