/**
 * The panel system's own chrome primitives — the pieces its header, menu and tab strip are built
 * from, and the ONE icon renderer they and the app both draw through.
 *
 * They are exported rather than kept private because a consumer's panel CONTENT sits inside this
 * chrome: a button in a panel toolbar that is not the same button as the one in the panel header
 * reads as a foreign control two pixels away from a native one. A consuming app takes these and
 * has one set, not two that must agree.
 *
 * A leaf, like `$lib/gesture` below it: no store, no host, nothing from the app.
 */
export { default as Button } from './Button.svelte';
export { default as IconButton } from './IconButton.svelte';
export { type ButtonVariant, type ButtonSize, type ButtonDensity } from './variantClass';
export { default as Icon } from './Icon.svelte';
export { CHROME_ICONS, registerIcons, iconGeometry, type ChromeIconName } from './icons';
export { default as Tabs } from './Tabs.svelte';
export { type TabItem } from './tabsState';
