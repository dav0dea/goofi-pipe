/**
 * The panel system's OWN icon geometry — Lucide (https://lucide.dev), hand-vendored, and only the
 * glyphs its chrome draws: the header's caret, overflow and ✕, the split and maximize actions, the
 * menu's tick and submenu arrow, the tab strip's ✕ and ＋.
 *
 * ISC License · Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of
 * Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 * Path data copied verbatim from `lucide-react` v0.561.0 (`dist/esm/icons/<name>.js`).
 *
 * A consuming app draws icons of its OWN through this same renderer — a panel type's icon, a
 * context-menu row's — so it hands its table over with [`registerIcons`] and names the entries by
 * string. That is the whole extension seam: one renderer, one merged table, and no app geometry
 * vendored in here. Names collide by app-wins, so an app can also restyle a chrome glyph.
 *
 * WHAT IS HERE is the geometry ONLY: the drawing elements, with no paint of their own. The box
 * (24×24), the 2px round-capped stroke and `currentColor` live in `Icon.svelte`, the single
 * renderer — so an icon takes the colour of whatever control it sits in, by construction.
 */
export const CHROME_ICONS = {
	check: '<path d="M20 6 9 17l-5-5"/>',
	'chevron-down': '<path d="m6 9 6 6 6-6"/>',
	'chevron-right': '<path d="m9 18 6-6-6-6"/>',
	ellipsis:
		'<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
	'maximize-2':
		'<path d="M15 3h6v6"/><path d="m21 3-7 7"/><path d="m3 21 7-7"/><path d="M9 21H3v-6"/>',
	'minimize-2':
		'<path d="m14 10 7-7"/><path d="M20 10h-6V4"/><path d="m3 21 7-7"/><path d="M4 14h6v6"/>',
	plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
	'square-split-horizontal':
		'<path d="M8 19H5c-1 0-2-1-2-2V7c0-1 1-2 2-2h3"/><path d="M16 5h3c1 0 2 1 2 2v10c0 1-1 2-2 2h-3"/><line x1="12" x2="12" y1="4" y2="20"/>',
	'square-split-vertical':
		'<path d="M5 8V5c0-1 1-2 2-2h10c1 0 2 1 2 2v3"/><path d="M19 16v3c0 1-1 2-2 2H7c-1 0-2-1-2-2v-3"/><line x1="4" x2="20" y1="12" y2="12"/>',
	x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'
} as const;

/** The glyphs the panel chrome draws itself. An app\'s own names are plain strings. */
export type ChromeIconName = keyof typeof CHROME_ICONS;

/** An app's icon geometry, in the same shape and drawn by the same renderer. */
const registered: Record<string, string> = {};

/** Hand the renderer an app's icon set, once at startup. Called again, it MERGES — a set is a
 * contribution, not a replacement, so two callers cannot silently unregister each other. */
export function registerIcons(icons: Record<string, string>): void {
	Object.assign(registered, icons);
}

/** The geometry for `name`, or undefined — an app's set first, so a consumer can restyle a chrome
 * glyph without forking the chrome that draws it. */
export function iconGeometry(name: string): string | undefined {
	return registered[name] ?? (CHROME_ICONS as Record<string, string>)[name];
}
