/**
 * Per-viewer settings schema — the single source of truth for which settings a
 * viewer kind exposes in its cog menu, their types, defaults, and grouping.
 *
 * The cog dropdown renders these (Blender-style collapsible groups); `resolveSettings`
 * merges a viewer instance's overrides over these defaults. Each viewer instance
 * owns its overrides via its ViewBinding (inline → node.viewers; panel → layout),
 * persisted into the .gfi.
 */
import type { ViewerKind } from './kind';
import { COLORMAPS } from './colormaps';

export type SettingValue = boolean | number | string;
export type SettingType = 'toggle' | 'select' | 'number';

export interface SettingDescriptor {
	key: string;
	label: string;
	type: SettingType;
	default: SettingValue;
	options?: string[]; // select
	min?: number;
	max?: number;
	step?: number; // number
	/** Only show this control when another setting holds a given value (e.g. the
	 * manual min/max only when "auto" is off). */
	showWhen?: { key: string; equals: SettingValue };
}

export interface SettingGroup {
	title: string;
	settings: SettingDescriptor[];
}

const SCHEMA: Record<ViewerKind, SettingGroup[]> = {
	line: [
		{
			title: 'Axes',
			settings: [
				{ key: 'logX', label: 'Log X', type: 'toggle', default: false },
				{ key: 'logY', label: 'Log Y', type: 'toggle', default: false }
			]
		},
		{
			title: 'Y range',
			settings: [
				{ key: 'yAuto', label: 'Auto', type: 'toggle', default: true },
				{ key: 'yMin', label: 'Min', type: 'number', default: -1, step: 0.1, showWhen: { key: 'yAuto', equals: false } },
				{ key: 'yMax', label: 'Max', type: 'number', default: 1, step: 0.1, showWhen: { key: 'yAuto', equals: false } }
			]
		},
		{ title: 'Style', settings: [{ key: 'points', label: 'Show points', type: 'toggle', default: false }] }
	],
	image: [
		{ title: 'Color', settings: [{ key: 'colormap', label: 'Colormap', type: 'select', default: 'gray', options: COLORMAPS }] },
		{
			title: 'Range',
			settings: [
				{ key: 'auto', label: 'Auto', type: 'toggle', default: true },
				{ key: 'vmin', label: 'Min', type: 'number', default: 0, step: 0.1, showWhen: { key: 'auto', equals: false } },
				{ key: 'vmax', label: 'Max', type: 'number', default: 1, step: 0.1, showWhen: { key: 'auto', equals: false } }
			]
		}
	],
	topomap: [
		{ title: 'Color', settings: [{ key: 'colormap', label: 'Colormap', type: 'select', default: 'coolwarm', options: COLORMAPS }] },
		{
			title: 'Range',
			settings: [
				{ key: 'auto', label: 'Auto', type: 'toggle', default: true },
				{ key: 'vmin', label: 'Min', type: 'number', default: -1, step: 0.1, showWhen: { key: 'auto', equals: false } },
				{ key: 'vmax', label: 'Max', type: 'number', default: 1, step: 0.1, showWhen: { key: 'auto', equals: false } }
			]
		},
		{ title: 'Style', settings: [{ key: 'contours', label: 'Contour lines', type: 'toggle', default: false }] }
	],
	trajectory: [
		{ title: 'Style', settings: [{ key: 'pointSize', label: 'Point size', type: 'number', default: 2, min: 0, max: 12, step: 1 }] },
		{ title: 'Range', settings: [{ key: 'auto', label: 'Auto', type: 'toggle', default: true }] }
	],
	table: [{ title: 'Format', settings: [{ key: 'decimals', label: 'Decimals', type: 'number', default: 3, min: 0, max: 10, step: 1 }] }],
	string: [
		{
			title: 'Render',
			settings: [
				{ key: 'markdown', label: 'Markdown', type: 'toggle', default: false },
				{ key: 'wrap', label: 'Word wrap', type: 'toggle', default: true }
			]
		}
	]
};

export function settingsSchemaFor(kind: ViewerKind): SettingGroup[] {
	return SCHEMA[kind] ?? [];
}

/** A bag of explicitly-set setting overrides (or resolved values). */
export type SettingsMap = Record<string, SettingValue>;

/** The default value bag for a kind — every declared key, its default. */
export function defaultSettings(kind: ViewerKind): SettingsMap {
	const out: SettingsMap = {};
	for (const g of settingsSchemaFor(kind)) for (const s of g.settings) out[s.key] = s.default;
	return out;
}

/** Resolved settings for a kind: its declared defaults with the explicit
 * overrides applied on top. */
export function resolveSettings(kind: ViewerKind, overrides: SettingsMap | undefined): SettingsMap {
	return { ...defaultSettings(kind), ...(overrides ?? {}) };
}
