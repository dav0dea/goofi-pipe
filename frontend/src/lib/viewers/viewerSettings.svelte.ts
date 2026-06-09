/**
 * Per-(node, slot) viewer settings — the values behind each slot's cog menu.
 *
 * Mirrors the `viewerState` (kind) store: hoisted out of the component so the
 * cog menu, the viewer component, a restored patch, and the agent surface all
 * read/write one place. Only explicitly-set values are stored; reads merge them
 * over the viewer-kind defaults so every declared key always resolves.
 */
import type { ViewerKind } from './kind';
import { defaultSettings, type SettingValue } from './settingsSchema';

export type SettingsMap = Record<string, SettingValue>;

function key(node: string, slot: string): string {
	return `${node}|${slot}`;
}

const store = $state<Record<string, SettingsMap>>({});

/** Resolved settings for a slot's current viewer kind (defaults + overrides). */
export function viewerSettings(node: string, slot: string, kind: ViewerKind): SettingsMap {
	return { ...defaultSettings(kind), ...(store[key(node, slot)] ?? {}) };
}

/** The explicitly-set overrides only (what gets persisted). */
export function rawViewerSettings(node: string, slot: string): SettingsMap {
	return store[key(node, slot)] ?? {};
}

export function setViewerSetting(node: string, slot: string, k: string, value: SettingValue): void {
	const id = key(node, slot);
	store[id] = { ...(store[id] ?? {}), [k]: value };
}

/** Seed a slot's overrides from a restored patch. */
export function seedViewerSettings(node: string, slot: string, settings: SettingsMap | undefined): void {
	if (settings && Object.keys(settings).length > 0) store[key(node, slot)] = { ...settings };
}

/** Drop every slot's settings for a node that no longer exists. */
export function forgetViewerSettings(node: string): void {
	const prefix = `${node}|`;
	for (const k of Object.keys(store)) if (k.startsWith(prefix)) delete store[k];
}
