/**
 * One viewer, many instances: a ViewBinding is the per-instance view state
 * (kind + settings) behind a viewer. The shared viewer components consume a
 * binding and never read a store directly, so the inline node viewer and every
 * docked panel are the SAME component differing only in where their binding
 * persists. The data source (node, slot, dtype) is separate and shared.
 *
 * `panelBinding` (a docked Viewer panel, backed by its layout state) lives here
 * as a reusable, rune-free factory — it's unit-tested. The inline binding is
 * constructed at its single use site (SlotViewer), where the node-scoped
 * inline-view store and graph.pushNodeViewers are already in scope; keeping that
 * rune coupling out of this module is what lets the factory be tested in
 * isolation.
 */
import { resolveKind, type ViewerKind } from './kind';
import { resolveSettings, type SettingValue, type SettingsMap } from './settingsSchema';
import { asStateObject } from '$lib/workspace/panelState';

export interface ViewBinding {
	readonly kind: ViewerKind;
	readonly settings: SettingsMap;
	setKind(kind: ViewerKind): void;
	setSetting(key: string, value: SettingValue): void;
}

/** Docked Viewer panel: backed by the panel's own state, which the MANAGER holds. Each setter names
 * its own undo step — the write is one `page_set_panel` command, so the manager owns the inverse and
 * the label is all the client has to supply. */
export function panelBinding(
	getState: () => unknown,
	setState: (s: unknown, label: string) => void,
	dtype: string | null
): ViewBinding {
	const raw = () => asStateObject(getState());
	const rawSettings = (): SettingsMap => (raw().settings as SettingsMap) ?? {};
	return {
		get kind() {
			return resolveKind(dtype, raw().kind as ViewerKind | undefined);
		},
		get settings() {
			return resolveSettings(this.kind, rawSettings());
		},
		setKind(kind) {
			setState({ ...raw(), kind }, `Viewer → ${kind}`);
		},
		setSetting(key, value) {
			setState({ ...raw(), settings: { ...rawSettings(), [key]: value } }, `Viewer ${key}`);
		}
	};
}
