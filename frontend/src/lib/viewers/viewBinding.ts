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
// Relative (not $lib) so this rune-free module resolves under vitest, which
// doesn't honour the SvelteKit $lib alias for runtime imports.
import { asStateObject } from '../workspace/panelState';

export interface ViewBinding {
	readonly kind: ViewerKind;
	readonly settings: SettingsMap;
	setKind(kind: ViewerKind): void;
	setSetting(key: string, value: SettingValue): void;
}

/** Docked Viewer panel: backed by the panel's own layout state blob. */
export function panelBinding(
	getState: () => unknown,
	setState: (s: unknown) => void,
	dtype: string | null
): ViewBinding {
	const raw = () => asStateObject(getState());
	return {
		get kind() {
			return resolveKind(dtype, raw().kind as ViewerKind | undefined);
		},
		get settings() {
			return resolveSettings(this.kind, (raw().settings as SettingsMap) ?? {});
		},
		setKind(kind) {
			setState({ ...raw(), kind });
		},
		setSetting(key, value) {
			setState({
				...raw(),
				settings: { ...((raw().settings as SettingsMap) ?? {}), [key]: value }
			});
		}
	};
}
