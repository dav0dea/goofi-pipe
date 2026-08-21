/**
 * A ViewBinding is one viewer instance's kind + settings; the components read nothing else.
 * The inline binding is built at its single use site (SlotViewer), which needs runes.
 */
import { resolveKind, type ViewerKind } from './kind';
import { resolveSettings, type SettingValue, type SettingsMap } from './settingsSchema';
import { asStateObject } from 'panelty';

export interface ViewBinding {
	readonly kind: ViewerKind;
	readonly settings: SettingsMap;
	setKind(kind: ViewerKind): void;
	setSetting(key: string, value: SettingValue): void;
}

/** Docked Viewer panel binding, backed by the panel state the manager holds. */
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
