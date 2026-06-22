import type { FsEntry } from '$lib/api/control';
import type { MenuItem } from '$lib/workspace/menu';

/** Build the Examples dropdown items from the backend's example-dir listing: one
 * entry per .gfi file, labelled without the extension, loading its path on click
 * (backlog #11). A disabled placeholder is shown when there are none. */
export function examplesToMenuItems(
	entries: FsEntry[],
	onLoad: (path: string) => void
): MenuItem[] {
	const files = entries.filter((e) => e.kind === 'file' && e.is_gfi);
	if (files.length === 0) return [{ label: '(no examples found)', disabled: true }];
	return files.map((e) => ({
		label: e.name.replace(/\.gfi$/i, ''),
		action: () => onLoad(e.path)
	}));
}
