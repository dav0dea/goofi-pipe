/** `/patch.gfi` — the patch as a file the BROWSER carries, reaching locations a containerised
 * backend cannot. A copy out and a copy in, never touching `savePath`. */

export const PATCH_FILE_URL = '/patch.gfi';

/** Ask the browser to save the open patch. A plain navigation, so the server names the file. */
export function downloadPatch(): void {
	window.location.href = PATCH_FILE_URL;
}

/** Replace the open patch with an archive the user picked. Rejects with the SERVER's message,
 * which names the real fault; a refused upload leaves the open patch untouched. */
export async function uploadPatch(file: Blob): Promise<void> {
	let res: Response;
	try {
		res = await fetch(PATCH_FILE_URL, { method: 'POST', body: file });
	} catch (e) {
		// Network-level: the server is gone, not the file is wrong.
		throw new Error(e instanceof Error ? e.message : String(e));
	}
	if (!res.ok) {
		const detail = (await res.text().catch(() => '')).trim();
		throw new Error(detail || `${res.status} ${res.statusText}`);
	}
}
