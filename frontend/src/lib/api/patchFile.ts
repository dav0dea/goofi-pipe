/**
 * `/patch.gfi` — the patch as a file the BROWSER carries, in both directions.
 *
 * The door onto locations the backend cannot reach. Running in a container, goofi sees only what
 * was bind-mounted, and no `docker run` flag means "the whole host filesystem" on Linux, macOS and
 * Windows alike. The browser has no such limit: it runs on the host, and its own Save/Open dialogs
 * reach anywhere the user can.
 *
 * This is a copy out and a copy in — NOT a second save. It never touches `savePath`, so Ctrl-S
 * keeps meaning "overwrite the file this patch came from" instead of silently retargeting a
 * download. That distinction is why an earlier browser-download save was removed from goofi.
 */

/** The one place the route is spelled. */
export const PATCH_FILE_URL = '/patch.gfi';

/**
 * Ask the browser to save the open patch.
 *
 * A plain navigation, not a `fetch` + blob: the response carries
 * `Content-Disposition: attachment`, so the browser downloads it and the page never leaves. That
 * also means the filename comes from the server — the patch's own name when it has one — rather
 * than from a guess made here.
 *
 * Untested by design: it is one assignment to `location`, and this project runs vitest without a
 * DOM. What IS testable lives in `uploadPatch` below.
 */
export function downloadPatch(): void {
	window.location.href = PATCH_FILE_URL;
}

/**
 * Replace the open patch with an archive the user picked on their own machine.
 *
 * Rejects with the SERVER's message, because that message is the useful one: "archive has no
 * patch.yaml" tells a user they picked the wrong file, where a generic "upload failed" sends them
 * looking for a network problem they do not have. A refused upload leaves the open patch untouched
 * — the backend's `load` arm guarantees that, and it is why this can report and do nothing else.
 */
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
