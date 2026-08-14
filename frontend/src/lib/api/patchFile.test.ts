import { describe, expect, it, vi, afterEach } from 'vitest';
import { PATCH_FILE_URL, uploadPatch } from './patchFile';

/** Stand in for `fetch`, recording what it was handed. */
function stubFetch(res: Partial<Response> & { text?: () => Promise<string> }) {
	const calls: Array<[string, RequestInit]> = [];
	vi.stubGlobal('fetch', (url: string, init: RequestInit) => {
		calls.push([url, init]);
		return Promise.resolve({ text: () => Promise.resolve(''), ...res } as Response);
	});
	return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe('uploadPatch', () => {
	it('POSTs the file itself, unwrapped', async () => {
		const calls = stubFetch({ ok: true });
		const file = new Blob(['PK']);
		await uploadPatch(file);

		expect(calls).toHaveLength(1);
		const [url, init] = calls[0];
		expect(url).toBe(PATCH_FILE_URL);
		expect(init.method).toBe('POST');
		// The BODY is the blob, not a FormData wrapper. The route reads the request body as the
		// archive bytes, so a multipart envelope would arrive as a zip with a MIME preamble glued
		// to the front of it — which fails as "not a zip archive", pointing at the user's file.
		expect(init.body).toBe(file);
	});

	/** The whole reason this function exists rather than a bare `fetch` at the call site. */
	it("rejects with the SERVER's reason, so a wrong file says so", async () => {
		stubFetch({
			ok: false,
			status: 400,
			statusText: 'Bad Request',
			text: () => Promise.resolve('load failed: archive has no patch.yaml\n')
		});
		await expect(uploadPatch(new Blob(['nope']))).rejects.toThrow('archive has no patch.yaml');
	});

	it('falls back to the status line when the server explains nothing', async () => {
		stubFetch({ ok: false, status: 500, statusText: 'Internal Server Error', text: () => Promise.resolve('  ') });
		await expect(uploadPatch(new Blob(['x']))).rejects.toThrow('500 Internal Server Error');
	});

	it('surfaces a transport failure as itself', async () => {
		vi.stubGlobal('fetch', () => Promise.reject(new Error('Failed to fetch')));
		await expect(uploadPatch(new Blob(['x']))).rejects.toThrow('Failed to fetch');
	});
});
