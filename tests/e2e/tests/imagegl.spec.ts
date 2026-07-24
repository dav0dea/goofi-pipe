import { test, expect } from '@playwright/test';

/**
 * The GL image path's texture-format contract, checked against a REAL browser.
 *
 * `glSupports` (frontend/src/lib/viewers/imageGL.ts) routes float RGB to the GPU on the grounds
 * that RGB32F is a required sized internal format for textures in WebGL2 — it is only
 * non-renderable and non-filterable, neither of which the renderer needs. That claim is a
 * platform fact a unit test cannot check, and getting it wrong is expensive: RGB is HD video, and
 * the fallback is a w*h*3-iteration JS loop on the main thread. So exercise the exact upload the
 * renderer performs and assert the driver accepted it.
 */
test('the GL image path can upload float grayscale, RGB and RGBA textures', async ({ page }) => {
	await page.goto('/');
	const r = await page.evaluate(() => {
		const gl = document.createElement('canvas').getContext('webgl2');
		if (!gl) return null;
		gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		// One 2x2 upload per channel count, mirroring GLImageRenderer.render's format ladder.
		const upload = (internal: number, format: number, c: number) => {
			gl.texImage2D(gl.TEXTURE_2D, 0, internal, 2, 2, 0, format, gl.FLOAT, new Float32Array(4 * c));
			return gl.getError();
		};
		const gray = upload(gl.R32F, gl.RED, 1);
		const rgb = upload(gl.RGB32F, gl.RGB, 3);
		const rgba = upload(gl.RGBA32F, gl.RGBA, 4);
		// NEAREST is always legal for a float texture; the renderer only asks for LINEAR when
		// OES_texture_float_linear is present.
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		return { gray, rgb, rgba, filter: gl.getError(), NO_ERROR: gl.NO_ERROR };
	});
	expect(r, 'WebGL2 must be available in the test browser').not.toBeNull();
	expect(
		[r!.gray, r!.rgb, r!.rgba, r!.filter],
		'every float texture format the renderer uploads must be accepted'
	).toEqual([r!.NO_ERROR, r!.NO_ERROR, r!.NO_ERROR, r!.NO_ERROR]);
});
