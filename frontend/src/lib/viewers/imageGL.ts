/** WebGL2 image renderer; `tryCreate` returns null so the caller can degrade to 2D. */
import { isFloatDtype } from '$lib/codec/decode';

const VERT = `#version 300 es
const vec2 verts[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 v_uv;
void main() {
	vec2 p = verts[gl_VertexID];
	// uv in [0,1], Y flipped so image row 0 renders at the top.
	v_uv = vec2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
	gl_Position = vec4(p, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform sampler2D u_lut;
uniform int u_mode;          // 0 = grayscale+colormap, 1 = rgb, 2 = rgba
uniform float u_lo;
uniform float u_span;
out vec4 outColor;
void main() {
	if (u_mode == 0) {
		float g = texture(u_tex, v_uv).r;
		float t = clamp((g - u_lo) / u_span, 0.0, 1.0);
		outColor = vec4(texture(u_lut, vec2(t, 0.5)).rgb, 1.0);
	} else if (u_mode == 1) {
		outColor = vec4(texture(u_tex, v_uv).rgb, 1.0);
	} else {
		outColor = texture(u_tex, v_uv);
	}
}`;

export interface RenderOpts {
	colormap: string;
	lut: Uint8Array;
	lo: number;
	hi: number;
	cssW: number;
	cssH: number;
}

/** Whether the GL path supports a given (channels, dtype); gray+alpha (c === 2) falls to 2D.
 * RGB32F is neither color-renderable nor filterable, which is fine — this renderer only samples. */
export function glSupports(c: number, dtype: string): boolean {
	if (!isFloatDtype(dtype)) return false;
	return c === 1 || c === 3 || c === 4;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
	const sh = gl.createShader(type);
	if (!sh) return null;
	gl.shaderSource(sh, src);
	gl.compileShader(sh);
	if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
		gl.deleteShader(sh);
		return null;
	}
	return sh;
}

export class GLImageRenderer {
	private gl: WebGL2RenderingContext;
	private prog: WebGLProgram;
	private tex: WebGLTexture;
	private lutTex: WebGLTexture;
	private uMode: WebGLUniformLocation | null;
	private uLo: WebGLUniformLocation | null;
	private uSpan: WebGLUniformLocation | null;
	private floatLinear: boolean;
	private lutName = '';

	private constructor(gl: WebGL2RenderingContext, prog: WebGLProgram) {
		this.gl = gl;
		this.prog = prog;
		this.tex = gl.createTexture()!;
		this.lutTex = gl.createTexture()!;
		this.floatLinear = gl.getExtension('OES_texture_float_linear') !== null;
		gl.useProgram(prog);
		this.uMode = gl.getUniformLocation(prog, 'u_mode');
		this.uLo = gl.getUniformLocation(prog, 'u_lo');
		this.uSpan = gl.getUniformLocation(prog, 'u_span');
		gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
		gl.uniform1i(gl.getUniformLocation(prog, 'u_lut'), 1);
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	static tryCreate(canvas: HTMLCanvasElement): GLImageRenderer | null {
		const gl = canvas.getContext('webgl2', { antialias: false, depth: false, premultipliedAlpha: false });
		if (!gl) return null;
		const vs = compile(gl, gl.VERTEX_SHADER, VERT);
		const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
		if (!vs || !fs) return null;
		const prog = gl.createProgram();
		if (!prog) return null;
		gl.attachShader(prog, vs);
		gl.attachShader(prog, fs);
		gl.linkProgram(prog);
		if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
			gl.deleteProgram(prog);
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return null;
		}
		return new GLImageRenderer(gl, prog);
	}

	render(values: ArrayLike<number>, w: number, h: number, c: number, opts: RenderOpts): void {
		const gl = this.gl;
		const canvas = gl.canvas as HTMLCanvasElement;
		// Fit to the CSS box, never upscaling: the GPU does the HD downsample, CSS letterboxes.
		const fit = Math.min(opts.cssW / w, opts.cssH / h, 1) || 1;
		const dw = Math.max(1, Math.round(w * fit));
		const dh = Math.max(1, Math.round(h * fit));
		if (canvas.width !== dw || canvas.height !== dh) {
			canvas.width = dw;
			canvas.height = dh;
		}

		const mode = c === 1 ? 0 : c === 3 ? 1 : 2;
		const internal = c === 1 ? gl.R32F : c === 3 ? gl.RGB32F : gl.RGBA32F;
		const format = c === 1 ? gl.RED : c === 3 ? gl.RGB : gl.RGBA;

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.tex);
		const filter = this.floatLinear ? gl.LINEAR : gl.NEAREST;
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.FLOAT, values as unknown as ArrayBufferView);

		if (mode === 0) {
			if (this.lutName !== opts.colormap) {
				this.lutName = opts.colormap;
				gl.activeTexture(gl.TEXTURE1);
				gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
				gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, opts.lut);
			}
			const span = opts.hi - opts.lo || 1;
			gl.useProgram(this.prog);
			gl.uniform1f(this.uLo, opts.lo);
			gl.uniform1f(this.uSpan, span);
		}
		gl.useProgram(this.prog);
		gl.uniform1i(this.uMode, mode);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.tex);
		gl.viewport(0, 0, dw, dh);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	}

	dispose(): void {
		const gl = this.gl;
		gl.deleteTexture(this.tex);
		gl.deleteTexture(this.lutTex);
		gl.deleteProgram(this.prog);
	}
}
