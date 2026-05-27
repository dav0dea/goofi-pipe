/**
 * Thin-plate spline (TPS) interpolation for EEG topographic maps.
 *
 * Approximates MNE's `mne.viz.plot_topomap` look. MNE's default is
 * `image_interp="cubic"` (scipy `CloughTocher2DInterpolator` over a
 * Delaunay triangulation of the sensor positions, with an outer ring
 * of extrapolation points so the field decays smoothly to the head
 * boundary). MNE's own source notes that the previous TPS-RBF default
 * "looks about the same" — and TPS is much easier to ship in pure JS
 * without a Delaunay dependency.
 *
 * Strategy:
 *  - Add a ring of N_extra extrapolation points just outside the head
 *    circle, each carrying the mean of its K nearest real channels
 *    (matches MNE's `border="mean"`).
 *  - Fit f(x,y) = a₀ + a₁x + a₂y + Σ wᵢ φ(‖(x,y) - pᵢ‖) with
 *    φ(r) = r²·log(r) over the augmented point set.
 *  - Pre-invert the (n+3)×(n+3) augmented kernel-polynomial matrix
 *    once per layout, and pre-compute φ at every inside-circle pixel
 *    once per (layout, canvas size). The per-frame cost is then a
 *    single matvec for the weights plus an O(n·pixels) FMA sweep —
 *    no log() calls in the hot path.
 */

const TPS_EPS_SQ = 1e-12;

function tpsKernel(r2: number): number {
	if (r2 <= TPS_EPS_SQ) return 0;
	return 0.5 * r2 * Math.log(r2);
}

export interface TopoLayout {
	nReal: number;
	nExtra: number;
	posX: Float64Array;
	posY: Float64Array;
	/** For each extra ring point, indices into the real-channel array whose mean becomes its value. */
	extraNN: number[][];
	/** Inverse of the augmented matrix; (nTotal+3)² row-major. */
	Minv: Float64Array;
	layoutKey: string;
}

export interface PixelCache {
	width: number;
	height: number;
	layoutKey: string;
	count: number;
	/** Byte offset in ImageData.data for each inside-circle pixel. */
	pixelByteOffsets: Int32Array;
	/** Normalized x (in [0, 1]) of each inside-circle pixel. */
	px: Float32Array;
	/** Normalized y (in [0, 1]) of each inside-circle pixel. */
	py: Float32Array;
	/** Per-point φ image; length = nTotal, each Float32Array length = count. */
	kernels: Float32Array[];
}

function invertMatrix(m: Float64Array, n: number): Float64Array {
	// Gauss-Jordan elimination with partial pivoting on the augmented [M | I] matrix.
	const stride = 2 * n;
	const a = new Float64Array(n * stride);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) a[i * stride + j] = m[i * n + j];
		a[i * stride + n + i] = 1;
	}
	for (let i = 0; i < n; i++) {
		let pivotRow = i;
		let pivotVal = Math.abs(a[i * stride + i]);
		for (let k = i + 1; k < n; k++) {
			const v = Math.abs(a[k * stride + i]);
			if (v > pivotVal) {
				pivotVal = v;
				pivotRow = k;
			}
		}
		if (pivotVal < 1e-12) throw new Error('topomap TPS: singular matrix');
		if (pivotRow !== i) {
			for (let j = 0; j < stride; j++) {
				const t = a[i * stride + j];
				a[i * stride + j] = a[pivotRow * stride + j];
				a[pivotRow * stride + j] = t;
			}
		}
		const pv = a[i * stride + i];
		const invPv = 1 / pv;
		for (let j = 0; j < stride; j++) a[i * stride + j] *= invPv;
		for (let k = 0; k < n; k++) {
			if (k === i) continue;
			const f = a[k * stride + i];
			if (f === 0) continue;
			for (let j = 0; j < stride; j++) a[k * stride + j] -= f * a[i * stride + j];
		}
	}
	const out = new Float64Array(n * n);
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) out[i * n + j] = a[i * stride + n + j];
	}
	return out;
}

function buildRing(
	realX: Float64Array,
	realY: Float64Array,
	count: number,
	radius: number,
	k: number
): { x: Float64Array; y: Float64Array; nn: number[][] } {
	const x = new Float64Array(count);
	const y = new Float64Array(count);
	const nn: number[][] = new Array(count);
	const nReal = realX.length;
	const kk = Math.min(k, nReal);
	const buf: { i: number; d: number }[] = new Array(nReal);
	for (let i = 0; i < count; i++) {
		const theta = (i * 2 * Math.PI) / count - Math.PI / 2;
		x[i] = 0.5 + Math.cos(theta) * radius;
		y[i] = 0.5 + Math.sin(theta) * radius;
		for (let j = 0; j < nReal; j++) {
			const dx = realX[j] - x[i];
			const dy = realY[j] - y[i];
			buf[j] = { i: j, d: dx * dx + dy * dy };
		}
		buf.sort((a, b) => a.d - b.d);
		const lst: number[] = new Array(kk);
		for (let j = 0; j < kk; j++) lst[j] = buf[j].i;
		nn[i] = lst;
	}
	return { x, y, nn };
}

export function buildLayout(
	channelPositions: ReadonlyArray<readonly [number, number]>,
	layoutKey: string,
	opts: { extraCount?: number; extraRadius?: number; knnForBorder?: number } = {}
): TopoLayout | null {
	const nReal = channelPositions.length;
	if (nReal < 3) return null;
	const extraCount = opts.extraCount ?? 16;
	const extraRadius = opts.extraRadius ?? 0.52;
	const knn = opts.knnForBorder ?? 3;

	const realX = new Float64Array(nReal);
	const realY = new Float64Array(nReal);
	for (let i = 0; i < nReal; i++) {
		realX[i] = channelPositions[i][0];
		realY[i] = channelPositions[i][1];
	}
	const ring = buildRing(realX, realY, extraCount, extraRadius, knn);
	const nExtra = ring.nn.length;
	const nTotal = nReal + nExtra;

	const posX = new Float64Array(nTotal);
	const posY = new Float64Array(nTotal);
	for (let i = 0; i < nReal; i++) {
		posX[i] = realX[i];
		posY[i] = realY[i];
	}
	for (let i = 0; i < nExtra; i++) {
		posX[nReal + i] = ring.x[i];
		posY[nReal + i] = ring.y[i];
	}

	const dim = nTotal + 3;
	const M = new Float64Array(dim * dim);
	for (let i = 0; i < nTotal; i++) {
		for (let j = 0; j < nTotal; j++) {
			if (i === j) continue;
			const dx = posX[i] - posX[j];
			const dy = posY[i] - posY[j];
			M[i * dim + j] = tpsKernel(dx * dx + dy * dy);
		}
		M[i * dim + nTotal + 0] = 1;
		M[i * dim + nTotal + 1] = posX[i];
		M[i * dim + nTotal + 2] = posY[i];
		M[(nTotal + 0) * dim + i] = 1;
		M[(nTotal + 1) * dim + i] = posX[i];
		M[(nTotal + 2) * dim + i] = posY[i];
	}

	const Minv = invertMatrix(M, dim);
	return { nReal, nExtra, posX, posY, extraNN: ring.nn, Minv, layoutKey };
}

export function buildPixelCache(layout: TopoLayout, w: number, h: number): PixelCache {
	const cx = w / 2;
	const cy = h / 2;
	const radius = Math.min(w, h) * 0.45;
	const r2 = radius * radius;
	let count = 0;
	for (let y = 0; y < h; y++) {
		const dy = y - cy;
		for (let x = 0; x < w; x++) {
			const dx = x - cx;
			if (dx * dx + dy * dy <= r2) count++;
		}
	}
	const pixelByteOffsets = new Int32Array(count);
	const px = new Float32Array(count);
	const py = new Float32Array(count);
	let idx = 0;
	for (let y = 0; y < h; y++) {
		const dy = y - cy;
		for (let x = 0; x < w; x++) {
			const dx = x - cx;
			if (dx * dx + dy * dy <= r2) {
				pixelByteOffsets[idx] = (y * w + x) * 4;
				px[idx] = x / w;
				py[idx] = y / h;
				idx++;
			}
		}
	}
	const nTotal = layout.nReal + layout.nExtra;
	const kernels: Float32Array[] = new Array(nTotal);
	for (let i = 0; i < nTotal; i++) {
		const arr = new Float32Array(count);
		const cxi = layout.posX[i];
		const cyi = layout.posY[i];
		for (let p = 0; p < count; p++) {
			const dx = px[p] - cxi;
			const dy = py[p] - cyi;
			arr[p] = tpsKernel(dx * dx + dy * dy);
		}
		kernels[i] = arr;
	}
	return { width: w, height: h, layoutKey: layout.layoutKey, count, pixelByteOffsets, px, py, kernels };
}

export function solveWeights(layout: TopoLayout, realVals: ArrayLike<number>): Float64Array {
	const { nReal, nExtra, extraNN, Minv } = layout;
	const nTotal = nReal + nExtra;
	const dim = nTotal + 3;
	const rhs = new Float64Array(dim);
	for (let i = 0; i < nReal; i++) rhs[i] = realVals[i];
	for (let i = 0; i < nExtra; i++) {
		const nn = extraNN[i];
		let s = 0;
		for (let j = 0; j < nn.length; j++) s += realVals[nn[j]];
		rhs[nReal + i] = s / nn.length;
	}
	const w = new Float64Array(dim);
	for (let i = 0; i < dim; i++) {
		let s = 0;
		for (let j = 0; j < dim; j++) s += Minv[i * dim + j] * rhs[j];
		w[i] = s;
	}
	return w;
}

export function evaluateField(
	layout: TopoLayout,
	cache: PixelCache,
	weights: Float64Array,
	out: Float32Array
): void {
	const nTotal = layout.nReal + layout.nExtra;
	const a0 = weights[nTotal + 0];
	const a1 = weights[nTotal + 1];
	const a2 = weights[nTotal + 2];
	const count = cache.count;
	const px = cache.px;
	const py = cache.py;
	for (let p = 0; p < count; p++) out[p] = a0 + a1 * px[p] + a2 * py[p];
	const kernels = cache.kernels;
	for (let i = 0; i < nTotal; i++) {
		const wi = weights[i];
		if (wi === 0) continue;
		const k = kernels[i];
		for (let p = 0; p < count; p++) out[p] += wi * k[p];
	}
}
