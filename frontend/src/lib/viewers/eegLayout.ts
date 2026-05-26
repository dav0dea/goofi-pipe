/**
 * Canonical EEG 10-20 / 10-10 electrode layout, normalized to [0, 1] × [0, 1].
 *
 * The original dearpygui topomap pulled this from MNE (`channels.read_layout("EEG1005")`).
 * Reproducing the full 1005 layout in JS is overkill; a curated 10-20 set
 * covers every common cap. Coordinates are azimuthal-equidistant projections
 * of the canonical 10-20 grid, with origin shifted to (0.5, 0.5) and scaled
 * so the head circle fits within the canvas.
 */
export const EEG_LAYOUT: Record<string, [number, number]> = {
	// midline
	Fpz: [0.5, 0.05],
	Fz: [0.5, 0.27],
	Cz: [0.5, 0.5],
	Pz: [0.5, 0.73],
	Oz: [0.5, 0.95],
	// frontal
	Fp1: [0.35, 0.07],
	Fp2: [0.65, 0.07],
	AF3: [0.36, 0.18],
	AF4: [0.64, 0.18],
	AF7: [0.27, 0.13],
	AF8: [0.73, 0.13],
	F3: [0.33, 0.27],
	F4: [0.67, 0.27],
	F5: [0.27, 0.27],
	F6: [0.73, 0.27],
	F7: [0.17, 0.25],
	F8: [0.83, 0.25],
	F1: [0.42, 0.27],
	F2: [0.58, 0.27],
	// central
	C3: [0.27, 0.5],
	C4: [0.73, 0.5],
	C5: [0.16, 0.5],
	C6: [0.84, 0.5],
	C1: [0.39, 0.5],
	C2: [0.61, 0.5],
	FC3: [0.31, 0.38],
	FC4: [0.69, 0.38],
	FC5: [0.22, 0.38],
	FC6: [0.78, 0.38],
	FC1: [0.41, 0.38],
	FC2: [0.59, 0.38],
	T7: [0.07, 0.5],
	T8: [0.93, 0.5],
	T3: [0.07, 0.5],
	T4: [0.93, 0.5],
	// parietal
	CP3: [0.31, 0.62],
	CP4: [0.69, 0.62],
	CP5: [0.22, 0.62],
	CP6: [0.78, 0.62],
	CP1: [0.41, 0.62],
	CP2: [0.59, 0.62],
	P3: [0.33, 0.73],
	P4: [0.67, 0.73],
	P5: [0.27, 0.73],
	P6: [0.73, 0.73],
	P7: [0.17, 0.75],
	P8: [0.83, 0.75],
	P1: [0.42, 0.73],
	P2: [0.58, 0.73],
	// occipital
	O1: [0.4, 0.92],
	O2: [0.6, 0.92],
	PO3: [0.36, 0.83],
	PO4: [0.64, 0.83],
	PO7: [0.27, 0.87],
	PO8: [0.73, 0.87],
	POz: [0.5, 0.83],
	// extras
	T5: [0.17, 0.75],
	T6: [0.83, 0.75],
	A1: [0.0, 0.5],
	A2: [1.0, 0.5]
};
