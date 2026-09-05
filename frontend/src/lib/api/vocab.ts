// GENERATED from backend/goofi-bridge/src/vocab.rs — do not edit by hand.
// Panel types and viewer kinds are declared ONCE, in the manager: naming one that is not
// in the table is a type error here and a teachable refusal there. The BEHAVIOUR keyed off
// a word stays client-side — which component renders a panel type (`panels/register.ts`),
// and whether a particular array draws (`viewers/kind.ts`). Regenerate by running
// `cargo test -p goofi-bridge`, which rewrites this file when it drifts.
import type { IconName } from '$lib/ui';

export type PanelTypeId =
	| 'empty'
	| 'node-editor'
	| 'parameters'
	| 'viewer'
	| 'metadata'
	| 'console'
	| 'globals'
	| 'agent';

/** The panel type a brand-new tab starts with. */
export const DEFAULT_PANEL_TYPE = 'node-editor';
/** The placeholder a split births, whose in-panel grid chooses its content. */
export const EMPTY_PANEL_TYPE = 'empty';

export type ViewerKind =
	| 'line'
	| 'image'
	| 'trajectory'
	| 'topomap'
	| 'string'
	| 'table';

export interface PanelTypeInfo {
	readonly id: PanelTypeId;
	readonly title: string;
	readonly icon: IconName;
	/** Whether a node dragged onto this panel binds to it (`state.node`). */
	readonly acceptsNode: boolean;
	readonly doc: string;
}

/** The slot kinds, as the manager names them. */
export type SlotDtype = 'ARRAY' | 'STRING' | 'TABLE' | 'AUDIO';

export interface ViewerKindInfo {
	readonly id: ViewerKind;
	/** The slot dtype this kind serves. A non-ARRAY dtype PINS its kind: a STRING slot is
	 * always drawn by the string viewer, whatever kind was stored. */
	readonly dtype: SlotDtype;
	/** The dimension range the component actually renders — null for a pinned kind. */
	readonly draws: readonly [number, number] | null;
	/** The dimension range its ViewSpec declares compatible: equal or wider than `draws`,
	 * so a frame the viewer only summarises still arrives reduced. */
	readonly accepts: readonly [number, number] | null;
	readonly doc: string;
}

export const PANEL_TYPES: readonly PanelTypeInfo[] = [
	{ id: 'empty', title: 'Empty', icon: 'square-dashed', acceptsNode: false, doc: 'a placeholder with no content yet — what a fresh split births' },
	{ id: 'node-editor', title: 'Node Editor', icon: 'workflow', acceptsNode: false, doc: 'the patch canvas — nodes, wires and sub-patches' },
	{ id: 'parameters', title: 'Parameters', icon: 'sliders-horizontal', acceptsNode: true, doc: 'the parameters of one node, with ranges and expression bindings' },
	{ id: 'viewer', title: 'Viewer', icon: 'activity', acceptsNode: true, doc: 'live frames from one output slot, drawn by `state.kind`' },
	{ id: 'metadata', title: 'Metadata', icon: 'info', acceptsNode: true, doc: 'frame metadata from one output slot (sfreq, channels, shape)' },
	{ id: 'console', title: 'Console', icon: 'terminal', acceptsNode: true, doc: 'the patch log; a bound node filters it to that node' },
	{ id: 'globals', title: 'Globals', icon: 'globe', acceptsNode: false, doc: 'the patch globals, which any expression can read' },
	{ id: 'agent', title: 'Agent', icon: 'bot', acceptsNode: false, doc: 'a terminal on an agent harness, running in the patch workspace' },
];

export const VIEWER_KINDS: readonly ViewerKindInfo[] = [
	{ id: 'line', dtype: 'ARRAY', draws: [0, 2], accepts: [0, 3], doc: 'a time plot: one series (1-D) or one per channel (C, N)' },
	{ id: 'image', dtype: 'ARRAY', draws: [2, 3], accepts: [2, 3], doc: 'a bitmap: (H, W), or (H, W, C) for 1–4 channels' },
	{ id: 'trajectory', dtype: 'ARRAY', draws: [2, 2], accepts: [2, 2], doc: 'a phase portrait over pairs of rows of a (D, N) frame' },
	{ id: 'topomap', dtype: 'ARRAY', draws: [1, 1], accepts: [1, 1], doc: 'a scalp map of one scalar per channel' },
	{ id: 'string', dtype: 'STRING', draws: null, accepts: null, doc: 'the text of a STRING slot' },
	{ id: 'table', dtype: 'TABLE', draws: null, accepts: null, doc: 'the rows of a TABLE slot' },
];

/** The closed vocabulary a node's `tags` come from — the palette facets by it. */
export const TAGS = ['input', 'output', 'generator', 'transform', 'analysis', 'control', 'image', 'text', 'midi', 'eeg', 'cardio', 'motion', 'music', 'ml', 'connectivity', 'simulation'] as const;

/** The type a sub-patch facade wears in the document. It is not in the palette — grouping
* is what makes one. */
export const SCOPE_TYPE = 'SubPatch';

/** The one slot a boundary port carries. */
export const BOUNDARY_SLOT = 'value';

export interface BoundaryTypeInfo {
	readonly type: string;
	/** An `in` port FEEDS the sub-patch, so it wears an output and is a link's SOURCE. */
	readonly dir: 'in' | 'out';
	readonly dtype: SlotDtype;
}

/** The boundary port types: a port's direction and dtype ARE its type. */
export const BOUNDARY_TYPES: readonly BoundaryTypeInfo[] = [
	{ type: 'InArray', dir: 'in', dtype: 'ARRAY' },
	{ type: 'InString', dir: 'in', dtype: 'STRING' },
	{ type: 'InTable', dir: 'in', dtype: 'TABLE' },
	{ type: 'InAudio', dir: 'in', dtype: 'AUDIO' },
	{ type: 'OutArray', dir: 'out', dtype: 'ARRAY' },
	{ type: 'OutString', dir: 'out', dtype: 'STRING' },
	{ type: 'OutTable', dir: 'out', dtype: 'TABLE' },
	{ type: 'OutAudio', dir: 'out', dtype: 'AUDIO' },
];

export const boundaryType = (type: string): BoundaryTypeInfo | undefined =>
	BOUNDARY_TYPES.find((b) => b.type === type);

/** Which output kind may feed which input kind — the manager's one link rule, projected. */
export const FEEDS: ReadonlySet<string> = new Set(['ARRAY>ARRAY', 'STRING>STRING', 'TABLE>TABLE', 'AUDIO>ARRAY', 'AUDIO>AUDIO']);
export const feeds = (out: SlotDtype, into: SlotDtype): boolean => FEEDS.has(`${out}>${into}`);
