/** Wire shapes for parameter descriptors as the bridge emits them. */

/** The one active source of a param's value. */
export const PARAM_MODES = ['constant', 'expression', 'reference'] as const;
export type ParamMode = (typeof PARAM_MODES)[number];

/** What `node param edit` takes beside a value: any subset, and a text given implies its mode. */
export interface SourcePatch {
	mode?: ParamMode;
	expression?: string;
	reference?: string;
	triggers?: boolean;
}

export interface BaseParam {
	value: unknown;
	doc: string | null;
	/** True when the node declared a refresh method for this param. */
	refreshable: boolean;
	mode: ParamMode;
	/** The retained expression text, whatever the mode; null when there is none. */
	expression: string | null;
	/** The retained `node.slot`, whatever the mode; null when there is none. */
	reference: string | null;
	/** When true, an arrival that changes the value wakes the node's `process()`. */
	triggers: boolean;
	/** The active source's bind, compile or arrival error, or null. */
	error: string | null;
}

export interface FloatParam extends BaseParam {
	type: 'float';
	value: number;
	vmin: number;
	vmax: number;
}

export interface IntParam extends BaseParam {
	type: 'int';
	value: number;
	vmin: number;
	vmax: number;
}

export interface BoolParam extends BaseParam {
	type: 'bool';
	value: boolean;
}

export interface StringParam extends BaseParam {
	type: 'string';
	value: string;
	options: string[] | null;
}

/** A request rather than a value: it holds none, and firing it is the whole edit. */
export interface PulseParam extends BaseParam {
	type: 'pulse';
	value: null;
}

export interface UnknownParam extends BaseParam {
	type: 'unknown';
}

export type ParamDescriptor =
	| FloatParam
	| IntParam
	| BoolParam
	| StringParam
	| PulseParam
	| UnknownParam;
