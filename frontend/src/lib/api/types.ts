/** Wire shapes for parameter descriptors as the bridge emits them. */

export interface BaseParam {
	value: unknown;
	doc: string | null;
	/** True when the node declared a refresh method for this param. */
	refreshable: boolean;
	/** The source of a bound expression; it survives a toggle-off of `expression_enabled`. */
	expression: string | null;
	expression_enabled: boolean;
	/** When true, a re-eval that changes the value wakes the node's `process()`. */
	expression_triggers_process: boolean;
	/** The last evaluation/compile error for this expression, or null. */
	expression_error: string | null;
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
	trigger: boolean;
}

export interface StringParam extends BaseParam {
	type: 'string';
	value: string;
	options: string[] | null;
}

export interface UnknownParam extends BaseParam {
	type: 'unknown';
}

export type ParamDescriptor = FloatParam | IntParam | BoolParam | StringParam | UnknownParam;
