/** Wire shapes for parameter descriptors as the bridge emits them. */

export interface BaseParam {
	value: unknown;
	doc: string | null;
	save_param: boolean;
	/** True when the node declared a refresh method for this param (device /
	 * stream pickers) — the UI shows a ⟳ button that re-evaluates the options. */
	refreshable: boolean;
	/** The source code of a bound expression, or null if none has been
	 * authored. Surviving a toggle-off of `expression_enabled` means a
	 * user can flip fx mode back on without losing what they typed. */
	expression: string | null;
	/** Whether the expression is currently being evaluated. UI's fx
	 * button is bound to this; toggling fx off preserves the source. */
	expression_enabled: boolean;
	/** Per-expression flag: when true, a re-eval that changes the param's
	 * value wakes the node's `process()`. */
	expression_triggers_process: boolean;
	/** Per-expression flag: when true, the expression is re-evaluated
	 * before every process() tick — useful for expressions with no slot
	 * reference (e.g. `time.time()`). */
	expression_autoeval: boolean;
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
