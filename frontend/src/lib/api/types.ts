/** Wire shapes for parameter descriptors as the bridge emits them. */

export interface BaseParam {
	value: unknown;
	doc: string | null;
	save_param: boolean;
	/** When non-null, the param's value is derived from a Python snippet
	 * evaluated in the owning node's process. UI shows an `fx` mode in
	 * place of the normal slider / number / text widget. */
	expression: string | null;
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
