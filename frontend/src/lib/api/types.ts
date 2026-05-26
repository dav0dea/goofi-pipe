/** Wire shapes for parameter descriptors as the bridge emits them. */

export interface BaseParam {
	value: unknown;
	doc: string | null;
	save_param: boolean;
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
