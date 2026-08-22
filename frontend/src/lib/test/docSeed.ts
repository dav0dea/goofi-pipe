/** Fill a store replica the way the manager does — by emitting `doc_state` / `doc_patch`, never by writing into it. */
import { applyMerge } from '$lib/crdt/mergePatch';
import { emptyDoc, type Doc } from '$lib/crdt/graphDoc';
import { SCOPE_TYPE } from '$lib/api/vocab';
import type { FakeControl } from './fakeControl';

type Obj = Record<string, unknown>;

export class DocSeed {
	private doc: Doc = emptyDoc();
	private v = 0;

	/** Seeds the empty document at once: a replica refuses a patch until it has a base. */
	constructor(private fc: FakeControl) {
		this.push();
	}

	get state(): Doc {
		return this.doc;
	}

	get version(): number {
		return this.v;
	}

	/** Send the whole document, as a connection or a lag recovery does. */
	push(doc: Doc = this.doc): this {
		this.doc = structuredClone(doc);
		this.v += 1;
		this.fc.emit({ event: 'doc_state', payload: { v: this.v, doc: structuredClone(this.doc) } });
		return this;
	}

	/** Send one merge patch — `null` at a key deletes it, as RFC 7386 says. */
	patch(patch: Obj): this {
		applyMerge(this.doc, patch);
		const from = this.v;
		this.v += 1;
		this.fc.emit({ event: 'doc_patch', payload: { from, v: this.v, patch: structuredClone(patch) } });
		return this;
	}

	/** One node, with whatever leaves the test cares about beyond identity. */
	node(uid: string, type: string, name: string, pos: [number, number] = [0, 0], extra: Obj = {}): this {
		return this.patch({ nodes: { [uid]: { type, name, pos: { x: pos[0], y: pos[1] }, ...extra } } });
	}

	/** One sub-patch facade — a node record wearing the scope type. Membership is each MEMBER's own
	 * `scope`, so seed the members with it rather than listing them here. */
	instance(uid: string, name: string, pos: [number, number] = [0, 0], extra: Obj = {}): this {
		return this.node(uid, SCOPE_TYPE, name, pos, extra);
	}

	/** One boundary port of `scope`. Its direction and dtype ARE its type; its inner wire is a link. */
	port(uid: string, type: string, name: string, scope: string, pos: [number, number] = [0, 0]): this {
		return this.node(uid, type, name, pos, { scope });
	}

	links(list: Obj[]): this {
		return this.patch({ links: list });
	}

	/** One global, `{value, type, system}`. */
	global(name: string, rec: Obj): this {
		return this.patch({ globals: { [name]: rec } });
	}

	arrangement(id: string, rec: Obj): this {
		return this.patch({ arrangement: { [id]: rec } });
	}

	remove(root: string, key: string): this {
		return this.patch({ [root]: { [key]: null } });
	}
}

export function seed(fc: FakeControl): DocSeed {
	return new DocSeed(fc);
}
