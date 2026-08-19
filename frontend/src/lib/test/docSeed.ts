/**
 * Fill a store's replica the way the manager fills it — by emitting the document events the wire
 * carries, never by writing into the replica.
 *
 * A helper that reached into the replica would prove the readers and nothing else. Going through
 * `doc_state` and `doc_patch` proves the readers AND that the store follows what the manager says,
 * versions included: a patch stated against the wrong base is refused here exactly as it is in the
 * browser.
 */
import { applyMerge } from '$lib/crdt/mergePatch';
import { emptyDoc, type Doc } from '$lib/crdt/graphDoc';
import type { FakeControl } from './fakeControl';

type Obj = Record<string, unknown>;

export class DocSeed {
	private doc: Doc = emptyDoc();
	private v = 0;

	/** Seeds the empty document at once, as a real connection does: the manager sends `doc_state`
	 * unprompted, and a replica refuses a patch until it has a base to apply it to. */
	constructor(private fc: FakeControl) {
		this.push();
	}

	/** The document as it stands — what the store should be reading. */
	get state(): Doc {
		return this.doc;
	}

	/** The version the last event carried. */
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

	/** One sub-patch scope. `members` is uid → is-itself-a-scope. */
	instance(uid: string, rec: Obj): this {
		return this.patch({ instances: { [uid]: rec } });
	}

	/** The link array, whole — as the projection carries it. */
	links(list: Obj[]): this {
		return this.patch({ links: list });
	}

	/** One global, `{value, type, system}`. */
	global(name: string, rec: Obj): this {
		return this.patch({ globals: { [name]: rec } });
	}

	/** One arrangement entry. */
	arrangement(id: string, rec: Obj): this {
		return this.patch({ arrangement: { [id]: rec } });
	}

	/** Drop a key from a root — the delete half of a merge patch. */
	remove(root: string, key: string): this {
		return this.patch({ [root]: { [key]: null } });
	}
}

/** `new DocSeed(fc)`, spelled the way a test reads. */
export function seed(fc: FakeControl): DocSeed {
	return new DocSeed(fc);
}
