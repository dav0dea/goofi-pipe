import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import { LOCAL_ORIGIN, REMOTE_ORIGIN } from './syncClient';
import { makeUndoManager } from './undoManager';
import { nodesMap, setParamValue, paramValue } from './graphDoc';

/** Seed a node BEFORE the UndoManager is created, so it isn't part of any tracked step. */
function seedNode(doc: Y.Doc, uid: string): void {
	const n = new Y.Map<unknown>();
	n.set('type', 'Oscillator');
	n.set('name', uid);
	nodesMap(doc).set(uid, n);
}

describe('makeUndoManager', () => {
	it('captures a LOCAL_ORIGIN leaf write and undoes it', () => {
		const doc = new Y.Doc();
		seedNode(doc, 'n1');
		const um = makeUndoManager(doc);

		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 5), LOCAL_ORIGIN);
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBe(5);
		expect(um.canUndo()).toBe(true);

		um.undo();
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBeUndefined();
		expect(um.canUndo()).toBe(false);
		um.redo();
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBe(5);
	});

	it('ignores a REMOTE_ORIGIN write (the manager / other clients)', () => {
		const doc = new Y.Doc();
		seedNode(doc, 'n1');
		const um = makeUndoManager(doc);

		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 5), REMOTE_ORIGIN);
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBe(5);
		expect(um.canUndo()).toBe(false); // not my edit → not on my undo stack
	});

	it('coalesces rapid same-origin writes into one undo step', () => {
		const doc = new Y.Doc();
		seedNode(doc, 'n1');
		const um = makeUndoManager(doc);

		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 5), LOCAL_ORIGIN);
		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 9), LOCAL_ORIGIN);
		// Within captureTimeout (same tick) → one merged step reverting BOTH.
		um.undo();
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBeUndefined();
		expect(um.canUndo()).toBe(false);
	});

	it('stopCapturing() splits writes into separate undo steps', () => {
		const doc = new Y.Doc();
		seedNode(doc, 'n1');
		const um = makeUndoManager(doc);

		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 5), LOCAL_ORIGIN);
		um.stopCapturing();
		doc.transact(() => setParamValue(doc, 'n1', 'common', 'frequency', 9), LOCAL_ORIGIN);

		um.undo(); // reverts only the second write
		expect(paramValue(doc, 'n1', 'common', 'frequency')).toBe(5);
		expect(um.canUndo()).toBe(true);
	});
});
