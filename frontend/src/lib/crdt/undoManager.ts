/**
 * The client-scoped `Y.UndoManager` for the control-plane doc.
 *
 * It tracks ONLY this client's own leaf writes (`LOCAL_ORIGIN`, stamped by `SyncClient.commit`) —
 * params, node/instance position, expressions, per-slot viewer state, and globals — so undo reverses
 * MY last leaf edit and ignores the manager's structural writes and other clients' edits (both
 * `REMOTE_ORIGIN`). Structural ops stay manager-validated intents undone via inverse intents (§6
 * Option A); this handles the merge-safe leaves natively. The `UndoCoordinator` sequences the two.
 */
import * as Y from 'yjs';
import { LOCAL_ORIGIN } from './syncClient';
import { nodesMap, linksArray, instancesMap, globalsMap } from './graphDoc';

/** Coalescing window (ms): rapid same-origin leaf edits (a slider drag's settled commits) fold into
 * one undo step — replacing the old `coalesceKey` logic. `stopCapturing()` forces a boundary. */
export const UNDO_CAPTURE_MS = 300;

/** Build the client-scoped UndoManager over the doc's shared roots. All four roots are tracked;
 * clients only ever write leaves into `nodes` / `instances` / `globals` under `LOCAL_ORIGIN`, and
 * `links` is manager-authored — but tracking it is harmless (no local writes land there) and keeps
 * the scope exhaustive should a leaf ever live there. */
export function makeUndoManager(doc: Y.Doc): Y.UndoManager {
	return new Y.UndoManager([nodesMap(doc), linksArray(doc), instancesMap(doc), globalsMap(doc)], {
		trackedOrigins: new Set([LOCAL_ORIGIN]),
		captureTimeout: UNDO_CAPTURE_MS
	});
}
