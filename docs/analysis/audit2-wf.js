export const meta = {
  name: 'deep-audit-2',
  description: 'Adversarial bug audit of the perf+critical-fix branch (18 commits, Phases 1-7)',
  phases: [{ title: 'Review' }, { title: 'Verify' }],
}

const BASE = '/home/philipp/projects/goofi-next/goofi-pipe'

// Each dimension targets the NEW/changed code from this branch (the highest-risk,
// least-reviewed surface) plus its integration points.
const DIMENSIONS = [
  {
    key: 'concurrency',
    prompt: `Audit thread-safety in the goofi-pipe backend changes. Read ${BASE}/src/goofi/transport.py (the WaitSet class + its lock), ${BASE}/src/goofi/node.py (the _expression_lock + _set_expression/_apply_expression/_match_fired_expression, the I2 input draining in _processing_loop + _subscribe_input/_unsubscribe_input + _drain_pending_inputs), and ${BASE}/src/goofi/bridge/data.py (the run_in_executor set_data_handler under _lock). Look for: deadlocks (lock-ordering between _expression_lock and the WaitSet lock, or the asyncio _lock held across run_in_executor while a re-subscribe races), lost wakeups, the WaitSet snapshot releasing the lock before the blocking wait, listeners attached but never detached, the I2 change breaking existing trigger semantics, re-subscribe races in the bridge teardown. Report only REAL, triggerable bugs.`,
  },
  {
    key: 'webgl',
    prompt: `Audit the new WebGL ImageViewer. Read ${BASE}/frontend/src/lib/viewers/imageGL.ts and ${BASE}/frontend/src/lib/viewers/ImageViewer.svelte. Look for: wrong texture internal-format/format/type combos per dtype+channels, UNPACK_ALIGNMENT issues for odd row widths, the grayscale value-range/sampleScale math vs the old 2D path, colormap LUT only re-uploaded on name change (stale if makeLUT changes), texture/program resource leaks (dispose, re-create on remount), the aspect-fit downsample math, float-linear-filter fallback, the two-canvas hidden toggling, ResizeObserver lifecycle, behavior when cssW/cssH are 0, RGBA32F/R32F availability, and any case where a BigInt-backed values array reaches texImage2D. Report only REAL bugs (wrong pixels, crashes, leaks, blank render).`,
  },
  {
    key: 'worker',
    prompt: `Audit the new Web Worker data plane. Read ${BASE}/frontend/src/lib/api/dataWorker.ts, ${BASE}/frontend/src/lib/api/data.ts, and ${BASE}/frontend/src/lib/api/frames.ts. Look for: ArrayBuffer transfer aliasing (the same buffer transferred twice → throw) or transferring a buffer still referenced, detached-buffer reuse, TABLE/nested-frame buffer collection correctness, WS reconnect/terminal-close (4000+) logic in the worker, refcount correctness on sub/unsub, the setInterval tick never cleared / leaking, worker never terminated, self.location host correctness, the rAF paint scheduler budget (starvation, the most-starved-first fairness, a deferred slot whose pending is overwritten, requestFlush re-arm), latestFrame correctness after the rewrite, SSR/non-browser safety (new Worker at import time). Report only REAL bugs.`,
  },
  {
    key: 'backend-raw',
    prompt: `Audit the backend raw-byte forwarding. Read ${BASE}/src/goofi/node_helpers.py (set_data_handler raw flag, the 4-tuple _data_handlers, _data_pump raw branch, the B4 listener close) and ${BASE}/src/goofi/bridge/data.py (on_frame forwarding the raw bytes, _SlotMux). Look for: the raw bytes being mutated/reused unsafely (is take_latest()'s bytes truly a private heap copy?), the 4-tuple unpacked wrong anywhere, a consumer still expecting the 3-tuple, the decoded path regressed for raw=False, the listener-close breaking the subscriber, fan-out of the same bytes object to multiple forwarders being unsafe. Report only REAL bugs.`,
  },
  {
    key: 'uint8-nodes',
    prompt: `Audit the uint8 image convention changes. Read ${BASE}/src/goofi/image_utils.py and the changed nodes: ${BASE}/src/goofi/nodes/inputs/videostream.py, loadfile.py, imagegeneration.py, ${BASE}/src/goofi/nodes/misc/edgedetector.py, colorenhancer.py, hsvtorgb.py, rgbtohsv.py, ${BASE}/src/goofi/nodes/analysis/poseestimation.py. Then grep ${BASE}/src/goofi/nodes for OTHER nodes that consume an image input (shape[-1]==3, cv2, PIL, colorsys, mediapipe, *255, /255) that were NOT updated and would now break on uint8 input. Look for: as_uint8/as_float01 edge cases (non-uint8 integer dtypes, NaN/inf floats, empty arrays), a producer whose downstream still assumes float [0,1], edgedetector's Sobel float {0,255} output path, videostream emitting a non-contiguous BGRA frame. Report only REAL bugs + any MISSED consumer.`,
  },
  {
    key: 'editor-decimate',
    prompt: `Audit the editor + viewer frontend changes. Read ${BASE}/frontend/src/lib/panels/NodeEditorPanel.svelte (the B8 carry/untrack live-selection preservation + the B9 paste-in-flow-coords + FlowApi wiring), ${BASE}/frontend/src/lib/editor/FlowApi.svelte, ${BASE}/frontend/src/lib/viewers/decimate.ts and its wiring in ${BASE}/frontend/src/lib/viewers/ArrayViewer.svelte. Look for: the carry() reading flowNodes untracked but a stale selection persisting after deselect, the FlowApi $bindable assigned once not updating, paste falling back to screen coords when screenToFlow is undefined, decimateMinMax x non-monotonic at bucket boundaries or when m < targetCols, the decimation threshold (m > cols*2) interacting with scalar/1D/2D shape branches, a channel-count mismatch after decimation. Report only REAL bugs.`,
  },
]

const FINDING = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'file', 'line', 'severity', 'detail', 'repro'],
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'number' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          detail: { type: 'string', description: 'What is wrong and why, quoting the code.' },
          repro: { type: 'string', description: 'Concrete trigger / scenario.' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'file', 'line', 'verdict', 'severity', 'reasoning', 'fix'],
  properties: {
    title: { type: 'string' },
    file: { type: 'string' },
    line: { type: 'number' },
    verdict: { type: 'string', enum: ['confirmed', 'false_positive', 'uncertain'] },
    severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'none'] },
    reasoning: { type: 'string', description: 'Quote the real code; explain why it holds or fails.' },
    fix: { type: 'string', description: 'Concrete fix if confirmed.' },
  },
}

phase('Review')
const reviewed = await pipeline(
  DIMENSIONS,
  (d) =>
    agent(
      `You are an adversarial code auditor for goofi-pipe (repo root ${BASE}). Find REAL bugs only — not style, not hypotheticals. Read the actual code before claiming anything.\n\n${d.prompt}`,
      { label: `review:${d.key}`, phase: 'Review', schema: FINDING, agentType: 'Explore' }
    ),
  (res, d) => {
    const findings = (res && res.findings) || []
    if (!findings.length) return []
    return parallel(
      findings.map((f) => () =>
        agent(
          `Adversarially VERIFY this claimed bug in goofi-pipe (repo root ${BASE}). Read the cited file:line and surrounding code yourself. Default to skepticism — many reports miss a guard, a caller invariant, or a runtime semantic. Decide 'confirmed' (real + triggerable), 'false_positive' (does not occur — say what was missed), or 'uncertain'. Provide a concrete fix only if confirmed.\n\nDIMENSION: ${d.key}\nFINDING: ${f.title}\nfile: ${f.file}:${f.line}\nseverity(claimed): ${f.severity}\ndetail: ${f.detail}\nrepro: ${f.repro}`,
          { label: `verify:${d.key}:${(f.title || '').slice(0, 32)}`, phase: 'Verify', schema: VERDICT, agentType: 'Explore' }
        )
      )
    )
  }
)

const verdicts = reviewed.flat().filter(Boolean)
const confirmed = verdicts.filter((v) => v.verdict === 'confirmed')
const uncertain = verdicts.filter((v) => v.verdict === 'uncertain')
return { confirmed, uncertain, allCount: verdicts.length }
