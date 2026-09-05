# VST3 commercial effects render silence

Some hosted VST3 plugins process audio correctly; a large class of commercial ones accept every
call, return `kResultOk`, and emit pure silence. This is an OPEN bug. The record below is what has
been ruled out by measurement, so it is not re-derived.

## The split, measured

One instrumented process loop, one server, one mono `Osc` sine (rms ~0.71) fanned to stereo:

| plugin | vendor / framework | result |
| --- | --- | --- |
| SonoBus | JUCE, open-source | **works** — passes audio, params sweep the output as expected |
| TR5 Classic Clipper | IK Multimedia | silent |
| Ozone / RX / Neutron | iZotope | silent |
| Graindad | Sugar Bytes | dry passes, wet silent |

The silent plugins WORK IN ABLETON on the same machine, so they are licensed and functional. The
bug is goofi's.

## What is proven, not assumed

Instrumentation in `goofi-audio/src/vst3/node.rs::block` (since reverted) established, at steady
state (block 200+), for the clipper vs SonoBus in the same process:

- goofi DELIVERS valid audio to the plugin's input buffer: `staged_before` = 85.29 = 2 × the mono
  port. The input copy and the mono→stereo fan-out are correct for both plugins.
- goofi READS the output correctly: a sentinel (0.123) written into the output before `process`
  is WIPED TO ZERO by the clipper — the plugin actively writes silence, it does not leave the
  buffer untouched. SonoBus, identical buffers, wipes the sentinel and writes real audio.
- The clipper also zeroes its own INPUT buffer (`staged_after` = 0); SonoBus leaves it at 85.29.
- `process()` returns `kResultOk`.
- Bus arrangement negotiates cleanly: `getBusArrangement` → kStereo, `setBusArrangements` accepted,
  `getBusInfo` → 2ch main bus, `activateBus`/`setupProcessing`/`setActive`/`setProcessing` all ok.
- The plugin queries the host for ZERO interfaces (`isPlugInterfaceSupported` never fires).

So goofi's audio path is correct in both directions, proven against a plugin that renders through
the identical code. The silent plugins CHOOSE silence.

## Ruled out, each by its own test

transport (zeroed ProcessContext) · setProcessing treated as mandatory · runtime component↔controller
pairing · IPlugInterfaceSupport unimplemented · no runtime IComponentHandler · host name (posed as
"Ableton Live") · parameter values (sent none — still silent) · the Bypass parameter (exposed and
sent off — still silent) · output-buffer routing (sentinel) · bus-activation order (moved to
canonical, after setupProcessing — still silent; kept as commit 408f72b2 because it is correct
regardless).

## What was fixed along the way (real, kept)

None cured this bug, but each was a genuine host-contract gap and is committed: the IConnectionPoint
introduction + IMessage/IAttributeList allocator (which was ALSO the zero-parameters-on-scan bug),
MAX_PORTS 16→64 with truncation, setProcessing accepted as optional, a running/advancing transport,
the component↔controller pairing at runtime, and the canonical bus-activation order.

## Where to look next

The plugin actively muting itself, while a JUCE plugin renders, with every host call accepted,
points at something in the host CONTEXT surface these plugins read and JUCE does not — not the
process call, which is proven correct. The efficient next step is NOT another hypothesis-per-restart
(that was tried to exhaustion): it is a standalone minimal host built on the same `vst3` crate that
performs the exact Steinberg reference init sequence and feeds one plugin a sine. If that host
renders the clipper, diff it against goofi. If even it renders silence, the cause is deeper than
goofi's hosting and needs the plugin's own diagnostics.

## Meanwhile

The parameter route works end to end and is the portable answer: plugin parameters are goofi params,
drivable by other nodes, headless and remote. Build on plugins proven to render (SonoBus-class).
