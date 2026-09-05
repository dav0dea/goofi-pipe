# VST3: the IK T-RackS suite renders silence

Most hosted VST3 effects render audio, and the parameter and runtime fixes are proven against a
fixture that reproduces the failure they repair. One family does not: the 39 IK Multimedia T-RackS
plugins accept every host call, return `kResultOk`, and emit silence. This is an OPEN bug.

## What is settled

goofi's audio path is correct in both directions. On a sine into the plugin, the input buffer it is
handed carries the signal (a mono source fanned to the plugin's stereo bus); a sentinel written into
the output before `process` is wiped to zero by the plugin, so the plugin actively writes silence
rather than leaving the buffer untouched. A free JUCE plugin (SonoBus), given the identical buffers
through the identical code, writes real audio and its `dryLevel` param tracks a sweep. So the plugin
chooses silence.

Ruled out by isolated tests: parameter values (sending none is still silent), the bypass parameter,
bus arrangement (negotiates stereo, accepted), the activation lifecycle (every call returns ok), the
host name, and output-buffer routing (the sentinel).

## What is open

Why an IK plugin mutes itself where a JUCE plugin, handed the identical buffers through the
identical code, renders. Every host call is accepted, so the difference is in what the plugin reads
back from the host rather than in what it is told. It is IK-specific: no other vendor here is
affected.
