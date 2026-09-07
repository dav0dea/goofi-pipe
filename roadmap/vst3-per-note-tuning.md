# VST3: a plugin plays a microtonal scale only if the host knows its bend range

goofi's pitch is continuous — volts per octave, zero at C4 — and goofi's own audio nodes read it
that way: `Osc` and `Filter` call `hz_of(pitch)` and sound whatever number they are given. A VST3
plugin does not: its note is an integer, and the fraction has to travel beside it.

`note_on` in `backend/audio/goofi-audio/src/vst3/node.rs` now sends that fraction two ways, and
which one it uses is asked of the plugin rather than assumed. What follows is what the asking
answered, measured against the 16 instruments installed on the development machine.

## What is settled

**The note's own `tuning` field is not the mechanism.** It is exact, it is optional, and it is
transmitted correctly — a probe reads back `tuning=40.000153` for a 40-cent request — and every one
of the 16 ignores it. It stays as the fallback because it costs nothing and it is right where it is
honoured, but nothing may depend on it.

**Note expression is not the mechanism either.** `kTuningTypeID` is the one per-note offset whose
support can be ASKED rather than guessed, and whose scale the SDK fixes at `240 x (norm - 0.5)`
cents. All 16 answer no to `INoteExpressionController`. The probe was written, run, and removed;
re-add it when a plugin that answers yes is in the room.

**MPE is not the mechanism.** An MPE Configuration Message — RPN 6, sent as CC 101/100/6 through the
mapped parameters — was ignored by both plugins it was sent to.

**Pitch bend is the mechanism, one per voice channel.** `wheels()` asks
`getMidiControllerAssignment(0, ch, kPitchBend, &id)` per channel, so a voice bends on its own
channel's wheel. Measured on Vital: a +40 cent request lands at +40.8 cents when the range matches.

**The range is the whole problem, and it is per plugin.** A wheel is a fraction of a range the
plugin chooses and mostly does not publish. Vital publishes `Pitch Bend Range` and Diva publishes
`PitchBend Up`/`Down`, both defaulting to 2, and setting Vital's to 12 turned the same request into
+240.3 cents — exactly 6x, so the arithmetic is understood. Synplant publishes 16 per-channel bend
VALUES and no range, and behaves as though its range were 12. Ten publish nothing bend-named,
which is a statement about their parameter list and not about whether they accept bend.

**A survey by parameter name is not a survey of what a plugin accepts.** A JUCE plugin exposes its
MIDI mappings as parameters that never appear in the visible list, so "no bend parameter" and "no
bend" are different findings and only the first was measured.

## What is open

**Whether a chord holds its voices' bends apart.** One note is proved. A test that claimed the bend
was global put both voices on one note number, which a synth may merge into one voice, so it proved
nothing and was withdrawn. Settling it needs two distinct pitches, both confirmed sounding, in the
one configuration under measurement.

**How to learn the range where the plugin does not publish one.** Setting it is exact and covers
two of 16. Measuring it — sound a note, read the pitch, apply full bend, read it again, divide —
needs no cooperation at all and would cover the rest, once per plugin, cached beside everything
else the scan already remembers.

## Why the reference implementation is not a model

The Max for Live device this feature is measured against (`MEME/m4l_bundle/Biotuner`) is worth
reading and is not worth copying. It is `is_mpe: 0` and echoes the incoming channel, so every note
shares one bend and a chord wears the last note's. Its `scale 0 2 -8191 8192` maps a frequency
RATIO onto full bend, which pins full travel to the octave and is exact only at a receiver range of
17.31 semitones; against the usual 2 it renders its own stored -191 — meaning -40 cents — as -4.7.
It reads as working because it fails quietly, in the direction of slightly flat.
