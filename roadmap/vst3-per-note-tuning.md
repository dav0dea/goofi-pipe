# VST3: a note-on rounds the pitch to the nearest semitone

goofi's pitch is continuous — volts per octave, zero at C4 — and goofi's own audio nodes read it
that way: `Osc` and `Filter` call `hz_of(pitch)` and sound whatever number they are given. A VST3
plugin does not. `note_on` in `backend/audio/goofi-audio/src/vst3/node.rs` rounds the pitch to a
whole MIDI note and sends `tuning: 0.0`, so everything between two semitones is discarded at the
plugin boundary — on the `voice` cable and on the three `voice` params alike.

That makes a microtonal scale unplayable through a plugin while it plays correctly through `Osc`,
which is a split no user can see and no doc explains.

## What is settled

The fix is the field already in the event. VST3's note-on carries a per-note `tuning` offset beside
the integer pitch for exactly this, and goofi sends zero into it. The residual — the part the round
threw away — is what belongs there.

The rounding itself stays: a plugin's note IS an integer, and the tuning offset is how VST3 says the
rest. So this is one term added, not a redesign of the voice path.

## What is open

Whether the offset is honoured widely enough to rely on. A plugin may ignore `tuning` and sound the
rounded note, and there is no way to ask it in advance. So the situation that pins this needs a
plugin known to honour it, and a plugin known not to is not a failure of goofi's.

## Why it is worth doing

A tuning is the point of a biotuner chain, and `dav0dea/goofi-pipe#21` builds one. A keyboard that
plays a seven-note scale is the demo, and today it plays that scale into `Osc` and a rounded
12-TET approximation of it into every synth a user already owns.
