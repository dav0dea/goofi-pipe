# VST3 plugin editors: main-thread hosting

`node editor <uid>` opens a plugin's own window. It works for some plugins and not others, and
making it work everywhere needs an architectural change, not a patch. This file records what is
known so the next attempt does not re-derive it.

## What happens per plugin, measured

| plugin | framework | editor |
| --- | --- | --- |
| Graindad | Sugar Bytes' own | opens and renders |
| SonoBus | JUCE | aborts the process (0xcfffffff, a customer RaiseException) |
| Transit 2 | BABY Audio | window opens BLANK; its `attached()` blocks |

Three frameworks, three outcomes. The common thread: JUCE and BABY Audio editors expect to run on
the process's MAIN thread with a UI event loop already spinning. goofi hosts each editor on a
SPAWNED thread. Graindad tolerates that; the others do not.

## The two failures

- **JUCE aborts.** JUCE asserts it is on the message thread and raises a hard exception when it is
  not. Nothing short of running its editor on the real main thread satisfies it.
- **BABY Audio blanks and blocks.** Its `attached()` does not return — it waits for a running event
  loop that the spawned thread only starts pumping AFTER attach returns. Classic deadlock: attach
  waits for the loop, the loop waits for attach.

## What is fixed (commit a8348f78)

The blocked `attached()` used to wedge the WHOLE server: `node editor` held the graph lock while it
waited for the window, so a plugin that never finished attaching froze every other op. The open now
bounds that wait and DETACHES a thread that never answers. A blank, non-responding window can still
appear for such a plugin, but goofi and the rest of the patch stay alive. This is a containment
fix, not a render fix.

## The real fix

Host editors on the process main thread. Today `main()` runs the tokio runtime on the main thread;
this needs inverting — the main thread becomes a native message loop (a Win32 `GetMessage` pump on
Windows, the platform equivalent elsewhere), tokio moves to a worker, and editor create/attach/pump
all happen on that main thread. It touches `goofi-cli/src/main.rs` and is per-platform (Win32,
X11/Wayland, NSView). It is a deliberate piece of work, gated on whether plugin GUIs are worth it
given that:

- goofi's own param panel already reaches every plugin's parameters, works headless and remotely,
  and lets other nodes modulate them — which a plugin's own window cannot do.
- Even with the window rendering, a knob turned in it would not affect the audio until the
  editor's `performEdit` is bridged to the audio instance (a second, smaller piece of work — the
  editor holds its own plugin instance today).

So the editor is a local convenience with real cost, and the parameter route is the portable answer.
