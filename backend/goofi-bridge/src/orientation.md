goofi-pipe is a live signal-processing patch: a graph of nodes running right now, in a window a
human has open beside you. Your edits reach their screen at once and theirs reach your next read,
so work in small steps and check each one. Call `nodes inspect` first and again between steps: it
draws the graph and lists every standing error with how long it has stood. Every write answers
with what it did, so read the reply instead of following it with another call. Never guess a name
— node types, panel types and viewer kinds are enumerated by the tool that takes them.

You drive goofi through ONE tool, `goofi_exec`: each command is a line, `<op> [--arg value …]`,
and `op list` answers every op with its arguments and result. `library list` is the palette;
`node add`, `link add` and `node edit` build; `library refresh` loads a Python node you wrote
into `nodes/` beside you. `undo`/`redo` are yours alone and never reach the human's edits. Below is
detail: read what a step needs.

## Seeing

`nodes inspect` draws one scope. No argument is the root; `scope` takes a sub-patch uid.

    scope: root

    ```mermaid
    flowchart LR
      n000000000001["oscillator0: Oscillator<br/>000000000001"]
      n000000000002["buffer0: Buffer<br/>000000000002"]
      n000000000001 -- out→data --> n000000000002
    ```

A node's uid is its mermaid id without the leading `n`; a boundary port's id is its mermaid id
verbatim. `session status` lists every standing error with how long it has stood — whole patch,
whichever scope you drew — and the age tells a pipeline still settling from a broken one: a 0.2s
error may clear itself, one standing 30s will not.

`node state <node>` is the cheap peek — params, output health, frame meta and error, all on by
default. Pass `--no-params` or `--no-error` to drop a section, `--slot` to narrow to one output.

    buffer0: Buffer (uid 000000000002, in-process, stage ready)

    params:
      buffer.size = 1000 (int 1..10000000)

    outputs:
      out: f32[1000] finite=1000/1000 range=[-1,1]
        meta: sfreq=250, ufreq=29.891284, index=358

    error: none

The `out:` line is the value-health line — shape, how many elements are real numbers, and the scale
of the ones that are. `finite=511/512` is a NaN leaking in, `range=[0,0]` is silence; reading it
never dumps data. A param bound to an expression prints as `expr: <source> → <value> (on)`, which
is what `node edit` takes back.

`layout inspect` names the pages and panel ids the layout ops address; `session status` says
where the patch is saved and whether it differs from disk; `global list` says what an expression
can read.

## Building

    node add Oscillator --pos 0,0
    → {"uid": "000000000001", "name": "oscillator0", "input_slots": {},
       "output_slots": {"out": "ARRAY"},
       "params": {"oscillator": {"frequency": 1.0, "waveform": "sine", …}, "common": {…}}}

    link add --node_out 000000000001 --slot_out out \
             --node_in 000000000002 --slot_in data   → {…, "dtype": "ARRAY"}

    node edit 000000000001 --params '{"oscillator": {"frequency": 7.5}}'
    → {"params": {"oscillator": {"frequency": {"value": 7.5, "error": null}}}}

`name` is what `nd()` addresses a node by; `uid` is what every tool takes. `node edit` answers each
param **as stored** — coerced to the param's declared type, so a fraction into an int comes back
rounded, and a declared min/max is the editor's range, not a clamp. It is also the rename, the move
and the viewer write, and any mix of them is one call and one undo. A param entry may be
`{"expression": "nd('other_node').sfreq"}` — or `globals.x`, or `t` — instead of a literal.

`link add` refuses a dtype mismatch and names both ends, and a wrong slot name is refused by naming
the slots that exist — but a uid naming nothing is *not* refused: it answers as though it wired and
no wire appears. Take uids from a read, never from memory.

Panel types and viewer kinds are **closed sets**, not free strings — guessing `params` for
`parameters` is a mistake a real agent made. Each tool's description lists its choices, and a guess
is refused with the whole set: `no panel type "params" — this app has: empty, node-editor,
parameters, viewer, metadata, console, globals, agent`.

## Custom Python nodes

The most powerful thing you can do here. The patch's own node library is `nodes/` under the
workspace: one file is one type, named by CamelCasing its stem (`gain.py` → `Gain`), overriding a
shipped type of the same name.

    # nodes/gain.py
    import goofi

    class Gain(goofi.Node):
        INPUTS = {"data": goofi.DataType.ARRAY}
        OUTPUTS = {"out": goofi.DataType.ARRAY}
        PARAMS = {"gain": {"factor": goofi.FloatParam(2.0, 0.0, 10.0)}}

        def process(self, data):
            if data is None:
                return None
            return {"out": (data.data * self.params.gain.factor.value, data.meta)}

    library refresh → {"added": ["Gain"], "changed": [], "removed": []}

Four constants declare the node, and each may be omitted: `INPUTS`, `OUTPUTS`, `PARAMS`, and
`PRODUCER = True` for a source that paces itself rather than waiting for a frame. An input slot that
`process()` reads unconditionally should say so — `goofi.InputSlot(goofi.DataType.ARRAY,
required=True)` — and the engine then refuses the tick rather than calling you with `None`.

Edit the file and rescan again: it returns under `changed`, and every live instance of that type
**restarts onto the new code** — `setup()` runs again, so a buffer empties and a device reopens. A
node whose imports are missing registers as unavailable and names the module; a node that raises
inside `process()` becomes that node's error, not a crash. `library get <type>` gives you a
shipped Python node to copy from (a native Rust type has no source text).

## The workspace

Your working directory **is** the patch's workspace, and it rides inside the `.gfi` when the human
saves — so anything you leave there returns with the patch, including this file, which is yours to
edit as you learn what this patch is for. `session status` says where it is (a per-run temp
directory, so ask rather than assume). `.goofiignore` says what is *not* packaged (`__pycache__/`,
`*.pyc`, …); its header documents its own syntax, and the same list decides whether the workspace
counts as changed, so a scratch file that should not travel belongs in it.

## Handle with care

`session load` and `session new` replace the patch you work inside and clear the undo history,
so they cannot be taken back. `session save` and `dir list` are the human's file browser,
`layout viewpoint edit` is the camera of a client with a screen, and the agent ops can stop the
process you speak through. All of these answer to you, but they act on what the human is looking
at — ask the human before you use one.
