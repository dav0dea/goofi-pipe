goofi-pipe is a live signal-processing patch: a graph of nodes running right now, in a window a
human has open beside you. Your edits reach their screen at once and theirs reach your next read,
so work in small steps and check each one. Call `goofi nodes inspect` first and again between
steps: it draws the graph, and `goofi session status` lists every standing error with how long it
has stood. Every write answers with what it did, so read the reply instead of following it with
another call. Never guess a name — node types, panel types and viewer kinds are enumerated by the
op that takes them.

You drive goofi with the `goofi` command in this shell — it is already on your PATH, pointed at
THIS server, and your ops land in your own undo stack: `undo`/`redo` are yours alone and never
touch the human's. One op is one line, `goofi <op> [--arg value …]`; `goofi op list` answers
every op with its arguments and result, `goofi help <group>` lists a group, `--help` on any op
explains it, and `--json` answers the raw JSON for `jq`. Several ops become ONE undo step through
stdin:

    goofi - <<'EOF'
    node add --type Oscillator --member_uid aaaaaaaaaaa1
    node add --type Buffer --member_uid aaaaaaaaaaa2
    link add aaaaaaaaaaa1/out aaaaaaaaaaa2/data
    EOF

`--member_uid` lets you CHOOSE a uid, so a later line can wire what an earlier one built. A batch
lands whole or not at all. (An MCP-connected agent has the same lines through the one tool
`goofi_exec`; everything below holds there too.)

## Seeing

`goofi nodes inspect` draws one scope. No argument is the root; `--scope` takes a sub-patch uid.

    scope: root

    ```mermaid
    flowchart LR
      n000000000001["oscillator0: Oscillator<br/>000000000001"]
      n000000000002["buffer0: Buffer<br/>000000000002"]
      n000000000001 -- out→data --> n000000000002
    ```

A node's uid is its mermaid id without the leading `n`. `goofi session status` lists every
standing error with its age — a 0.2s error may clear itself, one standing 30s will not — and says
where the patch is saved. `goofi node state <node>` is the cheap peek — params, output health,
frame meta and error; `--no-params`/`--no-error` drop a section, `--slot` narrows to one output.

    buffer0: Buffer (uid 000000000002, in-process, stage ready)

    params:
      buffer.size = 1000 (int 1..10000000)

    outputs:
      out: f32[1000] finite=1000/1000 range=[-1,1]
        meta: sfreq=250, ufreq=29.891284, index=358

    error: none

The `out:` line is value health: `finite=511/512` is a NaN leaking in, `range=[0,0]` is silence.
The DATA itself is one op away, raw:

    goofi node snapshot 000000000002/out \
      | python3 -c "import numpy,sys; print(numpy.load(sys.stdin.buffer).mean())"

It answers the slot's latest frame as NPY on stdout — a facade or boundary port resolves to the
stream behind it. The first ask on a never-watched slot opens its feed and answers null; ask
again after the node's next emit. A monitor is a loop over it. `goofi layout inspect` names the
pages and panel ids the layout ops address; `goofi global list` says what an expression can read.

## Building

    goofi node add Oscillator --pos 0,0
    → {"uid": "000000000001", "name": "oscillator0", "input_slots": {},
       "output_slots": {"out": "ARRAY"}, "params": {…}}

    goofi link add 000000000001/out 000000000002/data   → {"from": …, "to": …, "dtype": "ARRAY"}

    goofi node param edit 000000000001 oscillator/frequency --value 7.5
    → {"value": 7.5, "error": null}

`name` is what `nd()` addresses a node by; `uid` is what every op takes. `node param edit`
answers the param **as stored** — coerced to its declared type, so a fraction into an int comes
back rounded; the declared min/max are the editor's range, not a clamp. `--expression
"nd('other_node').sfreq"` — or `globals.x`, or `t` — binds instead of a literal. `node edit` is
the node's own record — the rename, the move and the viewer write, any mix in one call and one
undo.

`link add` refuses a dtype mismatch and names both ends, and a wrong slot name is refused by
naming the slots that exist — but a uid naming nothing is *not* refused: it answers as though it
wired and no wire appears. Take uids from a read, never from memory.

Panel types and viewer kinds are **closed sets**, not free strings — guessing `params` for
`parameters` is a mistake a real agent made. A guess is refused with the whole set: `no panel
type "params" — this app has: empty, node-editor, parameters, viewer, metadata, console,
globals, agent`.

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

    goofi library refresh → {"added": ["Gain"], "changed": [], "removed": []}

Four constants declare the node, and each may be omitted: `INPUTS`, `OUTPUTS`, `PARAMS`, and
`PRODUCER = True` for a source that paces itself rather than waiting for a frame. An input slot
that `process()` reads unconditionally should say so — `goofi.InputSlot(goofi.DataType.ARRAY,
required=True)` — and the engine then refuses the tick rather than calling you with `None`.

Edit the file and refresh again: it returns under `changed`, and every live instance of that type
**restarts onto the new code** — `setup()` runs again, so a buffer empties and a device reopens.
A node whose imports are missing registers as unavailable and names the module; a node that
raises inside `process()` becomes that node's error, not a crash. `goofi library get <type>`
gives you a shipped Python node to copy from (a native Rust type has no source text).

## The workspace

Your working directory **is** the patch's workspace, and it rides inside the `.gfi` when the
human saves — so anything you leave there returns with the patch, including this file, which is
yours to edit as you learn what this patch is for. `goofi session status` says where it is (a
per-run temp directory, so ask rather than assume). `.goofiignore` says what is *not* packaged
(`__pycache__/`, `*.pyc`, …); its header documents its own syntax, and the same list decides
whether the workspace counts as changed, so a scratch file that should not travel belongs in it.

## Handle with care

`session load` and `session new` replace the patch you work inside and clear the undo history,
so they cannot be taken back. `session save` and `dir list` are the human's file browser,
`layout viewpoint edit` is the camera of a client with a screen, and the agent ops can stop the
shell you run in. All of these answer to you, but they act on what the human is looking at — ask
the human before you use one.
