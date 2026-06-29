"""OSCIn runs the OSC server on its OWN thread: the server callback writes new
entries via `_handle_message` -> `self.messages[address] = val` while the
processing thread is inside `process()` handing `self.messages` downstream. With
the default config (keep_messages=True) `process()` did NOT swap the dict, so the
returned payload ALIASED the live `self.messages`. Downstream that dict becomes a
Data(TABLE, ...) whose encode iterates it on the processing thread — if the OSC
thread inserts a new address key meanwhile, the iteration raises
``RuntimeError: dictionary changed size during iteration``.

The fix: `process()` must hand out an INDEPENDENT snapshot copy of the messages,
so the consumer never iterates an object the OSC thread can mutate. These tests
pin that contract deterministically (snapshot identity) instead of racing real
threads, which would be flaky.
"""
import threading

from goofi.node import NodeEnv
from goofi.nodes.inputs.oscin import OSCIn


def _standalone(cls):
    in_slots, out_slots, params = cls._configure()
    return cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def _post_setup_state(node):
    """Mimic the post-`setup()` state WITHOUT starting the OSC backend.

    STANDALONE construction skips auto-setup, and we deliberately do NOT call
    `setup()` / `_start_backend()` here — that would bind a real UDP socket and
    spawn the server thread. We only replicate the attributes `process()` reads.
    """
    node.messages = {}
    node._srv = None
    node._srv_thread = None
    node._lock = threading.RLock()
    node._backend_running = True  # pretend the backend is up so process() won't (re)start it


def test_oscin_process_returns_independent_snapshot():
    """Default config (keep_messages=True): the payload must be a COPY, never the
    live `self.messages`, so a concurrent OSC-thread insert can't change size of
    the dict the consumer is iterating."""
    node = _standalone(OSCIn)
    _post_setup_state(node)

    # Populate via the real producer path (what the OSC server thread calls).
    node._handle_message("/freq", 10.0)
    node._handle_message("/amp", 0.5)
    assert node.messages, "precondition: messages should be populated"

    result = node.process(None, None)
    assert result is not None
    out_obj = result["message"][0]

    # The returned dict must be a distinct object from the live one...
    assert out_obj is not node.messages, (
        "process() handed out the LIVE self.messages dict; a concurrent OSC-thread "
        "insert would change its size during downstream encode iteration"
    )

    # ...and a later mutation of the live dict (exactly what `_handle_message`
    # does from the server thread) must NOT bleed into the already-returned snapshot.
    snapshot = dict(out_obj)
    node._handle_message("/new_after", 1.0)
    assert out_obj == snapshot, "returned snapshot changed when the live dict was mutated afterwards"
    assert "/new_after" not in out_obj

    # keep_messages=True semantics preserved: the live dict is NOT cleared.
    assert "/freq" in node.messages and "/new_after" in node.messages


def test_oscin_clears_live_dict_but_returns_independent_copy_when_keep_disabled():
    """keep_messages=False: the live dict is reset after the tick, and the
    returned snapshot still holds the messages (an independent copy)."""
    node = _standalone(OSCIn)
    _post_setup_state(node)
    node.params.osc.keep_messages.value = False

    node._handle_message("/freq", 10.0)

    result = node.process(None, None)
    assert result is not None
    out_obj = result["message"][0]

    # The tick consumed the messages: live dict is cleared...
    assert node.messages == {}
    # ...but the returned snapshot is its own object holding the data.
    assert out_obj is not node.messages
    assert "/freq" in out_obj
