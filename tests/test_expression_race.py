"""Expression engine eval must be serialized against concurrent SET_EXPRESSION.

Report H2: the processing thread re-evaluates an autoeval expression every tick
while the messaging thread can set/clear that same expression. Without a lock,
ExpressionEngine.evaluate() races set_source()/close() and corrupts the engine's
subscription bookkeeping. The per-node expression lock serializes them.

The corruption is GIL-/timing-dependent, so this is a concurrency-stability
guard: under heavy concurrent edits the node must not crash or latch an error.
"""
import threading
import time

import numpy as np

from goofi.data import DataType
from goofi.params import FloatParam
from goofi.transport import set_instance_id

from .utils import make_custom_node


def test_concurrent_set_expression_and_autoeval_is_stable():
    set_instance_id("h2-expr")

    Node = make_custom_node(
        output_slots={"out": DataType.ARRAY},
        params={"grp": {"val": FloatParam(0.0, -1000.0, 1000.0)}},
        process_callback=lambda **_k: {"out": (np.array([0.0], dtype=np.float32), {})},
    )
    ref, _ = Node.create_local(
        initial_params={"common": {"autotrigger": True, "max_frequency": 120.0}}
    )
    try:
        ref.wait_for_state(timeout=2.0)
        # autoeval=True so the processing loop re-evaluates every tick.
        ref.set_expression("grp", "val", "1+1", enabled=True, autoeval=True)

        stop = threading.Event()

        def hammer():
            for i in range(600):
                ref.set_expression("grp", "val", f"{i % 7}+1", enabled=True, autoeval=True)
            stop.set()

        t = threading.Thread(target=hammer)
        t.start()
        t.join(timeout=20)
        assert not t.is_alive(), "set_expression hammering stalled"

        time.sleep(0.3)  # let the processing loop keep autoevaluating
        assert ref.last_error is None, f"node latched an error under concurrent expr edits: {ref.last_error}"
    finally:
        ref.terminate()
