"""Non-triggering input slots must still be drained for process() (report I2).

A slot with trigger_process=False should not, by itself, fire process() — but
when the node DOES fire (from a triggering input), process() must see the
non-triggering slot's latest data. Previously the node never attached a
non-triggering input's listener, so its slot.data stayed None forever.
"""
import time

import numpy as np

from goofi.data import DataType
from goofi.node_helpers import InputSlot
from goofi.transport import set_instance_id

from .utils import make_custom_node


def test_non_triggering_input_delivers_data_to_process():
    set_instance_id("i2-nontrig")
    const = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    Producer = make_custom_node(
        output_slots={"out": DataType.ARRAY},
        process_callback=lambda **_k: {"out": (const.copy(), {})},
    )

    nontrig_seen: list = []

    def consume(trig=None, nontrig=None):
        nontrig_seen.append(nontrig)
        return {}

    Consumer = make_custom_node(
        input_slots={
            "trig": DataType.ARRAY,
            "nontrig": InputSlot(DataType.ARRAY, trigger_process=False),
        },
        output_slots={},
        process_callback=consume,
    )

    prod_ref, _ = Producer.create_local(
        initial_params={"common": {"autotrigger": True, "max_frequency": 30.0}}
    )
    cons_ref, _ = Consumer.create_local()
    try:
        prod_ref.wait_for_state(timeout=2.0)
        cons_ref.wait_for_state(timeout=2.0)

        service = prod_ref.data_service_for("out")
        prod_ref.register_subscriber("out")
        cons_ref.subscribe_input("trig", service, in_process=False)
        cons_ref.subscribe_input("nontrig", service, in_process=False)

        deadline = time.time() + 3.0
        while time.time() < deadline and not any(x is not None for x in nontrig_seen):
            time.sleep(0.02)

        delivered = [x for x in nontrig_seen if x is not None]
        assert delivered, "non-triggering input never delivered data to process()"
        assert np.array_equal(delivered[-1].data, const)
    finally:
        prod_ref.terminate()
        cons_ref.terminate()
