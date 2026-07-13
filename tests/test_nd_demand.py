"""nd() expression references create producer demand (functionally == a link)."""
import time


def _wait_until(pred, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _out_subs(mgr, uid, slot="out"):
    return (mgr.nodes[uid].serialized_state or {}).get("output_subscribers", {}).get(slot, 0)


def test_nd_reference_registers_producer_demand():
    """A node referenced ONLY by an nd() expression (no link, no viewer) must be
    made to produce to its node↔node ipc service — the reference is a first-class
    consumer, identical to a real link."""
    from tests.test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        prod = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        cons = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        assert _wait_until(lambda: all(mgr.nodes[u].stage == "ready" for u in (prod, cons)))

        prod_name = mgr.nodes[prod].name
        # No link and no viewer on prod: without the fix its OR-gate never publishes.
        assert _out_subs(mgr, prod) == 0
        mgr.set_expression(
            cons, "oscillator", "frequency",
            f'nd("{prod_name}").out.data.mean() + 1.0',
            enabled=True, autoeval=True,
        )
        # The reference now registers demand: prod has a subscriber on "out".
        assert _wait_until(lambda: _out_subs(mgr, prod) >= 1)
    finally:
        mgr.terminate(notify_gui=False)


def test_referenced_data_actually_reaches_the_consumer():
    """End to end: with demand registered, the producer publishes to its node↔node
    service and the consumer's nd() expression evaluates against real Data (no
    'NoneType has no attribute data' error) — the exact failure the user hit."""
    from tests.test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        prod = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        cons = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        assert _wait_until(lambda: all(mgr.nodes[u].stage == "ready" for u in (prod, cons)))
        prod_name = mgr.nodes[prod].name
        # .mean() on the referenced array — raises if the reference resolves to None.
        mgr.set_expression(cons, "oscillator", "frequency", f'nd("{prod_name}").out.data.mean() + 1.0', enabled=True, autoeval=True)
        assert _wait_until(lambda: _out_subs(mgr, prod) >= 1)
        # The consumer settles to NO error: the reference is delivering real Data.
        assert _wait_until(lambda: not mgr.nodes[cons].last_error)
    finally:
        mgr.terminate(notify_gui=False)


def test_clearing_the_expression_releases_demand():
    from tests.test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        prod = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        cons = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        assert _wait_until(lambda: all(mgr.nodes[u].stage == "ready" for u in (prod, cons)))
        prod_name = mgr.nodes[prod].name
        mgr.set_expression(cons, "oscillator", "frequency", f'nd("{prod_name}").out.data.mean()', enabled=True, autoeval=True)
        assert _wait_until(lambda: _out_subs(mgr, prod) >= 1)
        # Clearing the expression drops the demand back to zero.
        mgr.set_expression(cons, "oscillator", "frequency", None)
        assert _wait_until(lambda: _out_subs(mgr, prod) == 0)
    finally:
        mgr.terminate(notify_gui=False)
