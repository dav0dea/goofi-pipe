"""Strict-mirror surfaces sibling failures instead of swallowing them (backlog #8).

Editing a shared sub-patch member mirrors the edit to every sibling instance. A
sibling that fails to apply the edit must no longer be silently swallowed (which let
a shared family drift apart unnoticed) — the failure is surfaced (logged + pushed to
the UI as an error event). We assert the bridge error event, which is robust to global
logging state (unlike caplog under the full suite).
"""
from .test_manager import _bare_manager


class _Ctrl:
    def __init__(self):
        self.errors = []

    def broadcast_threadsafe(self, payload):
        self.errors.append(payload)


class _FakeBridge:
    def __init__(self):
        self.control = _Ctrl()


def test_set_expression_mirrors_and_persists_across_shared_siblings():
    # Strict mirror must cover EXPRESSION edits, not just value edits: binding an
    # expression on one shared member must reach every sibling AND the definition
    # (so a freshly-instantiated sibling inherits it).
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)
        m1 = next(iter(mgr._instances[inst1]["members"]))
        m2 = next(iter(mgr._instances[inst2]["members"]))

        mgr.set_expression(m1, "oscillator", "frequency", "5 + 5", enabled=True)

        # the existing sibling mirrors the binding...
        p2 = mgr.nodes[m2].params["oscillator"]["frequency"]
        assert p2.expression == "5 + 5"
        assert p2.expression_enabled is True

        # ...the definition persists it...
        local = mgr._instances[inst1]["members"][m1]
        defrec = mgr._definitions[def_id]["members"][local]
        assert defrec["params"]["oscillator"]["frequency"]["expression"] == "5 + 5"

        # ...so a fresh sibling inherits it too.
        inst3 = mgr.instantiate_definition(def_id)
        m3 = next(iter(mgr._instances[inst3]["members"]))
        assert mgr.nodes[m3].params["oscillator"]["frequency"].expression == "5 + 5"
    finally:
        mgr.terminate(notify_gui=False)


def test_shared_mirror_surfaces_sibling_failure():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        member1 = next(iter(mgr._instances[inst1]["members"]))
        member2 = next(iter(mgr._instances[inst2]["members"]))

        # make the sibling's mirror update raise
        def _boom(*_a, **_k):
            raise RuntimeError("sibling unreachable")

        mgr.nodes[member2].update_param = _boom

        # attach a bridge only for the edit, so its mirror failure is reported to the UI
        fake = _FakeBridge()
        mgr._bridge = fake
        mgr.update_param(member1, "oscillator", "frequency", 5.0)

        assert any(
            e.get("payload", {}).get("node") == member2 and "mirror" in e["payload"]["error"].lower()
            for e in fake.control.errors
        ), f"sibling mirror failure not surfaced to the UI; got: {fake.control.errors}"
    finally:
        mgr._bridge = None
        mgr.terminate(notify_gui=False)
