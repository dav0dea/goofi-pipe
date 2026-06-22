"""Strict-mirror surfaces sibling failures instead of swallowing them (backlog #8).

Editing a shared sub-patch member mirrors the edit to every sibling instance. A
sibling that fails to apply the edit must no longer be silently swallowed (which let
a shared family drift apart unnoticed) — the failure is logged and reported.
"""
import logging

from goofi.manager import SUBPATCH_SEP

from .test_manager import _bare_manager


def test_shared_mirror_surfaces_sibling_failure(caplog):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        member1 = next(iter(mgr._instances[inst1]["members"]))
        member2 = next(iter(mgr._instances[inst2]["members"]))

        # make the sibling's update raise
        def _boom(*_a, **_k):
            raise RuntimeError("sibling unreachable")

        mgr.nodes[member2].update_param = _boom

        with caplog.at_level(logging.WARNING):
            mgr.update_param(member1, "oscillator", "frequency", 5.0)

        # the primary still applied, but the sibling failure is surfaced, not swallowed
        assert any(
            "mirror" in r.getMessage().lower() and member2 in r.getMessage()
            for r in caplog.records
        ), f"sibling mirror failure not surfaced; logs: {[r.getMessage() for r in caplog.records]}"
    finally:
        mgr.terminate(notify_gui=False)
