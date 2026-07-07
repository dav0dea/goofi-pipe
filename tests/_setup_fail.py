"""A node whose setup() raises. The child stays alive (a setup failure idles
the node, ctrl still works), so it pushes a STATE_UPDATE with setup_complete
False — but its first PROCESSING_ERROR is message #1 and is lost to iceoryx2's
no-history. Used to prove the error rides the (idempotent, re-pushed) state
plane so the node doesn't hang on an eternal 'setting up…' spinner."""
from goofi.node import Node


class SetupFail(Node):
    def config_params():
        return {}

    def setup(self):
        raise RuntimeError("setup boom xyz")

    def process(self):
        return {}
