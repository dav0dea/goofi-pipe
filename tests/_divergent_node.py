"""A node whose real config default is only knowable after a real import.

`config_params` references the node's OWN class — a ClassDef, which the AST
registry never binds — so the static catalog falls to a "fallback" default,
while the real child import yields "real-a". Used to prove spawn_node feeds the
child the caller's raw overrides, never the AST-fallback params (which would
clobber the child's real _configure() default, the AudioStream device bug).
"""
from goofi.data import DataType
from goofi.node import Node
from goofi.params import StringParam


class DivergentDefault(Node):
    @staticmethod
    def real_options():
        return ["real-a", "real-b"]

    def config_params():
        try:
            opts = DivergentDefault.real_options()  # class is unbound in the AST catalog
        except Exception:
            opts = ["fallback"]
        return {"g": {"choice": StringParam(opts[0], options=opts)}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        return {"out": (0, {})}
