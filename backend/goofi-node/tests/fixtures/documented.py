import goofi


class Documented(goofi.Node):
    """Every param kind, each with help text."""

    def config_input_slots(self):
        return {"data": goofi.DataType.ARRAY}

    def config_output_slots(self):
        return {"out": goofi.DataType.ARRAY}

    def config_params(self):
        return {
            "kinds": {
                "count": goofi.IntParam(4, 1, 8, doc="how many"),
                "gain": goofi.FloatParam(1.0, 0.0, 2.0, doc="how loud"),
                "enabled": goofi.BoolParam(True, doc="whether to run"),
                "mode": goofi.StringParam("a", options=["a", "b"], doc="which mode"),
            }
        }

    def process(self, data):
        return {"out": data.data}
