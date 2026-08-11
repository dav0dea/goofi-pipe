import platform

import goofi


class DevicePick(goofi.Node):
    def config_params(self):
        # A REAL import at declaration time — the whole reason the probe runs the
        # module instead of AST-sandboxing it. stdlib `platform`, so no extra dep.
        systems = [platform.system(), "manual"]
        return {"device": {"name": goofi.StringParam(systems[0], options=systems)}}
