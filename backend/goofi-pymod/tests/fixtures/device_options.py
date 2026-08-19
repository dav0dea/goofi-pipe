import platform

import goofi

# A REAL import at declaration time — the whole reason the probe runs the module instead of
# AST-sandboxing it. stdlib `platform`, so no extra dep.
SYSTEMS = [platform.system(), "manual"]


class DevicePick(goofi.Node):
    manifest = goofi.Manifest(
        params={"device": {"name": goofi.StringParam(SYSTEMS[0], options=SYSTEMS)}}
    )
