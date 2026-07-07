"""LSLClient's source_name is a refreshable dropdown fed by network discovery,
replacing the old print-to-stdout refresh trigger."""
from types import SimpleNamespace

from goofi.nodes.inputs.lslclient import LSLClient


def test_source_name_is_a_refreshable_dropdown():
    params = LSLClient.config_params()
    src = params["lsl_stream"]["source_name"]
    assert src.options is not None, "source_name must be a dropdown, not free-form"
    assert src.refresh == "_refresh_lsl_sources"
    # the old print-only refresh trigger is gone
    assert "refresh" not in params["lsl_stream"]


def _info(sid):
    return SimpleNamespace(
        source_id=lambda: sid, name=lambda: "n", type=lambda: "EEG", hostname=lambda: "h"
    )


def test_refresh_lsl_sources_returns_resolved_source_ids(monkeypatch):
    node = LSLClient.create_standalone()
    fake = [_info("id-b"), _info("id-a"), _info("id-a")]  # duplicate -> deduped
    monkeypatch.setattr(node, "_resolve_stream_infos", lambda: fake)
    assert node._refresh_lsl_sources() == ["id-a", "id-b"]  # sorted + unique
