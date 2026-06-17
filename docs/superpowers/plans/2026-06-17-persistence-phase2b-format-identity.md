# Phase 2a-B — v2 format + member_uid identity + converter — Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`. Design of record: the spec §1.1/§1.2/§2.1/§2.5/Migration.

**Goal:** Introduce the recursive `version: 2` `.gfi` envelope (definitions/instances empty for now), a stable per-node `member_uid` persisted across save/load, and a one-shot v1→v2 converter — with v1 files still loading. No user-visible behavior change; this is the format/identity groundwork for grouping (2a-C).

**Global constraints:**
- `flat_view` HARD-FAILS on non-empty `definitions`/`instances` until the expander lands in 2a-C (no silent partial load).
- `member_uid = uuid4().hex[:12]`, minted once, deduped against live refs, persisted.
- Reserved separator `::` rejected on every name ingress.
- Tests: `.venv/bin/python -m pytest tests/test_patch_format.py tests/test_manager.py -p no:cacheprovider -q`

**Doc shape (spec §2.1):**
```yaml
version: 2
definitions: {}
root:
  nodes: {name: {uid, _type, category, params, gui_kwargs}}
  links: [...]
  instances: {}
layout: <opaque>   # optional
```

## Tasks
1. `src/goofi/patch_format.py` (NEW) — `normalize_loaded`/`_v1_to_v2`/`build_v2`/`flat_view` + `tests/test_patch_format.py`.
2. `NodeRef.member_uid` / `NodeRef.membership` fields (`node_helpers.py`).
3. `Manager`: `_refs_by_uid` state; `add_node(member_uid=, membership=)` mint+dedup+reserved-name guard; maintain index on add/remove.
4. `Manager.serialize_patch` → emit v2 (uid per node); `Manager.load` → `normalize_loaded`+`flat_view`, read uid, keep layout broadcast.
5. Converter CLI `python -m goofi.patch_format migrate <glob>` + migrate `examples/*.gfi`; test idempotent + examples still load.
