//! Sub-patch forest model + the **pure projector** (Phase 1 of the sub-patch spec,
//! `docs/superpowers/specs/2026-07-16-subpatches-design.md`).
//!
//! A sub-patch is a pillar-agnostic composition primitive. The authoritative model is a
//! *definition/instance forest*: a [`SubPatchDef`] is a reusable template; an [`Instance`]
//! places one into a scope with concrete member uids. [`materialize`] is a **total,
//! side-effect-free** function projecting that forest into a flat leaf graph ([`FlatPlan`]) —
//! the deterministic image the engine `reconcile`s into its live `nodes`/`links` (Phase 5).
//! Because it touches no live engine, it is exhaustively unit-tested here.
//!
//! Only the *pure* layer lives here: types, deterministic member-uid derivation
//! ([`fold_u64`]), boundary chain-to-leaf resolution ([`resolve_boundary`]), and
//! [`materialize`]. Engine wiring (group/expand/share/reconcile, snapshot projection,
//! `.gfi`) arrives in later phases and consumes these.

use indexmap::IndexMap;

use crate::Uid;
use goofi_core::SlotType;
use goofi_node::ParamGroups;

/// A reusable sub-patch *template* id. Same 12-hex wire shape as [`Uid`] but a distinct
/// namespace (a def is never a live node).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DefId(pub u64);

impl DefId {
    pub fn to_hex(self) -> String {
        format!("{:012x}", self.0)
    }
    pub fn from_hex(s: &str) -> Option<DefId> {
        u64::from_str_radix(s, 16).ok().map(DefId)
    }
}

/// Template-local name of a member inside a def (`"osc"`, `"psd"`). Unique within a def.
pub type Local = String;
/// Stable boundary key inside a def's interface (`"in0"`, `"out0"`). Never re-minted on
/// rename, so external wires survive a relabel.
pub type BndId = String;

/// The pillar a boundary's wired inner leaf belongs to. Signal-only for now; audio/video
/// extend this without reshaping the model (spec §9). Boundaries are typed `(pillar, dtype)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pillar {
    Signal,
}

impl Pillar {
    pub fn name(self) -> &'static str {
        match self {
            Pillar::Signal => "signal",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    In,
    Out,
}

/// One exposed port of a def's interface: a naming indirection over an inner leaf slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Boundary {
    pub dir: Dir,
    /// The pillar of the wired inner leaf (used to reject cross-pillar wiring at plan time).
    pub pillar: Pillar,
    /// Resolved from the wired inner slot; the port's advertised dtype.
    pub dtype: SlotType,
    /// `(member local, member slot-or-bnd)` — `None` = UNWIRED (contributes no flat link).
    /// When the member is a nested instance, the second element is that instance's `BndId`.
    pub inner: Option<(Local, String)>,
    /// Pill position inside the entered view.
    pub pos: [f64; 2],
    /// Renameable display label (defaults `in0`/`out0`).
    pub name: String,
}

/// A param-expression declaration captured into a def (mirror of the engine's `ExprBinding`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExprDecl {
    pub group: String,
    pub name: String,
    pub source: String,
    pub enabled: bool,
    pub triggers_process: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemberDecl {
    Leaf(LeafDecl),
    /// A member that is itself a sub-patch instance.
    Nested(NestedDecl),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeafDecl {
    pub type_name: String,
    pub params: ParamGroups,
    pub expressions: Vec<ExprDecl>,
    pub pos: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct NestedDecl {
    pub def_id: DefId,
    pub pos: [f64; 2],
}

/// A link WHOLLY inside a def, between two members. An endpoint local may name a leaf member
/// (then the slot is a real slot) or a nested-instance member (then the slot is a `BndId`).
#[derive(Clone, Debug, PartialEq)]
pub struct LocalLink {
    pub out: Local,
    pub out_slot: String,
    pub in_: Local,
    pub in_slot: String,
}

/// A reusable sub-patch template.
#[derive(Clone, Debug, PartialEq)]
pub struct SubPatchDef {
    /// Display label for the palette (a saved/shared def).
    pub name: String,
    /// Insertion-ordered members (stable projection order).
    pub members: IndexMap<Local, MemberDecl>,
    /// Links wholly inside the def.
    pub links: Vec<LocalLink>,
    /// The exposed ports (interface / manifest mapping).
    pub interface: IndexMap<BndId, Boundary>,
}

/// A live placement of a def into a scope.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub uid: Uid,
    pub name: String,
    pub def_id: DefId,
    /// `None` ⇒ direct child of ROOT.
    pub parent: Option<Uid>,
    pub pos: [f64; 2],
    /// local -> live leaf/instance uid (the SSOT for uids; stable across re-projection).
    pub members: IndexMap<Local, Uid>,
}

/// Deterministic member-uid derivation: `fold(instance_uid, local)`. Stable so a member keeps
/// its uid — hence its buffers, ufreq meter, and `/data` subscriptions — across a def topology
/// re-projection. 64-bit FNV-1a over the instance uid bytes then the local's bytes.
pub fn fold_u64(instance_uid: u64, local: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for b in instance_uid.to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(PRIME);
    }
    for b in local.as_bytes() {
        h = (h ^ *b as u64).wrapping_mul(PRIME);
    }
    h
}

/// One projected leaf node in a [`FlatPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedLeaf {
    pub uid: Uid,
    pub type_name: String,
    pub params: ParamGroups,
    pub expressions: Vec<ExprDecl>,
    pub name: String,
    pub pos: [f64; 2],
    /// The instance this leaf belongs to (its scope).
    pub scope: Uid,
    pub local: Local,
}

/// One resolved leaf→leaf link in a [`FlatPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedLink {
    pub out: Uid,
    pub out_slot: String,
    pub in_: Uid,
    pub in_slot: String,
}

/// The flat projection of a forest: the sub-patch member leaves + their internal links (with
/// every boundary chain resolved to a concrete leaf). ROOT-scope user-placed leaves live
/// directly in the engine's `nodes` and are NOT part of a plan — reconcile merges the two.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FlatPlan {
    pub nodes: Vec<PlannedLeaf>,
    pub links: Vec<PlannedLink>,
}

/// The uid a member of `inst` takes: the SSOT `instance.members[local]` if bound, else the
/// deterministic derivation (reconcile's salt-rehash then records the resolved uid back).
fn member_uid(inst: &Instance, local: &str) -> Uid {
    inst.members.get(local).copied().unwrap_or_else(|| Uid(fold_u64(inst.uid.0, local)))
}

/// Chain-to-leaf: resolve `(inst_uid, bnd_id)` to the single physical inner leaf `(uid, slot)`
/// it exposes, walking through nested-instance boundaries. `None` if the boundary (or any link
/// in its chain) is unwired, or the ids don't resolve. This is the resolution the data plane
/// performs before subscribing, and that link projection uses for nested endpoints.
pub fn resolve_boundary(
    defs: &IndexMap<DefId, SubPatchDef>,
    instances: &IndexMap<Uid, Instance>,
    inst_uid: Uid,
    bnd_id: &str,
) -> Option<(Uid, String)> {
    let inst = instances.get(&inst_uid)?;
    let def = defs.get(&inst.def_id)?;
    let bnd = def.interface.get(bnd_id)?;
    let (local, slot) = bnd.inner.as_ref()?;
    match def.members.get(local)? {
        MemberDecl::Leaf(_) => Some((member_uid(inst, local), slot.clone())),
        MemberDecl::Nested(_) => {
            let nested_uid = member_uid(inst, local);
            resolve_boundary(defs, instances, nested_uid, slot)
        }
    }
}

/// Resolve one endpoint of a local link — a `(local, slot)` inside `inst`'s def — to a concrete
/// leaf `(uid, slot)`. A leaf member resolves directly; a nested-instance member resolves its
/// named boundary chain-to-leaf.
fn resolve_endpoint(
    defs: &IndexMap<DefId, SubPatchDef>,
    instances: &IndexMap<Uid, Instance>,
    inst: &Instance,
    def: &SubPatchDef,
    local: &str,
    slot: &str,
) -> Option<(Uid, String)> {
    match def.members.get(local)? {
        MemberDecl::Leaf(_) => Some((member_uid(inst, local), slot.to_string())),
        MemberDecl::Nested(_) => resolve_boundary(defs, instances, member_uid(inst, local), slot),
    }
}

/// Project the whole forest into a flat leaf plan. Total and side-effect-free: iterating every
/// instance covers nested ones (they are entries in `instances` too), so each instance emits
/// only its OWN direct leaf members and resolves its def's local links. Endpoints that don't
/// resolve (an unwired nested boundary) drop that link — never a dangling endpoint.
pub fn materialize(
    defs: &IndexMap<DefId, SubPatchDef>,
    instances: &IndexMap<Uid, Instance>,
) -> FlatPlan {
    let mut plan = FlatPlan::default();
    for inst in instances.values() {
        let Some(def) = defs.get(&inst.def_id) else { continue };
        for (local, decl) in &def.members {
            if let MemberDecl::Leaf(leaf) = decl {
                plan.nodes.push(PlannedLeaf {
                    uid: member_uid(inst, local),
                    type_name: leaf.type_name.clone(),
                    params: leaf.params.clone(),
                    expressions: leaf.expressions.clone(),
                    name: local.clone(),
                    pos: leaf.pos,
                    scope: inst.uid,
                    local: local.clone(),
                });
            }
        }
        for l in &def.links {
            let Some((out, out_slot)) = resolve_endpoint(defs, instances, inst, def, &l.out, &l.out_slot)
            else {
                continue;
            };
            let Some((in_, in_slot)) = resolve_endpoint(defs, instances, inst, def, &l.in_, &l.in_slot)
            else {
                continue;
            };
            plan.links.push(PlannedLink { out, out_slot, in_, in_slot });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(ty: &str) -> MemberDecl {
        MemberDecl::Leaf(LeafDecl {
            type_name: ty.to_string(),
            params: ParamGroups::new(),
            expressions: vec![],
            pos: [0.0, 0.0],
        })
    }

    fn out_boundary(inner_local: &str, inner_slot: &str) -> Boundary {
        Boundary {
            dir: Dir::Out,
            pillar: Pillar::Signal,
            dtype: SlotType::Array,
            inner: Some((inner_local.to_string(), inner_slot.to_string())),
            pos: [0.0, 0.0],
            name: "out0".to_string(),
        }
    }

    /// A def with an `osc -> psd` chain and an output boundary exposing psd.out.
    fn osc_psd_def() -> SubPatchDef {
        let mut members = IndexMap::new();
        members.insert("osc".to_string(), leaf("Oscillator"));
        members.insert("psd".to_string(), leaf("PSD"));
        let mut interface = IndexMap::new();
        interface.insert("out0".to_string(), out_boundary("psd", "out"));
        SubPatchDef {
            name: "Osc+PSD".to_string(),
            members,
            links: vec![LocalLink {
                out: "osc".to_string(),
                out_slot: "out".to_string(),
                in_: "psd".to_string(),
                in_slot: "data".to_string(),
            }],
            interface,
        }
    }

    fn one_instance(uid: Uid, def_id: DefId) -> Instance {
        let mut members = IndexMap::new();
        members.insert("osc".to_string(), Uid(fold_u64(uid.0, "osc")));
        members.insert("psd".to_string(), Uid(fold_u64(uid.0, "psd")));
        Instance { uid, name: "subpatch0".to_string(), def_id, parent: None, pos: [0.0, 0.0], members }
    }

    #[test]
    fn fold_is_deterministic_and_local_sensitive() {
        assert_eq!(fold_u64(5, "osc"), fold_u64(5, "osc"), "stable");
        assert_ne!(fold_u64(5, "osc"), fold_u64(5, "psd"), "local-sensitive");
        assert_ne!(fold_u64(5, "osc"), fold_u64(6, "osc"), "instance-sensitive");
    }

    #[test]
    fn def_id_hex_roundtrips_like_uid() {
        let d = DefId(0x0a1b2c);
        assert_eq!(d.to_hex().len(), 12);
        assert_eq!(DefId::from_hex(&d.to_hex()), Some(d));
    }

    #[test]
    fn materialize_projects_members_and_the_internal_link() {
        let mut defs = IndexMap::new();
        let def_id = DefId(0xdef0);
        defs.insert(def_id, osc_psd_def());
        let inst = one_instance(Uid(1), def_id);
        let mut instances = IndexMap::new();
        instances.insert(inst.uid, inst.clone());

        let plan = materialize(&defs, &instances);
        assert_eq!(plan.nodes.len(), 2, "two member leaves");
        let osc = plan.nodes.iter().find(|n| n.local == "osc").unwrap();
        let psd = plan.nodes.iter().find(|n| n.local == "psd").unwrap();
        assert_eq!(osc.uid, Uid(fold_u64(1, "osc")), "deterministic member uid");
        assert_eq!(osc.scope, Uid(1), "scoped to the instance");
        assert_eq!(plan.links, vec![PlannedLink {
            out: osc.uid,
            out_slot: "out".to_string(),
            in_: psd.uid,
            in_slot: "data".to_string(),
        }]);
        // Every planned link endpoint is a planned node (the core invariant).
        let uids: std::collections::HashSet<_> = plan.nodes.iter().map(|n| n.uid).collect();
        for l in &plan.links {
            assert!(uids.contains(&l.out) && uids.contains(&l.in_), "endpoints exist");
        }
    }

    #[test]
    fn resolve_boundary_reaches_the_inner_leaf() {
        let mut defs = IndexMap::new();
        let def_id = DefId(0xdef0);
        defs.insert(def_id, osc_psd_def());
        let inst = one_instance(Uid(1), def_id);
        let mut instances = IndexMap::new();
        instances.insert(inst.uid, inst.clone());
        assert_eq!(
            resolve_boundary(&defs, &instances, Uid(1), "out0"),
            Some((Uid(fold_u64(1, "psd")), "out".to_string())),
            "output boundary → psd.out",
        );
        assert_eq!(resolve_boundary(&defs, &instances, Uid(1), "nope"), None, "unknown bnd → None");
    }

    #[test]
    fn resolve_boundary_walks_two_deep_nesting() {
        // outer def has a nested instance member `child` whose `out0` it re-exposes as `out0`.
        let mut defs = IndexMap::new();
        let inner_def = DefId(0xd00);
        defs.insert(inner_def, osc_psd_def());

        let outer_def = DefId(0xd01);
        let mut outer_members = IndexMap::new();
        outer_members.insert("child".to_string(), MemberDecl::Nested(NestedDecl { def_id: inner_def, pos: [0.0, 0.0] }));
        let mut outer_iface = IndexMap::new();
        // re-expose the child instance's out0 boundary
        outer_iface.insert("out0".to_string(), out_boundary("child", "out0"));
        defs.insert(outer_def, SubPatchDef {
            name: "Outer".to_string(),
            members: outer_members,
            links: vec![],
            interface: outer_iface,
        });

        // outer instance uid=1; its `child` member is a nested instance uid=2.
        let child_uid = Uid(2);
        let mut outer_members_map = IndexMap::new();
        outer_members_map.insert("child".to_string(), child_uid);
        let outer = Instance {
            uid: Uid(1),
            name: "outer0".to_string(),
            def_id: outer_def,
            parent: None,
            pos: [0.0, 0.0],
            members: outer_members_map,
        };
        let child = one_instance(child_uid, inner_def);

        let mut instances = IndexMap::new();
        instances.insert(outer.uid, outer);
        instances.insert(child.uid, child);

        // outer.out0 → child.out0 → child's psd.out leaf
        assert_eq!(
            resolve_boundary(&defs, &instances, Uid(1), "out0"),
            Some((Uid(fold_u64(child_uid.0, "psd")), "out".to_string())),
            "chain walks through the nested instance to the deepest leaf",
        );
    }

    #[test]
    fn unwired_boundary_resolves_to_none() {
        let mut def = osc_psd_def();
        def.interface.get_mut("out0").unwrap().inner = None; // unwire it
        let mut defs = IndexMap::new();
        let def_id = DefId(0xdef0);
        defs.insert(def_id, def);
        let inst = one_instance(Uid(1), def_id);
        let mut instances = IndexMap::new();
        instances.insert(inst.uid, inst);
        assert_eq!(resolve_boundary(&defs, &instances, Uid(1), "out0"), None);
    }

    #[test]
    fn materialize_two_shared_siblings_have_distinct_member_uids() {
        let mut defs = IndexMap::new();
        let def_id = DefId(0xdef0);
        defs.insert(def_id, osc_psd_def());
        let mut instances = IndexMap::new();
        instances.insert(Uid(1), one_instance(Uid(1), def_id));
        instances.insert(Uid(2), one_instance(Uid(2), def_id));

        let plan = materialize(&defs, &instances);
        assert_eq!(plan.nodes.len(), 4, "two siblings × two members");
        let uids: std::collections::HashSet<_> = plan.nodes.iter().map(|n| n.uid).collect();
        assert_eq!(uids.len(), 4, "all member uids distinct across siblings");
    }
}
