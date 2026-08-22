//! The editor's panel arrangement, held as ONE tree: a stack shows one child and draws the rest as
//! tabs, a split divides its slot between its children, a panel is a leaf. A workspace tab is a
//! child of the ROOT stack and nothing else, so there is no second kind of thing in here.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeMap;

/// A stable stack/split/panel id. Minted here, never by a client.
pub type Id = String;

/// What one entry HOLDS, addressed by id — the unit a CONTENTS command lands and inverts. It says
/// nothing about where the entry sits, which makes an inverse safe against an arrangement a peer
/// has moved under it. Only a panel holds anything: nothing in the tree carries a name, because a
/// stack derives every member's label from what that member IS.
#[derive(Clone, Debug, PartialEq)]
pub struct Contents {
    pub panel_type: String,
    pub state: Value,
}

/// See [`Contents`].
pub type Write = (Id, Contents);

/// The panel type a fresh tab starts with, and the placeholder a split births (which is what keeps
/// a split from assuming content). Both mirror `model.ts`.
pub const DEFAULT_PANEL_TYPE: &str = "node-editor";
pub const EMPTY_PANEL_TYPE: &str = "empty";

/// Smallest share a split may hand a child, so a panel can always be grabbed again (`MIN_FRACTION`).
const MIN_FRACTION: f64 = 0.05;

/// A newcomer's share of the slot it is taking over, clamped so neither side becomes ungrabbable —
/// `insertNodeAtPanel`'s `Math.max(0.05, Math.min(0.95, fraction))`.
fn fraction(ratio: f64) -> Result<f64, String> {
    if !ratio.is_finite() {
        return Err("ratio must be a number between 0 and 1".into());
    }
    Ok(ratio.clamp(MIN_FRACTION, 1.0 - MIN_FRACTION))
}

/// A split's axis. `Row` = children left→right, `Column` = top→bottom — the CSS `flex-direction`
/// spelling the renderer maps straight through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Row,
    Column,
}

/// One spelling, not two: the wire is `name`/`parse`, which the op argument already reads through.
impl Serialize for Axis {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Axis {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Axis, D::Error> {
        let s = String::deserialize(d)?;
        Axis::parse(&s).ok_or_else(|| serde::de::Error::custom("an axis is `row` or `column`"))
    }
}

impl Axis {
    pub fn name(self) -> &'static str {
        match self {
            Axis::Row => "row",
            Axis::Column => "column",
        }
    }
    pub fn parse(s: &str) -> Option<Axis> {
        match s {
            "row" => Some(Axis::Row),
            "column" => Some(Axis::Column),
            _ => None,
        }
    }
}

/// A node of the arrangement. `size` is its share of its PARENT split; a stack's children each take
/// the whole slot, and the root's is not read — it is carried anyway so a lifted subtree can name
/// the share it asks for on the way back in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Node {
    Split { id: Id, #[serde(default = "whole")] size: f64, axis: Axis, children: Vec<Node> },
    Stack { id: Id, #[serde(default = "whole")] size: f64, children: Vec<Node> },
    Panel {
        id: Id,
        #[serde(default = "whole")]
        size: f64,
        panel_type: String,
        #[serde(with = "json_string", default)]
        state: Value,
    },
}

/// A root's share of the window it fills. Also what a hand-written entry that omits one means.
fn whole() -> f64 {
    1.0
}

/// A panel's `state` rides the wire as a JSON STRING, and it must stay one: a panel clears a key
/// with an explicit `null`, and a null LEAF would make the merge-patch delta ambiguous.
mod json_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;

    pub fn serialize<S: Serializer>(v: &Value, s: S) -> Result<S::Ok, S::Error> {
        v.to_string().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        // A `state` written any other way — the natural shape of a hand edit — is REFUSED rather
        // than read as absent, which would load the panel with its binding silently wiped.
        let s = String::deserialize(d)?;
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl Node {
    pub fn id(&self) -> &str {
        match self {
            Node::Split { id, .. } | Node::Stack { id, .. } | Node::Panel { id, .. } => id,
        }
    }
    fn size(&self) -> f64 {
        match self {
            Node::Split { size, .. } | Node::Stack { size, .. } | Node::Panel { size, .. } => *size,
        }
    }
    fn set_size(&mut self, v: f64) {
        match self {
            Node::Split { size, .. } | Node::Stack { size, .. } | Node::Panel { size, .. } => *size = v,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Node::Split { .. } => "split",
            Node::Stack { .. } => "tab group",
            Node::Panel { .. } => "panel",
        }
    }
    fn children(&self) -> &[Node] {
        match self {
            Node::Split { children, .. } | Node::Stack { children, .. } => children,
            Node::Panel { .. } => &[],
        }
    }
    fn children_mut(&mut self) -> Option<&mut Vec<Node>> {
        match self {
            Node::Split { children, .. } | Node::Stack { children, .. } => Some(children),
            Node::Panel { .. } => None,
        }
    }
    /// This node and every descendant, parents before children.
    fn walk(&self) -> Vec<&Node> {
        let mut out = vec![self];
        for c in self.children() {
            out.extend(c.walk());
        }
        out
    }
    fn panels(&self) -> usize {
        match self {
            Node::Panel { .. } => 1,
            _ => self.children().iter().map(Node::panels).sum(),
        }
    }
}

/// How a subtree sits under its parent — the half of a [`Home`] that decides how to put it back.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    /// Beside its sibling, splitting along this axis.
    Beside(Axis),
    /// As a tab of the stack its sibling is in.
    Tab,
}

/// Where a subtree sat before it moved — what [`Layout::re_home`] needs to plan a move BACK without
/// restoring one slot of it. Recorded as IDS: an id either still stands or is gone.
#[derive(Clone, Debug, PartialEq)]
pub struct Home {
    /// Its old siblings, NEAREST FIRST — the landing is the first of them still standing.
    siblings: Vec<Id>,
    /// Every entry's share at capture time — what gives the arrangement its geometry back. A move
    /// disturbs more slices than one level of siblings can name.
    shares: BTreeMap<Id, f64>,
    /// The old parent's id, handed back only to a wrapper that has to be minted anyway, and only
    /// while it is free — restoring its SLOT would strand a peer's work.
    parent: Id,
    place: Place,
    /// It sat before its nearest sibling, and held this share of its parent.
    before: bool,
    size: f64,
}

impl Home {
    /// The share `id` held before the move, if this home remembers it.
    fn share(&self, id: &str) -> Option<f64> {
        self.shares.get(id).copied()
    }
}

/// A route to one node: a child index at each level down from the root. Empty = the root itself.
type Path = Vec<usize>;

/// The whole arrangement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    /// Always a stack: its children are the workspace's pages, and it is the one node that never
    /// collapses, so the strip it draws cannot vanish under its last tab.
    root: Node,
    /// The id counter, monotone: nothing ever lowers it, so a closed panel's id is never handed out
    /// again. Recycling would give a fresh panel a dead one's client-side state.
    #[serde(rename = "#seq", default)]
    seq: u64,
}

/// Two arrangements are the same when they DRAW the same; the id counter is bookkeeping.
impl PartialEq for Layout {
    fn eq(&self, other: &Layout) -> bool {
        self.root == other.root
    }
}

impl Default for Layout {
    /// The arrangement a fresh patch opens with: one page holding one node-editor panel. Also the
    /// fallback a corrupt stored arrangement lands on.
    fn default() -> Layout {
        let mut l = Layout { root: Node::Stack { id: String::new(), size: 1.0, children: vec![] }, seq: 0 };
        let root = l.mint("stack");
        let panel = l.mint("panel");
        l.root = Node::Stack {
            id: root,
            size: 1.0,
            children: vec![Node::Panel {
                id: panel,
                size: 1.0,
                panel_type: DEFAULT_PANEL_TYPE.into(),
                state: Value::Null,
            }],
        };
        l
    }
}

impl Layout {
    /// A fresh id — one counter across all three kinds. Advances the counter, so minting twice
    /// without attaching still cannot collide.
    fn mint(&mut self, prefix: &str) -> Id {
        self.seq += 1;
        format!("{prefix}-{}", self.seq)
    }

    /// Raise the counter past an id being ADMITTED rather than minted — a revive putting dead ids
    /// back, or a stored arrangement being read. Every id this arrangement held stays spent.
    fn spend(&mut self, id: &str) {
        if let Some(n) = id.rsplit_once('-').and_then(|(_, n)| n.parse::<u64>().ok()) {
            self.seq = self.seq.max(n);
        }
    }

    /// The root stack's id — what a drop on the workspace strip names.
    pub fn root_id(&self) -> &str {
        self.root.id()
    }

    /// The root stack's children, in strip order: the workspace's pages.
    pub fn pages(&self) -> Vec<Id> {
        self.root.children().iter().map(|n| n.id().to_string()).collect()
    }

    /// The route to `id`, or `None` when nothing in the arrangement carries it.
    fn path_of(&self, id: &str) -> Option<Path> {
        fn down(n: &Node, id: &str, at: &mut Vec<usize>) -> bool {
            if n.id() == id {
                return true;
            }
            for (i, c) in n.children().iter().enumerate() {
                at.push(i);
                if down(c, id, at) {
                    return true;
                }
                at.pop();
            }
            false
        }
        let mut at = Vec::new();
        down(&self.root, id, &mut at).then_some(at)
    }

    fn at(&self, p: &Path) -> &Node {
        let mut n = &self.root;
        for i in p {
            n = &n.children()[*i];
        }
        n
    }

    fn at_mut(&mut self, p: &Path) -> &mut Node {
        let mut n = &mut self.root;
        for i in p {
            n = &mut n.children_mut().expect("a path only descends containers")[*i];
        }
        n
    }

    /// The node carrying `id`, or `None`.
    pub fn node(&self, id: &str) -> Option<&Node> {
        fn down<'a>(n: &'a Node, id: &str) -> Option<&'a Node> {
            if n.id() == id {
                return Some(n);
            }
            n.children().iter().find_map(|c| down(c, id))
        }
        down(&self.root, id)
    }

    /// The same, mutably.
    fn node_mut(&mut self, id: &str) -> Option<&mut Node> {
        fn down<'a>(n: &'a mut Node, id: &str) -> Option<&'a mut Node> {
            if n.id() == id {
                return Some(n);
            }
            n.children_mut()?.iter_mut().find_map(|c| down(c, id))
        }
        down(&mut self.root, id)
    }

    /// What `id` holds right now — what a contents inverse captures before it lands.
    pub fn contents(&self, id: &str) -> Option<Contents> {
        match self.node(id) {
            Some(Node::Panel { panel_type, state, .. }) => {
                Some(Contents { panel_type: panel_type.clone(), state: state.clone() })
            }
            _ => None,
        }
    }

    /// A panel's opaque state bag, for the bind validation one layer up.
    pub fn panel_state(&self, panel: &str) -> Option<&Value> {
        match self.node(panel) {
            Some(Node::Panel { state, .. }) => Some(state),
            _ => None,
        }
    }

    /// Every node in the arrangement, parents before children.
    fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.root.walk().into_iter()
    }

    /// Every id the arrangement holds.
    fn ids(&self) -> Vec<Id> {
        self.nodes().map(|n| n.id().to_string()).collect()
    }

    /// `root` and every descendant.
    fn subtree(&self, root: &str) -> Vec<Id> {
        self.node(root)
            .map(|n| n.walk().into_iter().map(|x| x.id().to_string()).collect())
            .unwrap_or_default()
    }

    /// Adopt a planned arrangement. A structural plan is the whole next tree, so there is nothing
    /// to merge and nothing that can half-land; the id counter only ever goes up.
    pub fn apply(&mut self, next: Layout) {
        let seq = self.seq.max(next.seq);
        *self = next;
        self.seq = seq;
    }

    // --- normalisation -------------------------------------------------------------------------

    /// The rules every plan's output goes through, applied ONCE here rather than remembered at each
    /// planner: an empty container is gone, a container of ONE **is** that child, and a split inside
    /// a split along the same axis is one split. The ROOT is exempt from the second, which is what
    /// keeps the strip drawable under its last page.
    ///
    /// A stack inside a stack is deliberately left alone: it is a tab group that is one tab of an
    /// outer group, and folding it up would scatter the members of a group built inside one page.
    fn normalized(node: Node, is_root: bool) -> Option<Node> {
        let Node::Panel { .. } = &node else {
            let size = node.size();
            let axis = match &node {
                Node::Split { axis, .. } => Some(*axis),
                _ => None,
            };
            let (id, kids) = match node {
                Node::Split { id, children, .. } | Node::Stack { id, children, .. } => (id, children),
                Node::Panel { .. } => unreachable!("matched above"),
            };
            let mut out: Vec<Node> = Vec::with_capacity(kids.len());
            for child in kids {
                let Some(child) = Layout::normalized(child, false) else { continue };
                // Flatten a same-axis split into its parent, its children taking their parent's slice.
                match (axis, &child) {
                    (Some(a), Node::Split { axis: b, .. }) if a == *b => {
                        let share = child.size();
                        let Node::Split { children, .. } = child else { unreachable!() };
                        let total: f64 = children.iter().map(Node::size).sum();
                        let total = if total > 0.0 { total } else { 1.0 };
                        for mut c in children {
                            let s = c.size();
                            c.set_size(share * s / total);
                            out.push(c);
                        }
                    }
                    _ => out.push(child),
                }
            }
            if out.is_empty() {
                return is_root.then(|| Node::Stack { id, size, children: vec![] });
            }
            if out.len() == 1 && !is_root {
                let mut only = out.pop().expect("length checked");
                only.set_size(size);
                return Some(only);
            }
            let mut next = match axis {
                Some(axis) => Node::Split { id, size, axis, children: out },
                None => {
                    // A stack's children each take the WHOLE slot, so a share is meaningless on
                    // them — held at 1 rather than left to drift as members come and go.
                    let mut kids = out;
                    for c in kids.iter_mut() {
                        c.set_size(1.0);
                    }
                    Node::Stack { id, size, children: kids }
                }
            };
            Layout::rescale(&mut next);
            return Some(next);
        };
        Some(node)
    }

    /// Scale a split's children so their sizes sum to 1. A stack's children each take the whole
    /// slot, so there is nothing to scale.
    fn rescale(n: &mut Node) {
        let Node::Split { children, .. } = n else { return };
        let total: f64 = children.iter().map(Node::size).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        for c in children.iter_mut() {
            let s = c.size() / total;
            c.set_size(s);
        }
    }

    fn normalize(&mut self) {
        let root = std::mem::replace(&mut self.root, Node::Stack { id: String::new(), size: 1.0, children: vec![] });
        let id = root.id().to_string();
        self.root = Layout::normalized(root, true)
            .unwrap_or(Node::Stack { id, size: 1.0, children: vec![] });
    }

    // --- the shared halves ---------------------------------------------------------------------

    /// Lift `id` out of its parent and hand it back — the shared half of a close and a move. The
    /// freed slice goes to the siblings in proportion, and the rules do the rest.
    fn detach(&mut self, id: &str) -> Result<Node, String> {
        let Some(at) = self.path_of(id) else {
            return Err(format!("no such layout entry `{id}`"));
        };
        let Some((&mine, up)) = at.split_last() else {
            return Err("the root of the arrangement cannot be moved or closed".into());
        };
        if self.root.panels() <= self.at(&at).panels() {
            return Err(format!("`{id}` holds every panel there is — it has nowhere to go"));
        }
        let parent = self.at_mut(&up.to_vec());
        let split = matches!(parent, Node::Split { .. });
        let children = parent.children_mut().expect("a path only descends containers");
        let gone = children.remove(mine);
        // The freed slice goes to the siblings in proportion — in a SPLIT. A stack hands every
        // member the whole slot, so there is nothing to hand on.
        if split {
            let total: f64 = children.iter().map(Node::size).sum();
            let total = if total > 0.0 { total } else { 1.0 };
            for c in children.iter_mut() {
                let v = c.size();
                c.set_size(v + gone.size() * v / total);
            }
        }
        Layout::rescale(parent);
        self.normalize();
        Ok(gone)
    }

    /// The stack `id` names, WRAPPING it in a fresh one when it is not one. Dropping on a lone
    /// panel's header groups the two; dropping on a group's joins the group already there.
    fn as_stack(&mut self, id: &str, wrap: Option<&str>) -> Result<Id, String> {
        match self.node(id) {
            Some(Node::Stack { .. }) => return Ok(id.to_string()),
            Some(_) => {}
            None => return Err(format!("no such layout entry `{id}`")),
        }
        let free = wrap.filter(|w| self.node(w).is_none());
        let stack_id = match free {
            Some(w) => w.to_string(),
            None => self.mint("stack"),
        };
        let p = self.path_of(id).expect("looked up above");
        let held = std::mem::replace(self.at_mut(&p), Node::Stack { id: String::new(), size: 0.0, children: vec![] });
        let slot = held.size();
        let mut inner = held;
        inner.set_size(1.0);
        *self.at_mut(&p) = Node::Stack { id: stack_id.clone(), size: slot, children: vec![inner] };
        Ok(stack_id)
    }

    /// Put `node` beside `target` along `axis` — the ONE place split-or-wrap lives. `node`'s `size`
    /// is READ as the share it asks for; `wrap` names the id a minted wrapper takes if still free.
    fn insert_beside(&mut self, mut node: Node, target: &str, axis: Axis, before: bool, wrap: Option<&str>) {
        let Some(at) = self.path_of(target) else { return };
        let f = node.size();
        if let Some((&mine, up)) = at.split_last() {
            let parent_path: Path = up.to_vec();
            if matches!(self.at(&parent_path), Node::Split { axis: a, .. } if *a == axis) {
                let parent = self.at_mut(&parent_path);
                let children = parent.children_mut().expect("a split holds children");
                let slot = children[mine].size();
                children[mine].set_size(slot - slot * f);
                node.set_size(slot * f);
                children.insert(if before { mine } else { mine + 1 }, node);
                return;
            }
        }
        let free = wrap.filter(|w| self.node(w).is_none());
        let id = match free {
            Some(w) => w.to_string(),
            None => self.mint("split"),
        };
        let slot = self.at(&at).size();
        let held = std::mem::replace(
            self.at_mut(&at),
            Node::Panel { id: String::new(), size: 0.0, panel_type: String::new(), state: Value::Null },
        );
        let mut kept = held;
        kept.set_size(1.0 - f);
        node.set_size(f);
        let children = if before { vec![node, kept] } else { vec![kept, node] };
        *self.at_mut(&at) = Node::Split { id, size: slot, axis, children };
    }

    /// Put `node` into the stack `target` names, at `index`.
    fn insert_tab(&mut self, mut node: Node, target: &str, index: Option<usize>, wrap: Option<&str>) -> Result<(), String> {
        let stack = self.as_stack(target, wrap)?;
        let host = self.node_mut(&stack).expect("just resolved");
        let children = host.children_mut().expect("a stack holds children");
        node.set_size(1.0);
        let at = index.unwrap_or(children.len()).min(children.len());
        children.insert(at, node);
        Ok(())
    }

    /// Put `node` NEXT TO `sibling` in the stack that already holds it — the tab half of
    /// [`Self::insert_beside`], and the same rule: join the parent when it is the right kind, and
    /// wrap otherwise. Only an inverse asks for this; a drop names the stack it landed on.
    fn insert_tab_beside(&mut self, mut node: Node, sibling: &str, before: bool, wrap: Option<&str>) -> Result<(), String> {
        if let Some(at) = self.path_of(sibling) {
            if let Some((&mine, up)) = at.split_last() {
                let parent_path: Path = up.to_vec();
                if matches!(self.at(&parent_path), Node::Stack { .. }) {
                    let parent = self.at_mut(&parent_path);
                    let children = parent.children_mut().expect("a stack holds children");
                    node.set_size(1.0);
                    children.insert(if before { mine } else { mine + 1 }, node);
                    return Ok(());
                }
            }
        }
        self.insert_tab(node, sibling, None, wrap)
    }

    // --- the planners --------------------------------------------------------------------------

    /// Add a panel and return its id. With an `axis` it splits `at`, taking `ratio` of its slot and
    /// starting EMPTY — content is a choice, not an inheritance. Without one it joins the stack `at`
    /// names, starting as the default type, which is what a fresh page is.
    pub fn add_panel(
        &self,
        at: &str,
        axis: Option<Axis>,
        before: bool,
        ratio: f64,
        index: Option<usize>,
    ) -> Result<(Layout, Id), String> {
        if self.node(at).is_none() {
            return Err(format!("no such layout entry `{at}`"));
        }
        let mut next = self.clone();
        let fresh = next.mint("panel");
        match axis {
            Some(axis) => {
                let f = fraction(ratio)?;
                let born = Node::Panel {
                    id: fresh.clone(),
                    size: f,
                    panel_type: EMPTY_PANEL_TYPE.into(),
                    state: Value::Null,
                };
                next.insert_beside(born, at, axis, before, None);
            }
            None => {
                let born = Node::Panel {
                    id: fresh.clone(),
                    size: 1.0,
                    panel_type: DEFAULT_PANEL_TYPE.into(),
                    state: Value::Null,
                };
                next.insert_tab(born, at, index, None)?;
            }
        }
        next.normalize();
        Ok((next, fresh))
    }

    /// Move the subtree rooted at `root` to `to` — beside it along `axis`, or into it as a tab. One
    /// drag as ONE plan, or the user pays three ctrl-Z for it and every peer sees two arrangements.
    pub fn move_subtree(
        &self,
        root: &str,
        to: &str,
        axis: Option<Axis>,
        before: bool,
        ratio: f64,
        index: Option<usize>,
    ) -> Result<Layout, String> {
        if self.node(root).is_none() {
            return Err(format!("no such layout entry `{root}`"));
        }
        if self.node(to).is_none() {
            return Err(format!("no such layout entry `{to}`"));
        }
        if root == to {
            return Err(format!("`{root}` cannot be dropped onto itself"));
        }
        if self.subtree(root).iter().any(|d| d == to) {
            return Err(format!("`{to}` is inside `{root}` — that would make a cycle"));
        }
        let mut next = self.clone();
        // A reorder INSIDE one stack is not a lift: taking the node first can collapse the very
        // stack the move is aimed at, and the order is the only thing that changes.
        if axis.is_none() {
            if let (Some(at), Some(Node::Stack { .. })) = (next.path_of(root), next.node(to)) {
                let inside = at.split_last().map(|(_, up)| next.at(&up.to_vec()).id() == to);
                if inside == Some(true) {
                    let (from, _) = at.split_last().expect("checked above");
                    let host = next.node_mut(to).expect("checked above");
                    let children = host.children_mut().expect("a stack holds children");
                    let moved = children.remove(*from);
                    let target = index.unwrap_or(children.len()).min(children.len());
                    children.insert(target, moved);
                    return Ok(next);
                }
            }
        }
        // Lifted first: closing up behind the source can promote a sibling into the very slot the
        // newcomer is about to share.
        let mut moved = next.detach(root)?;
        match axis {
            Some(axis) => {
                let f = fraction(ratio)?;
                moved.set_size(f);
                // The lift may have collapsed the target away — a split of one promotes its child.
                if next.node(to).is_none() {
                    return Err(format!("`{to}` did not survive the move"));
                }
                next.insert_beside(moved, to, axis, before, None);
            }
            None => {
                if next.node(to).is_none() {
                    return Err(format!("`{to}` did not survive the move"));
                }
                next.insert_tab(moved, to, index, None)?;
            }
        }
        next.normalize();
        Ok(next)
    }

    /// Remove the subtree rooted at `root`, promoting and renormalizing what is left.
    pub fn remove_subtree(&self, root: &str) -> Result<Layout, String> {
        let mut next = self.clone();
        next.detach(root)?;
        Ok(next)
    }

    /// Where `root` sits now, in the terms [`Self::re_home`] needs to put it back.
    pub fn home_of(&self, root: &str) -> Option<Home> {
        let at = self.path_of(root)?;
        let (&mine, up) = at.split_last()?;
        let parent = self.at(&up.to_vec());
        let kids = parent.children();
        // Nearest first: the neighbour that shared an edge with `root` is the one whose survival
        // reconstructs the old pairing exactly.
        let mut sibs: Vec<(usize, Id)> = kids
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != mine)
            .map(|(j, n)| (j.abs_diff(mine), n.id().to_string()))
            .collect();
        sibs.sort();
        Some(Home {
            siblings: sibs.into_iter().map(|(_, k)| k).collect(),
            shares: self.shares(),
            parent: parent.id().to_string(),
            place: match parent {
                Node::Split { axis, .. } => Place::Beside(*axis),
                _ => Place::Tab,
            },
            before: mine == 0,
            size: self.at(&at).size(),
        })
    }

    /// Every node's share, by id — what a [`Home`] remembers so an undisturbed undo is exact.
    fn shares(&self) -> BTreeMap<Id, f64> {
        self.nodes().map(|n| (n.id().to_string(), n.size())).collect()
    }

    /// Plan a move of `root` back to `home`, against the arrangement AS IT STANDS — the inverse of
    /// every layout op that moves something. What it never does is restore its old parent's slot,
    /// which the move may have promoted away and a peer may have built over.
    pub fn re_home(&self, root: &str, home: &Home) -> Result<Layout, String> {
        let mut next = self.clone();
        // Lifted FIRST, so the landing is chosen among what survives closing up behind it.
        let mut e = next.detach(root)?;
        let inside = self.subtree(root);
        let landing = home
            .siblings
            .iter()
            .find(|s| next.node(s).is_some() && !inside.contains(s))
            .cloned()
            // Even its siblings are gone: the root stack always stands, so the page strip is where
            // anything homeless belongs.
            .unwrap_or_else(|| next.root.id().to_string());
        e.set_size(home.size);
        match home.place {
            Place::Beside(axis) => next.insert_beside(e, &landing, axis, home.before, Some(&home.parent)),
            Place::Tab => next.insert_tab_beside(e, &landing, home.before, Some(&home.parent))?,
        }
        next.normalize();
        next.give_back_shares(self, home);
        Ok(next)
    }

    /// Re-assert the shares `home` remembers wherever this plan disturbed them. SHARES only: where
    /// an entry sits is still re-planned, so a peer's resize elsewhere survives.
    fn give_back_shares(&mut self, before: &Layout, home: &Home) {
        let kids_of = |l: &Layout, id: &str| -> Vec<Id> {
            l.node(id).map(|n| n.children().iter().map(|c| c.id().to_string()).collect()).unwrap_or_default()
        };
        let disturbed: Vec<Id> = self
            .nodes()
            .filter(|n| matches!(n, Node::Split { .. }))
            .map(|n| n.id().to_string())
            .filter(|id| kids_of(self, id) != kids_of(before, id))
            .collect();
        for p in disturbed {
            let Some(split) = self.node_mut(&p) else { continue };
            let Some(children) = split.children_mut() else { continue };
            for c in children.iter_mut() {
                if let Some(s) = home.share(c.id()) {
                    c.set_size(s);
                }
            }
            Layout::rescale(split);
        }
    }

    /// What a close carries into its own inverse. The ids are dead the moment the close lands and
    /// nothing ever mints one again, so putting them back strands nobody.
    pub fn dead_subtree(&self, root: &str) -> Option<Node> {
        self.node(root).cloned()
    }

    /// Plan the inverse of a close: put `dead` back, then RE-PLAN where it belongs. What it never
    /// does is pin the root into the slot it held, which the close promoted away.
    pub fn revive(&self, dead: &Node, home: Option<&Home>) -> Result<Layout, String> {
        let Some(h) = home else {
            return Err(format!("`{}` is not something a close can give back", dead.id()));
        };
        let mut back = self.clone();
        for d in dead.walk() {
            back.spend(d.id());
        }
        // Parked as a page so `re_home` can LIFT it back out — the same path every other landing
        // takes, rather than a second way in.
        let root = back.root.id().to_string();
        back.insert_tab(dead.clone(), &root, None, None)?;
        back.re_home(dead.id(), h)
    }

    /// Land `writes` as CONTENTS edits. WHERE each entry sits is never touched, and an id that has
    /// since gone is skipped, so a stale replay degrades instead of resurrecting it.
    pub fn set_contents(&mut self, writes: &[Write]) {
        for (id, c) in writes {
            let Some(Node::Panel { panel_type, state, .. }) = self.node_mut(id) else { continue };
            *panel_type = c.panel_type.clone();
            *state = c.state.clone();
        }
    }

    /// A split's children's shares, in child order — what a resize's inverse aims at, read at flip
    /// time so a peer's own resize is what it puts back.
    pub fn fractions(&self, split: &str) -> Option<Vec<f64>> {
        match self.node(split) {
            Some(n @ Node::Split { .. }) => Some(n.children().iter().map(Node::size).collect()),
            _ => None,
        }
    }

    /// Set every child of `split` at once — what a resize drag commits on pointer-up. Scaling ONE
    /// child and renormalizing its siblings would never land on the fraction set the user drew.
    pub fn resize_split(&self, split: &str, fractions: &[f64]) -> Result<Layout, String> {
        let n = match self.node(split) {
            Some(n @ Node::Split { .. }) => n,
            Some(n) => return Err(format!("`{split}` is a {} — only a split divides its slot", n.kind())),
            None => return Err(format!("`{split}` is not in the arrangement")),
        };
        let kids = n.children().len();
        if kids != fractions.len() {
            return Err(format!(
                "`{split}` has {kids} children, so it needs {kids} fractions, not {}",
                fractions.len()
            ));
        }
        if fractions.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err("every fraction must be a positive number".into());
        }
        let mut next = self.clone();
        let target = next.node_mut(split).expect("looked up above");
        if let Some(children) = target.children_mut() {
            for (c, f) in children.iter_mut().zip(fractions) {
                c.set_size(f.max(MIN_FRACTION));
            }
        }
        Layout::rescale(target);
        Ok(next)
    }

    /// Clear the node binding of every panel naming a uid in `gone`. A panel's `state` is opaque
    /// here save for this one key — a panel pointing at a deleted node is the one knowable wrong.
    pub fn unbind(&self, gone: &std::collections::HashSet<crate::Uid>) -> Vec<Write> {
        let mut writes = Vec::new();
        for n in self.nodes() {
            let Node::Panel { id, panel_type, state, .. } = n else { continue };
            let bound = state.get("node").and_then(|v| v.as_str());
            if !bound.and_then(crate::Uid::from_hex).is_some_and(|u| gone.contains(&u)) {
                continue;
            }
            let mut state = state.clone();
            if let Some(o) = state.as_object_mut() {
                o.insert("node".into(), Value::Null);
            }
            writes.push((id.clone(), Contents { panel_type: panel_type.clone(), state }));
        }
        writes
    }

    /// Set a panel's type and/or state. `panel_type` lands FIRST because changing it clears the old
    /// type's state, and re-asserting the SAME type must not wipe. `state` MERGES key by key, so two
    /// writes in one round trip cannot drop the first's key.
    pub fn set_panel(
        &self,
        panel: &str,
        panel_type: Option<&str>,
        state: Option<Value>,
    ) -> Result<Vec<Write>, String> {
        let (mut pt, mut st) = match self.node(panel) {
            Some(Node::Panel { panel_type, state, .. }) => (panel_type.clone(), state.clone()),
            Some(n) => {
                return Err(format!("`{panel}` is a {} — only a panel carries a type and state", n.kind()))
            }
            None => return Err(format!("`{panel}` is not in the arrangement")),
        };
        if let Some(t) = panel_type.filter(|t| **t != pt) {
            pt = t.to_string();
            st = Value::Null;
        }
        match (state, st.as_object_mut()) {
            (Some(Value::Object(s)), Some(cur)) => cur.extend(s),
            (Some(s), _) => st = s,
            (None, _) => {}
        }
        Ok(vec![(panel.to_string(), Contents { panel_type: pt, state: st })])
    }

    /// The arrangement as plain JSON — the types' own `Serialize`, so there is no second
    /// description of the wire.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| Value::Object(Default::default()))
    }

    /// Parse a stored arrangement. The caller falls back to the default rather than refusing the
    /// patch — the graph is the value, the arrangement is chrome.
    pub fn from_json(v: &Value) -> Result<Layout, String> {
        let v = migrate_tabs(v);
        let mut l: Layout = serde_json::from_value(v).map_err(|e| format!("arrangement: {e}"))?;
        l.validate()?;
        // Every id a stored arrangement carries is SPENT, so a reopened patch never mints one twice.
        for id in l.ids() {
            l.spend(&id);
        }
        Ok(l)
    }

    /// What a TREE can still get wrong. A duplicate id is the one class a tree admits and a keyed
    /// map could not; the shapes the rules forbid have no spelling here at all.
    fn validate(&self) -> Result<(), String> {
        if !matches!(self.root, Node::Stack { .. }) {
            return Err("arrangement: the root is a tab group".into());
        }
        if self.root.children().is_empty() {
            return Err("arrangement: no pages".into());
        }
        let mut ids = std::collections::HashSet::new();
        for n in self.nodes() {
            if n.children().is_empty() && !matches!(n, Node::Panel { .. }) {
                return Err(format!("arrangement: {} `{}` holds nothing", n.kind(), n.id()));
            }
            if !ids.insert(n.id()) {
                return Err(format!("arrangement: `{}` appears twice", n.id()));
            }
        }
        Ok(())
    }
}

/// Read an arrangement written before the root became a stack: its `tabs` array is exactly the root
/// stack's children, and each tab's label is dropped — a stack derives every member's.
fn migrate_tabs(v: &Value) -> Value {
    let Some(tabs) = v.get("tabs").and_then(|t| t.as_array()) else { return v.clone() };
    let children: Vec<Value> = tabs.iter().filter_map(|t| t.get("root").cloned()).collect();
    let seq = v.get("#seq").cloned().unwrap_or(Value::from(0));
    serde_json::json!({
        "root": { "kind": "stack", "id": "stack-0", "size": 1.0, "children": children },
        "#seq": seq,
    })
}
