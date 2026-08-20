//! The editor's panel arrangement, held as a TREE: an ordered strip of tabs, each holding one root
//! node, each split holding its children in order.
//!
//! It was flat and id-keyed — every entry naming its `parent` and its `order` among that parent's
//! children — because the CRDT reconciler mirrored nested maps and scalars but ERASED nested
//! arrays, so an id-keyed map was the only shape it could carry. There is no CRDT any more, and a
//! tree makes unconstructible most of what the flat model could express and rendering could not: an
//! entry hanging off a leaf panel, two children claiming one order, a parent pointer into nothing,
//! a cycle, a tab with two roots or none. Those were five of `validate`'s seven checks; what is
//! left is what a tree can still get wrong.
//!
//! The WIRE is still flat ([`Layout::to_json`]), and so is the delta a command applies
//! ([`Write`]) — one projection of this tree, derived on the way out and validated on the way in.
//!
//! Every mutation is a PLANNER: it mutates a clone and returns the writes that turn this
//! arrangement into that one. That is what makes a layout op an ordinary command with an exact
//! inverse, and it keeps the tricky cases — a promote whose survivor is also the move's destination
//! — out of hand-rolled delta bookkeeping.

use serde_json::Value;
use std::collections::BTreeMap;

/// A stable tab/split/panel id. Minted here, never by a client.
pub type Id = String;

/// One flat entry's new value, or `None` to remove it — the unit a layout command applies and
/// inverts, and the shape [`Layout::to_json`] writes.
pub type Write = (Id, Option<Entry>);

/// The panel type a fresh tab starts with, and the placeholder a split births (which is what keeps
/// a split from assuming content). Both mirror `model.ts`.
pub const DEFAULT_PANEL_TYPE: &str = "node-editor";
pub const EMPTY_PANEL_TYPE: &str = "empty";
/// The first tab's name. Numbered from 1, like every one the client claims after it — an
/// unnumbered first name reads as a different KIND of tab beside `Tab 2`.
const DEFAULT_TAB_NAME: &str = "Tab 1";

/// Smallest share a split may hand a child, so a panel can always be grabbed again (`MIN_FRACTION`).
const MIN_FRACTION: f64 = 0.05;

/// Where the id counter rides in [`Layout::to_json`]. A minted id is always `{prefix}-{n}`, so no
/// entry can ever claim this key — which is also how [`Layout::from_json`] knows to skip it.
const SEQ_KEY: &str = "#seq";

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

/// One tab: a labelled root of the arrangement. Its POSITION in [`Layout::tabs`] is its position in
/// the strip — there is no `order` field to keep in step with it, which is one of the invariants
/// this shape makes unconstructible.
#[derive(Clone, Debug, PartialEq)]
pub struct Tab {
    pub id: Id,
    pub name: String,
    pub root: Node,
}

/// A node of one tab's tree. `size` is its share of its PARENT; a root's is not read (it fills its
/// tab), and it is carried there anyway so a lifted subtree can name the share it asks for on the
/// way back in.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Split { id: Id, size: f64, axis: Axis, children: Vec<Node> },
    Panel { id: Id, size: f64, panel_type: String, state: Value },
}

impl Node {
    pub fn id(&self) -> &str {
        match self {
            Node::Split { id, .. } | Node::Panel { id, .. } => id,
        }
    }
    fn size(&self) -> f64 {
        match self {
            Node::Split { size, .. } | Node::Panel { size, .. } => *size,
        }
    }
    fn set_size(&mut self, v: f64) {
        match self {
            Node::Split { size, .. } | Node::Panel { size, .. } => *size = v,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Node::Split { .. } => "split",
            Node::Panel { .. } => "panel",
        }
    }
    fn children(&self) -> &[Node] {
        match self {
            Node::Split { children, .. } => children,
            Node::Panel { .. } => &[],
        }
    }
    /// This node and every descendant, parents before children.
    fn walk<'a>(&'a self, out: &mut Vec<&'a Node>) {
        out.push(self);
        for c in self.children() {
            c.walk(out);
        }
    }
}

/// What a close carried away, for its own inverse to put back. A tab and a subtree are different
/// enough that the branch belongs in the type: one is re-born by NAME into the strip, the other is
/// re-homed beside a surviving sibling.
#[derive(Clone, Debug, PartialEq)]
pub enum Dead {
    Tab(Tab),
    Node(Node),
}

/// One flat entry — the wire's shape and the delta's, derived from the tree on the way out.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Tab {
        name: String,
        order: usize,
    },
    Split {
        parent: Id,
        order: usize,
        size: f64,
        axis: Axis,
    },
    Panel {
        parent: Id,
        order: usize,
        size: f64,
        panel_type: String,
        state: Value,
    },
}

impl Entry {
    fn kind(&self) -> &'static str {
        match self {
            Entry::Tab { .. } => "tab",
            Entry::Split { .. } => "split",
            Entry::Panel { .. } => "panel",
        }
    }
}

/// Where a subtree sat before it moved — everything [`Layout::re_home`] needs to plan a move BACK
/// without restoring one slot of it. Recorded as IDS rather than positions: a peer's concurrent edit
/// moves positions about, where an id either still stands or is gone.
#[derive(Clone, Debug, PartialEq)]
pub struct Home {
    /// Its old siblings, NEAREST FIRST — the landing is the first of them still standing.
    siblings: Vec<Id>,
    /// Every entry's share at capture time. The move widens the split it leaves and narrows the one
    /// it lands in; this is what gives the arrangement its geometry back. A move renormalizes the
    /// split it leaves AND the split it enters — and a promote pushes that another level up — so the
    /// set of slices it disturbs is not knowable from one level of siblings.
    shares: BTreeMap<Id, f64>,
    /// The old parent's id and axis. The id is handed back only to a wrapper that has to be minted
    /// anyway, and only while it is free — an absent split is referenced by nothing, so reusing its
    /// id strands nobody, where restoring its SLOT would.
    parent: Id,
    axis: Axis,
    /// It sat before its nearest sibling, and held this share of its parent.
    before: bool,
    size: f64,
    /// Its tab's name and strip index — the last resort, for the frozen drag where the tab went with
    /// its last panel and there is neither a sibling nor a tab left to land on.
    tab: (String, usize),
}

impl Home {
    /// The share `id` held before the move, if this home remembers it.
    fn share(&self, id: &str) -> Option<f64> {
        self.shares.get(id).copied()
    }
}

/// A route to one node: the tab's index in the strip, then a child index at each level down. Empty
/// tail = the tab's own root.
type Path = (usize, Vec<usize>);

/// The whole arrangement.
#[derive(Clone, Debug)]
pub struct Layout {
    tabs: Vec<Tab>,
    /// The id counter, monotone: nothing ever lowers it, so a closed panel's id is never handed out
    /// again. It rides [`Self::to_json`] so a reopened patch keeps counting forward. Recycling would
    /// silently give a fresh panel a dead one's client-side state, the viewpoint's `subpatchPath`
    /// among it.
    seq: u64,
}

/// Two arrangements are the same when they DRAW the same. The id counter is bookkeeping that only
/// counts up — an undo deliberately does NOT wind it back, which is the whole point of it.
impl PartialEq for Layout {
    fn eq(&self, other: &Layout) -> bool {
        self.tabs == other.tabs
    }
}

impl Default for Layout {
    /// The arrangement a fresh patch opens with: one tab holding one node-editor panel. Also the
    /// fallback a corrupt stored arrangement lands on.
    fn default() -> Layout {
        let mut l = Layout { tabs: Vec::new(), seq: 0 };
        let tab = l.mint("tab");
        let panel = l.mint("panel");
        l.tabs.push(Tab {
            id: tab,
            name: DEFAULT_TAB_NAME.into(),
            root: Node::Panel {
                id: panel,
                size: 1.0,
                panel_type: DEFAULT_PANEL_TYPE.into(),
                state: Value::Null,
            },
        });
        l
    }
}

impl Layout {
    // --- ids ---------------------------------------------------------------

    /// A fresh id — one counter across all three kinds. Advances the counter, so minting twice
    /// without attaching still cannot collide.
    fn mint(&mut self, prefix: &str) -> Id {
        self.seq += 1;
        format!("{prefix}-{}", self.seq)
    }

    /// Raise the counter past an id being ADMITTED rather than minted — a revive putting dead ids
    /// back, or a stored arrangement being read. Every id this arrangement has ever held stays
    /// spent.
    fn spend(&mut self, id: &str) {
        if let Some(n) = id.rsplit_once('-').and_then(|(_, n)| n.parse::<u64>().ok()) {
            self.seq = self.seq.max(n);
        }
    }

    // --- navigation --------------------------------------------------------

    /// The tabs, in strip order.
    pub fn tabs(&self) -> Vec<Id> {
        self.tabs.iter().map(|t| t.id.clone()).collect()
    }

    /// Where `tab` sits in the strip — what a reorder's inverse aims at, read at flip time so a
    /// peer's added tab has already been counted.
    pub fn tab_index(&self, tab: &str) -> Option<usize> {
        self.tabs.iter().position(|t| t.id == tab)
    }

    pub fn tab_named(&self, name: &str) -> Option<Id> {
        self.tabs.iter().find(|t| t.name == name).map(|t| t.id.clone())
    }

    pub fn name_of(&self, tab: &str) -> Option<&str> {
        self.tabs.iter().find(|t| t.id == tab).map(|t| t.name.as_str())
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
        for (i, t) in self.tabs.iter().enumerate() {
            if t.id == id {
                return None; // a tab is not a node; `tab_index` answers for it
            }
            let mut at = Vec::new();
            if down(&t.root, id, &mut at) {
                return Some((i, at));
            }
        }
        None
    }

    fn at(&self, p: &Path) -> &Node {
        let mut n = &self.tabs[p.0].root;
        for i in &p.1 {
            n = &n.children()[*i];
        }
        n
    }

    fn at_mut(&mut self, p: &Path) -> &mut Node {
        let mut n = &mut self.tabs[p.0].root;
        for i in &p.1 {
            let Node::Split { children, .. } = n else { unreachable!("a path only descends splits") };
            n = &mut children[*i];
        }
        n
    }

    /// The node carrying `id`, or `None` — a tab is not one.
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.path_of(id).map(|p| self.at(&p))
    }

    /// The tab `id` sits on. `id` may name the tab itself.
    pub fn tab_of(&self, id: &str) -> Option<Id> {
        if self.tab_index(id).is_some() {
            return Some(id.to_string());
        }
        self.path_of(id).map(|p| self.tabs[p.0].id.clone())
    }

    /// The id of a tab's root — what a caller gives content to after adding one.
    pub fn root_of(&self, tab: &str) -> Option<Id> {
        self.tabs.iter().find(|t| t.id == tab).map(|t| t.root.id().to_string())
    }

    /// One id's FLAT entry, cloned — what a contents inverse captures before it lands.
    pub fn entry(&self, id: &str) -> Option<Entry> {
        self.flat().remove(id)
    }

    /// A panel's opaque state bag, for the bind validation one layer up.
    pub fn panel_state(&self, panel: &str) -> Option<&Value> {
        match self.node(panel) {
            Some(Node::Panel { state, .. }) => Some(state),
            _ => None,
        }
    }

    /// `root` and every descendant, parents before children. `root` may name a tab, in which case
    /// the tab's id leads and its tree follows.
    fn subtree(&self, root: &str) -> Vec<Id> {
        if let Some(t) = self.tabs.iter().find(|t| t.id == root) {
            let mut out = vec![t.id.clone()];
            let mut ns = Vec::new();
            t.root.walk(&mut ns);
            out.extend(ns.into_iter().map(|n| n.id().to_string()));
            return out;
        }
        match self.node(root) {
            Some(n) => {
                let mut ns = Vec::new();
                n.walk(&mut ns);
                ns.into_iter().map(|x| x.id().to_string()).collect()
            }
            None => Vec::new(),
        }
    }

    // --- the flat projection ------------------------------------------------

    /// This tree as the flat map the wire and the delta speak. The ONE place the two shapes meet.
    fn flat(&self) -> BTreeMap<Id, Entry> {
        fn down(n: &Node, parent: &str, order: usize, out: &mut BTreeMap<Id, Entry>) {
            match n {
                Node::Split { id, size, axis, children } => {
                    out.insert(
                        id.clone(),
                        Entry::Split { parent: parent.into(), order, size: *size, axis: *axis },
                    );
                    for (i, c) in children.iter().enumerate() {
                        down(c, id, i, out);
                    }
                }
                Node::Panel { id, size, panel_type, state } => {
                    out.insert(
                        id.clone(),
                        Entry::Panel {
                            parent: parent.into(),
                            order,
                            size: *size,
                            panel_type: panel_type.clone(),
                            state: state.clone(),
                        },
                    );
                }
            }
        }
        let mut out = BTreeMap::new();
        for (i, t) in self.tabs.iter().enumerate() {
            out.insert(t.id.clone(), Entry::Tab { name: t.name.clone(), order: i });
            down(&t.root, &t.id, 0, &mut out);
        }
        out
    }

    /// Rebuild a tree from the flat map — the wire's direction, and the one place every shape a
    /// flat arrangement can take and a tree cannot is refused.
    fn from_flat(entries: &BTreeMap<Id, Entry>, seq: u64) -> Result<Layout, String> {
        let kids = |parent: &str| -> Vec<Id> {
            let mut v: Vec<(usize, &Id)> = entries
                .iter()
                .filter(|(_, e)| match e {
                    Entry::Split { parent: p, .. } | Entry::Panel { parent: p, .. } => p == parent,
                    Entry::Tab { .. } => false,
                })
                .map(|(id, e)| {
                    let o = match e {
                        Entry::Split { order, .. } | Entry::Panel { order, .. } => *order,
                        Entry::Tab { order, .. } => *order,
                    };
                    (o, id)
                })
                .collect();
            v.sort();
            v.into_iter().map(|(_, id)| id.clone()).collect()
        };
        // Depth-bounded by the entry count, so a parent cycle cannot spin here — it simply fails to
        // reach a tab, and every entry is counted at the end.
        fn build(
            id: &str,
            entries: &BTreeMap<Id, Entry>,
            kids: &dyn Fn(&str) -> Vec<Id>,
            depth: usize,
            seen: &mut Vec<Id>,
        ) -> Result<Node, String> {
            if depth > entries.len() {
                return Err(format!("arrangement: `{id}` is inside a cycle"));
            }
            seen.push(id.to_string());
            match entries.get(id) {
                Some(Entry::Split { size, axis, .. }) => {
                    let mut children = Vec::new();
                    for k in kids(id) {
                        children.push(build(&k, entries, kids, depth + 1, seen)?);
                    }
                    if children.is_empty() {
                        return Err(format!("arrangement: split `{id}` divides nothing"));
                    }
                    Ok(Node::Split { id: id.into(), size: *size, axis: *axis, children })
                }
                Some(Entry::Panel { size, panel_type, state, .. }) => {
                    if !kids(id).is_empty() {
                        return Err(format!("arrangement: `{id}` is a panel, and panels are leaves"));
                    }
                    Ok(Node::Panel {
                        id: id.into(),
                        size: *size,
                        panel_type: panel_type.clone(),
                        state: state.clone(),
                    })
                }
                Some(Entry::Tab { .. }) => Err(format!("arrangement: tab `{id}` is inside a tree")),
                None => Err(format!("arrangement: `{id}` is not in the arrangement")),
            }
        }

        // Reachability FIRST, because an entry that reaches no tab is the CAUSE of the tab-root
        // count being wrong, and a caller shown only the symptom cannot find the entry.
        for (id, e) in entries {
            let mut cur = match e {
                Entry::Tab { .. } => continue,
                Entry::Split { parent, .. } | Entry::Panel { parent, .. } => parent,
            };
            let mut hops = 0;
            loop {
                match entries.get(cur) {
                    Some(Entry::Tab { .. }) => break,
                    Some(Entry::Split { parent, .. }) | Some(Entry::Panel { parent, .. })
                        if hops <= entries.len() =>
                    {
                        cur = parent;
                        hops += 1;
                    }
                    _ => {
                        return Err(format!(
                            "arrangement: `{id}` reaches no tab — a missing parent, or a cycle"
                        ))
                    }
                }
            }
        }

        let mut strip: Vec<(usize, &Id, &String)> = entries
            .iter()
            .filter_map(|(id, e)| match e {
                Entry::Tab { name, order } => Some((*order, id, name)),
                _ => None,
            })
            .collect();
        strip.sort();
        if strip.is_empty() {
            return Err("arrangement: no tabs".into());
        }
        let mut names = std::collections::HashSet::new();
        let mut seen: Vec<Id> = Vec::new();
        let mut tabs = Vec::new();
        for (_, id, name) in strip {
            if !names.insert(name) {
                return Err(format!("arrangement: two tabs are both named `{name}`"));
            }
            seen.push(id.clone());
            let roots = kids(id);
            // A tab holds exactly one root — the tree's own shape. Zero or many is what flattening
            // admitted and rendering could not.
            let [root] = roots.as_slice() else {
                return Err(format!("arrangement: tab `{name}` has {} roots, not 1", roots.len()));
            };
            tabs.push(Tab {
                id: id.clone(),
                name: name.clone(),
                root: build(root, entries, &kids, 0, &mut seen)?,
            });
        }
        let mut l = Layout { tabs, seq };
        for id in seen {
            l.spend(&id);
        }
        Ok(l)
    }

    /// Apply a planner's writes wholesale. Every producer hands over a plan that lands as a whole —
    /// there is no per-entry command any more — so a write list that does not describe a tree is a
    /// planner bug, and the arrangement is left standing rather than half-written.
    pub fn apply(&mut self, writes: Vec<Write>) {
        let mut flat = self.flat();
        for (id, entry) in writes {
            match entry {
                Some(e) => flat.insert(id, e),
                None => flat.remove(&id),
            };
        }
        if let Ok(next) = Layout::from_flat(&flat, self.seq) {
            *self = next;
        }
    }

    /// The per-entry writes that turn `self` into `next` — every planner's return value.
    fn diff(&self, next: &Layout) -> Vec<Write> {
        let (a, b) = (self.flat(), next.flat());
        let mut w: Vec<Write> = b
            .iter()
            .filter(|(id, e)| a.get(*id) != Some(e))
            .map(|(id, e)| (id.clone(), Some(e.clone())))
            .collect();
        w.extend(a.keys().filter(|id| !b.contains_key(*id)).map(|id| (id.clone(), None)));
        w
    }

    // --- tree surgery -------------------------------------------------------

    /// Scale a split's children so their sizes sum to 1.
    fn normalize(n: &mut Node) {
        let Node::Split { children, .. } = n else { return };
        let total: f64 = children.iter().map(Node::size).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        for c in children.iter_mut() {
            let s = c.size() / total;
            c.set_size(s);
        }
    }

    /// Lift `id` out of its parent and hand it back — the shared half of a close and a move. The
    /// freed slice goes to the siblings in proportion, and a split left with ONE child is replaced
    /// by that child in its own slot, so the tree never keeps a one-armed wrapper. The subtree
    /// hanging off `id` is untouched (a move re-attaches it whole).
    fn detach(&mut self, id: &str) -> Result<Node, String> {
        let Some((tab, at)) = self.path_of(id) else {
            return Err(format!("no such panel `{id}`"));
        };
        let Some((&mine, up)) = at.split_last() else {
            return Err(format!(
                "`{id}` is tab `{}`'s only root — a tab always keeps one",
                self.tabs[tab].name
            ));
        };
        let parent_path: Path = (tab, up.to_vec());
        let parent = self.at_mut(&parent_path);
        let Node::Split { children, .. } = parent else { unreachable!("a path only descends splits") };
        let gone = children.remove(mine);
        let total: f64 = children.iter().map(Node::size).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        for c in children.iter_mut() {
            let v = c.size();
            c.set_size(v + gone.size() * v / total);
        }
        Layout::normalize(parent);
        // One child left: it takes its parent's whole slot, and the wrapper goes.
        let lone = matches!(parent, Node::Split { children, .. } if children.len() == 1);
        if lone {
            let slot = parent.size();
            let Node::Split { children, .. } = parent else { unreachable!() };
            let mut survivor = children.remove(0);
            survivor.set_size(slot);
            *parent = survivor;
        }
        Ok(gone)
    }

    /// Put `node` under the split at `parent` at `index`. The newcomer takes an EQUAL share (the
    /// average of what is there) and the siblings keep their relative proportions.
    fn attach(&mut self, mut node: Node, parent: &str, index: usize) -> Result<(), String> {
        let Some(p) = self.path_of(parent) else {
            return Err(format!("no such layout entry `{parent}`"));
        };
        let target = self.at_mut(&p);
        let Node::Split { children, .. } = target else {
            return Err(format!(
                "`{parent}` is a {} — a subtree moves into a split (split a panel first)",
                target.kind()
            ));
        };
        node.set_size(if children.is_empty() { 1.0 } else { 1.0 / children.len() as f64 });
        children.insert(index.min(children.len()), node);
        Layout::normalize(target);
        Ok(())
    }

    /// Put `node` beside `target` along `axis` — the ONE place split-or-wrap lives. A parent already
    /// running along `axis` gains a sibling; otherwise the target is wrapped in a fresh split
    /// inheriting its slot. `node`'s `size` is READ as the share it asks for, so a caller hands over
    /// a lifted subtree or a new panel indifferently. `wrap` names the id a minted wrapper takes if
    /// still free — how an undo gives a promoted-away split its id back without restoring its slot.
    fn insert_at(&mut self, mut node: Node, target: &str, axis: Axis, before: bool, wrap: Option<&str>) {
        let Some((tab, at)) = self.path_of(target) else { return };
        let f = node.size();
        // A parent running along the same axis absorbs the newcomer as a sibling.
        if let Some((&mine, up)) = at.split_last() {
            let parent_path: Path = (tab, up.to_vec());
            if matches!(self.at(&parent_path), Node::Split { axis: a, .. } if *a == axis) {
                let parent = self.at_mut(&parent_path);
                let Node::Split { children, .. } = parent else { unreachable!() };
                let slot = children[mine].size();
                children[mine].set_size(slot - slot * f);
                node.set_size(slot * f);
                children.insert(if before { mine } else { mine + 1 }, node);
                return;
            }
        }
        // Otherwise the target is wrapped, and the wrapper inherits its slot.
        let free = wrap.filter(|w| self.path_of(w).is_none() && self.tab_index(w).is_none());
        let id = match free {
            Some(w) => w.to_string(),
            None => self.mint("split"),
        };
        let p = (tab, at);
        let slot = self.at(&p).size();
        let held = std::mem::replace(
            self.at_mut(&p),
            Node::Panel { id: String::new(), size: 0.0, panel_type: String::new(), state: Value::Null },
        );
        let mut kept = held;
        kept.set_size(1.0 - f);
        node.set_size(f);
        let children = if before { vec![node, kept] } else { vec![kept, node] };
        *self.at_mut(&p) = Node::Split { id, size: slot, axis, children };
    }

    /// Lift a subtree out for re-homing. Normally a [`Self::detach`] — but when it is its tab's ONLY
    /// root the TAB goes with it, which is the frozen "the panel was the tab's only node → the tab
    /// goes with it" branch of `_takeNode`. The last tab never goes.
    fn take(&mut self, root: &str) -> Result<Node, String> {
        if self.tab_index(root).is_some() {
            return Err("a tab is not a subtree — reorder it with reorder_tab".into());
        }
        let Some((tab, at)) = self.path_of(root) else {
            return Err(format!("no such panel `{root}`"));
        };
        if !at.is_empty() {
            return self.detach(root);
        }
        if self.tabs.len() <= 1 {
            return Err(format!("`{root}` is the only panel on the only tab — it has nowhere to go"));
        }
        Ok(self.tabs.remove(tab).root)
    }

    // --- planners -----------------------------------------------------------

    /// Add a tab and return its id. It holds one fresh node-editor panel — unless `subtree` names an
    /// existing one, in which case the tab is built AROUND it: the frozen drop-onto-the-tab-bar
    /// gesture, which `add_tab` + `move_panel` cannot express (a move needs a split to land in, and
    /// a fresh tab has none). `index` places it in the strip.
    pub fn add_tab(
        &self,
        name: &str,
        index: Option<usize>,
        subtree: Option<&str>,
    ) -> Result<(Vec<Write>, Id), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("a tab needs a name".into());
        }
        if self.tab_named(name).is_some() {
            return Err(format!("a tab named `{name}` already exists"));
        }
        let mut next = self.clone();
        // Lifted FIRST, because taking a tab's last panel takes the tab — which is what the new
        // tab's own position is counted against.
        let adopted = match subtree {
            Some(s) => Some(next.take(s)?),
            None => None,
        };
        let id = next.mint("tab");
        let mut root = adopted.unwrap_or_else(|| {
            let panel = next.mint("panel");
            Node::Panel {
                id: panel,
                size: 1.0,
                panel_type: DEFAULT_PANEL_TYPE.into(),
                state: Value::Null,
            }
        });
        root.set_size(1.0);
        let at = index.unwrap_or(next.tabs.len()).min(next.tabs.len());
        next.tabs.insert(at, Tab { id: id.clone(), name: name.to_string(), root });
        Ok((self.diff(&next), id))
    }

    /// Re-home the subtree rooted at `subtree` beside `target`, splitting along `axis` —
    /// `dropOnPanel` as ONE plan. Three ops would cost the user three ctrl-Z for one drag and show
    /// every peer two arrangements that were never on screen.
    pub fn insert_at_panel(
        &self,
        subtree: &str,
        target: &str,
        axis: Axis,
        before: bool,
        ratio: f64,
    ) -> Result<Vec<Write>, String> {
        // A PANEL target is what the gesture means AND what makes the plan safe: lifting the source
        // can promote a split away, but never a panel, so the target still stands afterwards.
        match self.node(target) {
            Some(Node::Panel { .. }) => {}
            Some(n) => return Err(format!("`{target}` is a {} — a drop lands on a panel", n.kind())),
            None => return Err(format!("no such panel `{target}`")),
        }
        if subtree == target {
            return Err(format!("`{subtree}` cannot be dropped onto itself"));
        }
        if self.subtree(subtree).iter().any(|d| d == target) {
            return Err(format!("`{target}` is inside `{subtree}` — that would make a cycle"));
        }
        let f = fraction(ratio)?;
        let mut next = self.clone();
        // Lifted first, exactly as `_takeNode` runs before `insertNodeAtPanel`: closing up behind
        // the source can promote a sibling into the slot the newcomer is about to share.
        let mut moved = next.take(subtree)?;
        moved.set_size(f);
        next.insert_at(moved, target, axis, before, None);
        Ok(self.diff(&next))
    }

    /// Where `root` sits now, in the terms [`Self::re_home`] needs to put it back. `None` for a tab,
    /// which is reordered rather than moved.
    pub fn home_of(&self, root: &str) -> Option<Home> {
        let (tab, at) = self.path_of(root)?;
        let Some((&mine, up)) = at.split_last() else {
            // A tab's ROOT has no split above it, and its home is the tab. Flattening gave it a
            // parent pointer here and the tree does not — without this branch a torn-off panel's
            // undo captured no home at all and silently degraded to a no-op.
            return Some(Home {
                siblings: Vec::new(),
                shares: self.shares(),
                parent: self.tabs[tab].id.clone(),
                axis: Axis::Row,
                before: true,
                size: self.tabs[tab].root.size(),
                tab: (self.tabs[tab].name.clone(), tab),
            });
        };
        let parent = self.at(&(tab, up.to_vec()));
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
            axis: match parent {
                Node::Split { axis, .. } => *axis,
                _ => Axis::Row,
            },
            before: mine == 0,
            size: self.at(&(tab, at.clone())).size(),
            tab: (self.tabs[tab].name.clone(), tab),
        })
    }

    /// Every node's share, by id — what a [`Home`] remembers so an undisturbed undo is exact.
    fn shares(&self) -> BTreeMap<Id, f64> {
        let mut out = BTreeMap::new();
        for t in &self.tabs {
            let mut ns = Vec::new();
            t.root.walk(&mut ns);
            for n in ns {
                out.insert(n.id().to_string(), n.size());
            }
        }
        out
    }

    /// Plan a move of `root` back to `home`, against the arrangement AS IT STANDS — the inverse of
    /// every layout op that moves something. It lands beside the first old sibling still standing,
    /// else beside its old tab's current root, else inside a tab re-born around it. What it never
    /// does is restore its old parent's slot: the move may have promoted that split away, and a peer
    /// may have built on whatever took its place, which a restore would strand.
    pub fn re_home(&self, root: &str, home: &Home) -> Result<Vec<Write>, String> {
        let mut next = self.clone();
        // Lifted FIRST, so the landing is chosen among what survives closing up behind it.
        let e = next.take(root)?;
        let inside = self.subtree(root);
        let landing = home
            .siblings
            .iter()
            .find(|s| next.path_of(s).is_some() && !inside.contains(s))
            .cloned()
            .or_else(|| next.tab_named(&home.tab.0).and_then(|t| next.root_of(&t)));
        let Some(landing) = landing else {
            // Even the tab went with it (the tab followed its last panel) — re-born AROUND the
            // subtree, which is `add_tab`'s own adopt branch rather than a raw restore. Lifting it
            // out still widens the split it leaves, so the shares are given back the same way.
            let (writes, _) = self.add_tab(&home.tab.0, Some(home.tab.1), Some(root))?;
            let mut born = self.clone();
            born.apply(writes);
            born.give_back_shares(self, home);
            return Ok(self.diff(&born));
        };
        let mut e = e;
        e.set_size(home.size);
        next.insert_at(e, &landing, home.axis, home.before, Some(&home.parent));
        next.give_back_shares(self, home);
        Ok(self.diff(&next))
    }

    /// Re-assert the shares `home` remembers wherever this plan disturbed them — what makes an
    /// undisturbed undo exact to the pixel. SHARES only: where an entry sits is still re-planned
    /// and never restored. A split the plan left alone keeps what it holds, so a peer's resize
    /// elsewhere survives.
    fn give_back_shares(&mut self, before: &Layout, home: &Home) {
        let kids_of = |l: &Layout, id: &str| -> Vec<Id> {
            l.node(id).map(|n| n.children().iter().map(|c| c.id().to_string()).collect()).unwrap_or_default()
        };
        let mut disturbed: Vec<Id> = Vec::new();
        for t in &self.tabs {
            let mut ns = Vec::new();
            t.root.walk(&mut ns);
            for n in ns {
                if matches!(n, Node::Split { .. }) {
                    let id = n.id().to_string();
                    if kids_of(self, &id) != kids_of(before, &id) {
                        disturbed.push(id);
                    }
                }
            }
        }
        for p in disturbed {
            let Some(path) = self.path_of(&p) else { continue };
            let split = self.at_mut(&path);
            let Node::Split { children, .. } = split else { continue };
            for c in children.iter_mut() {
                if let Some(s) = home.share(c.id()) {
                    c.set_size(s);
                }
            }
            Layout::normalize(split);
        }
    }

    /// What a close carries into its own inverse. The ids are dead the moment the close lands and
    /// nothing ever mints one again, so putting them back strands nobody; WHERE the root lands is
    /// [`Self::revive`]'s question, not theirs.
    pub fn dead_subtree(&self, root: &str) -> Option<Dead> {
        if let Some(t) = self.tabs.iter().find(|t| t.id == root) {
            return Some(Dead::Tab(t.clone()));
        }
        self.node(root).cloned().map(Dead::Node)
    }

    /// Plan the inverse of a close: put `dead` back, then RE-PLAN where it belongs —
    /// [`Self::re_home`] for a subtree, the strip for a tab. What it never does is pin the root into
    /// the slot it held: the close promoted that split away, a peer may have built where it stood,
    /// and a later undo may even have handed its id to a live wrapper.
    pub fn revive(&self, dead: &Dead, home: Option<&Home>) -> Result<Vec<Write>, String> {
        let mut back = self.clone();
        match dead {
            // A tab hangs off nothing, so only its place in the strip needs re-planning — a peer's
            // new tab has taken an index since, and restoring the old one collides with it.
            Dead::Tab(t) => {
                if back.tab_named(&t.name).is_some() {
                    return Err(format!("a tab named `{}` already exists", t.name));
                }
                for id in {
                    let mut ns = Vec::new();
                    t.root.walk(&mut ns);
                    ns.into_iter().map(|n| n.id().to_string()).collect::<Vec<_>>()
                } {
                    back.spend(&id);
                }
                back.spend(&t.id);
                let at = home.map(|h| h.tab.1).unwrap_or(back.tabs.len()).min(back.tabs.len());
                back.tabs.insert(at, t.clone());
                Ok(self.diff(&back))
            }
            Dead::Node(n) => {
                let Some(h) = home else {
                    return Err(format!("`{}` is not something a close can give back", n.id()));
                };
                // Parked on the first tab's root so `re_home` can LIFT it back out — the same path
                // every other landing takes, rather than a second way in.
                let mut ns = Vec::new();
                n.walk(&mut ns);
                for id in ns.into_iter().map(|x| x.id().to_string()).collect::<Vec<_>>() {
                    back.spend(&id);
                }
                let anchor = back.tabs[0].root.id().to_string();
                back.insert_at(n.clone(), &anchor, h.axis, h.before, None);
                let writes = back.re_home(n.id(), h)?;
                back.apply(writes);
                Ok(self.diff(&back))
            }
        }
    }

    /// Land `writes` as CONTENTS edits: what each entry HOLDS arrives, but WHERE it sits is read off
    /// the arrangement as it stands and never off the write. That is what makes the inverse of a
    /// type change safe — the slot an entry held at plan time may be a peer's by undo time. An id
    /// that has since gone is skipped, so a stale replay degrades instead of resurrecting it.
    pub fn set_contents(&self, writes: &[Write]) -> Vec<Write> {
        let mut next = self.clone();
        for (id, entry) in writes {
            match entry {
                // A tab's contents is its LABEL; its position in the strip is where it sits, and a
                // reorder is the op for that.
                Some(Entry::Tab { name, .. }) => {
                    if let Some(i) = next.tab_index(id) {
                        next.tabs[i].name = name.clone();
                    }
                }
                Some(Entry::Panel { panel_type, state, .. }) => {
                    let Some(p) = next.path_of(id) else { continue };
                    if let Node::Panel { panel_type: pt, state: st, .. } = next.at_mut(&p) {
                        *pt = panel_type.clone();
                        *st = state.clone();
                    }
                }
                // A split holds only its axis, and no op edits one.
                Some(Entry::Split { .. }) | None => {}
            }
        }
        self.diff(&next)
    }

    /// Set every child of `split` at once — what a resize drag commits on pointer-up, and the only
    /// op that sizes anything. Scaling ONE child and renormalizing its siblings would make N of them
    /// chase a moving target and never land on the fraction set the user drew.
    pub fn resize_split(&self, split: &str, fractions: &[f64]) -> Result<Vec<Write>, String> {
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
        let p = next.path_of(split).expect("looked up above");
        let target = next.at_mut(&p);
        if let Node::Split { children, .. } = target {
            for (c, f) in children.iter_mut().zip(fractions) {
                c.set_size(f.max(MIN_FRACTION));
            }
        }
        Layout::normalize(target);
        Ok(self.diff(&next))
    }

    /// Refuse an id that is not a tab. Every tab op addresses BY ID — a name is what the tab holds,
    /// not how it is found, so renaming one cannot make a caller's next op miss.
    fn is_tab(&self, tab: &str) -> Result<usize, String> {
        match self.tab_index(tab) {
            Some(i) => Ok(i),
            None if self.path_of(tab).is_some() => Err(format!("`{tab}` is not a tab")),
            None => Err(format!("no such tab `{tab}`")),
        }
    }

    pub fn remove_tab(&self, tab: &str) -> Result<Vec<Write>, String> {
        let i = self.is_tab(tab)?;
        if self.tabs.len() <= 1 {
            return Err("the last tab cannot be removed".into());
        }
        let mut next = self.clone();
        next.tabs.remove(i);
        Ok(self.diff(&next))
    }

    /// Relabel a tab. A field edit, so the id — and therefore every panel on it — stands.
    pub fn rename_tab(&self, tab: &str, to: &str) -> Result<Vec<Write>, String> {
        let i = self.is_tab(tab)?;
        let to = to.trim();
        if to.is_empty() {
            return Err("a tab needs a name".into());
        }
        if self.tab_named(to).is_some_and(|other| other != tab) {
            return Err(format!("a tab named `{to}` already exists"));
        }
        let mut next = self.clone();
        next.tabs[i].name = to.to_string();
        Ok(self.diff(&next))
    }

    pub fn reorder_tab(&self, tab: &str, to_index: usize) -> Result<Vec<Write>, String> {
        let i = self.is_tab(tab)?;
        let mut next = self.clone();
        let t = next.tabs.remove(i);
        let at = to_index.min(next.tabs.len());
        next.tabs.insert(at, t);
        Ok(self.diff(&next))
    }

    /// Split `panel` along `axis`, birthing an EMPTY panel that takes `ratio` of its slot — the same
    /// [`Self::insert_at`] a drop uses, handed a brand-new panel instead of a lifted subtree.
    pub fn split_panel(
        &self,
        panel: &str,
        axis: Axis,
        place_before: bool,
        ratio: f64,
    ) -> Result<(Vec<Write>, Id), String> {
        match self.node(panel) {
            Some(Node::Panel { .. }) => {}
            Some(n) => return Err(format!("`{panel}` is a {} — only a panel splits", n.kind())),
            None => return Err(format!("no such panel `{panel}`")),
        }
        let f = fraction(ratio)?;
        let mut next = self.clone();
        let fresh = next.mint("panel");
        let born = Node::Panel {
            id: fresh.clone(),
            size: f,
            panel_type: EMPTY_PANEL_TYPE.into(),
            state: Value::Null,
        };
        next.insert_at(born, panel, axis, place_before, None);
        Ok((self.diff(&next), fresh))
    }

    /// Clear the node binding of every panel naming a uid in `gone`, as the writes a
    /// [`crate::Command::LayoutContents`] lands. A panel's `state` is opaque here save for this one
    /// key, which the frontend and the bind validation already share (`set_panel`'s `state.node`) —
    /// a panel pointing at a deleted node is the one arrangement the manager can know is wrong.
    pub fn unbind(&self, gone: &std::collections::HashSet<crate::Uid>) -> Vec<Write> {
        let mut writes = Vec::new();
        for (id, e) in self.flat() {
            let Entry::Panel { state, .. } = &e else { continue };
            let bound = state.get("node").and_then(|v| v.as_str());
            if !bound.and_then(crate::Uid::from_hex).is_some_and(|u| gone.contains(&u)) {
                continue;
            }
            let mut next = e.clone();
            if let Entry::Panel { state, .. } = &mut next {
                if let Some(o) = state.as_object_mut() {
                    o.insert("node".into(), Value::Null);
                }
            }
            writes.push((id, Some(next)));
        }
        writes
    }

    /// Set a panel's type and/or state. `panel_type` lands FIRST because changing it clears the old
    /// type's state — so a combined `{type, state}` must land the state afterwards, and re-asserting
    /// the SAME type must not wipe, or an agent passing `type` redundantly destroys a live binding.
    ///
    /// `state` MERGES key by key. Every caller reads the bag, edits one key and writes it back, so
    /// two writes in one round trip would have the second replace a bag missing the first's key.
    /// Merging where the write lands kills that class rather than asking each caller to be careful.
    pub fn set_panel(
        &self,
        panel: &str,
        panel_type: Option<&str>,
        state: Option<Value>,
    ) -> Result<Vec<Write>, String> {
        let mut next = self.clone();
        let Some(p) = next.path_of(panel) else {
            return Err(format!("`{panel}` is not in the arrangement"));
        };
        let Node::Panel { panel_type: pt, state: st, .. } = next.at_mut(&p) else {
            return Err(format!("`{panel}` is a split — only a panel carries a type and state"));
        };
        if let Some(t) = panel_type.filter(|t| *t != pt) {
            *pt = t.to_string();
            *st = Value::Null;
        }
        match (state, st.as_object_mut()) {
            (Some(Value::Object(s)), Some(cur)) => cur.extend(s),
            (Some(s), _) => *st = s,
            (None, _) => {}
        }
        Ok(self.diff(&next))
    }

    /// Move the subtree rooted at `root` under `new_parent` at `order_index`. A panel is a subtree of
    /// one, so this covers both the panel case and the tab-onto-panel merge that carries an
    /// arbitrary subtree across tabs — identity, state and every descendant preserved.
    pub fn move_subtree(
        &self,
        root: &str,
        new_parent: &str,
        order_index: usize,
    ) -> Result<Vec<Write>, String> {
        if self.tab_index(root).is_some() {
            return Err("a tab is not a subtree — reorder it with reorder_tab".into());
        }
        if self.path_of(root).is_none() {
            return Err(format!("no such panel `{root}`"));
        }
        match self.node(new_parent) {
            Some(Node::Split { .. }) => {}
            Some(d) => {
                return Err(format!(
                    "`{new_parent}` is a {} — a subtree moves into a split (split a panel first)",
                    d.kind()
                ))
            }
            None => return Err(format!("no such layout entry `{new_parent}`")),
        }
        if self.subtree(root).iter().any(|d| d == new_parent) {
            return Err(format!("`{new_parent}` is inside `{root}` — that would make a cycle"));
        }
        let mut next = self.clone();
        let same_parent = {
            let (tab, at) = next.path_of(root).expect("checked above");
            !at.is_empty() && next.at(&(tab, at[..at.len() - 1].to_vec())).id() == new_parent
        };
        if same_parent {
            // A pure reorder inside one split: detaching would renormalize twice and could promote
            // away the very split being reordered, so only the order changes.
            let p = next.path_of(new_parent).expect("checked above");
            let split = next.at_mut(&p);
            let Node::Split { children, .. } = split else { unreachable!() };
            let from = children.iter().position(|c| c.id() == root).expect("checked above");
            let moved = children.remove(from);
            children.insert(order_index.min(children.len()), moved);
        } else {
            let moved = next.detach(root)?;
            next.attach(moved, new_parent, order_index)?;
        }
        Ok(self.diff(&next))
    }

    /// Remove the subtree rooted at `root`, promoting and renormalizing what is left.
    pub fn remove_subtree(&self, root: &str) -> Result<Vec<Write>, String> {
        if self.tab_index(root).is_some() {
            return Err("a tab is removed with remove_tab".into());
        }
        let mut next = self.clone();
        next.detach(root)?;
        Ok(self.diff(&next))
    }

    // --- the wire -----------------------------------------------------------

    /// The arrangement as plain JSON, one object per entry keyed by id. The `.gfi` section and the
    /// document root share this ONE shape, so the two projections cannot drift.
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        for (id, e) in self.flat() {
            let mut o = serde_json::Map::new();
            o.insert("kind".into(), Value::from(e.kind()));
            match &e {
                Entry::Tab { name, order } => {
                    o.insert("order".into(), Value::from(*order));
                    o.insert("name".into(), Value::from(name.as_str()));
                }
                Entry::Split { parent, order, size, axis } => {
                    o.insert("order".into(), Value::from(*order));
                    o.insert("parent".into(), Value::from(parent.as_str()));
                    o.insert("size".into(), Value::from(*size));
                    o.insert("axis".into(), Value::from(axis.name()));
                }
                Entry::Panel { parent, order, size, panel_type, state } => {
                    o.insert("order".into(), Value::from(*order));
                    o.insert("parent".into(), Value::from(parent.as_str()));
                    o.insert("size".into(), Value::from(*size));
                    o.insert("panel_type".into(), Value::from(panel_type.as_str()));
                    // The state rides as a JSON STRING leaf, and it must stay one: a panel clears a
                    // key with an explicit `null` (`set_panel {state: {node: null}}`), and a null
                    // LEAF in the document is exactly what would make the merge-patch delta
                    // ambiguous — merge patch spends `null` on "delete this key".
                    o.insert("state".into(), Value::from(state.to_string()));
                }
            }
            m.insert(id, Value::Object(o));
        }
        m.insert(SEQ_KEY.into(), Value::from(self.seq));
        Value::Object(m)
    }

    /// Parse a stored arrangement. The shape checks live in [`Self::from_flat`], which is also what
    /// every applied write list goes through. The caller falls back to the default rather than
    /// refusing the patch — the graph is the value, the arrangement is chrome.
    pub fn from_json(v: &Value) -> Result<Layout, String> {
        let obj = v.as_object().ok_or("arrangement: not an object")?;
        let seq = obj.get(SEQ_KEY).and_then(|v| v.as_u64()).unwrap_or(0);
        let mut entries = BTreeMap::new();
        for (id, rec) in obj.iter().filter(|(k, _)| *k != SEQ_KEY) {
            let order = rec
                .get("order")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("arrangement: `{id}` has no order"))? as usize;
            let parent = || {
                rec.get("parent")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .ok_or_else(|| format!("arrangement: `{id}` has no parent"))
            };
            let size = || {
                let s = rec
                    .get("size")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("arrangement: `{id}` has no size"))?;
                if !s.is_finite() || s <= 0.0 || s > 1.0 {
                    return Err(format!("arrangement: `{id}` has size {s}, outside (0, 1]"));
                }
                Ok(s)
            };
            let text = |k: &str| {
                rec.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(str::to_string)
            };
            let e = match rec.get("kind").and_then(|v| v.as_str()).unwrap_or("") {
                "tab" => Entry::Tab {
                    name: text("name").ok_or_else(|| format!("arrangement: tab `{id}` has no name"))?,
                    order,
                },
                "split" => Entry::Split {
                    parent: parent()?,
                    order,
                    size: size()?,
                    axis: rec
                        .get("axis")
                        .and_then(|v| v.as_str())
                        .and_then(Axis::parse)
                        .ok_or_else(|| format!("arrangement: split `{id}` has no row/column axis"))?,
                },
                "panel" => Entry::Panel {
                    parent: parent()?,
                    order,
                    size: size()?,
                    panel_type: text("panel_type")
                        .ok_or_else(|| format!("arrangement: panel `{id}` has no type"))?,
                    // The state rides as a JSON STRING leaf (see `to_json`). A `state` written any
                    // other way — the natural shape of a hand edit — is REFUSED rather than read as
                    // absent, which would load the panel with its binding silently wiped.
                    state: match rec.get("state") {
                        None | Some(Value::Null) => Value::Null,
                        Some(Value::String(s)) => serde_json::from_str(s)
                            .map_err(|e| format!("arrangement: panel `{id}` has malformed state: {e}"))?,
                        Some(_) => {
                            return Err(format!("arrangement: panel `{id}`'s state is not a JSON string"))
                        }
                    },
                },
                other => return Err(format!("arrangement: `{id}` has unknown kind `{other}`")),
            };
            entries.insert(id.clone(), e);
        }
        Layout::from_flat(&entries, seq)
    }
}
