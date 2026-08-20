//! The editor's panel arrangement, held FLAT and id-keyed — every tab, split and panel is one
//! entry naming its parent and its order among that parent's children. The tree is reconstructed
//! from those pointers at render time; `parent` always names a stable id, never a name.
//!
//! Flat is what lets the arrangement ride the SAME machinery as the graph. The CRDT reconciler
//! mirrors nested maps and scalars but ERASES nested arrays (which is why `links` has its own flat
//! path), so an id-keyed map is exactly the `nodes` shape it already handles — no new CRDT
//! machinery. Flattening also turns move/reorder/reparent into field edits with panel identity
//! preserved, instead of identity-losing tree surgery.
//!
//! Every mutation is a PLANNER: it mutates a clone and returns the per-entry writes that turn this
//! arrangement into that one. That is what makes a layout op an ordinary command compound with an
//! exact inverse, and it keeps the tricky cases — a promote whose survivor is also the move's
//! destination — out of hand-rolled delta bookkeeping.

use serde_json::Value;
use std::collections::BTreeMap;

/// A stable tab/split/panel id. Minted here, never by a client.
pub type Id = String;

/// One entry's new value, or `None` to remove it — the unit a layout command applies and inverts.
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

/// One node of the arrangement. A `Tab` is a root (no parent, exactly one child); a `Split` divides
/// its slot between its children along one axis; a `Panel` is a leaf hosting one registered type.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Tab {
        /// The unique human alias every session/tab op addresses. Renaming it preserves the id, so
        /// no descendant's `parent` moves.
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
        /// The panel type's own opaque state (a viewer's `{node, slot}`). Never interpreted here.
        state: Value,
    },
}

impl Entry {
    pub fn parent(&self) -> Option<&str> {
        match self {
            Entry::Tab { .. } => None,
            Entry::Split { parent, .. } | Entry::Panel { parent, .. } => Some(parent),
        }
    }
    fn order(&self) -> usize {
        match self {
            Entry::Tab { order, .. } | Entry::Split { order, .. } | Entry::Panel { order, .. } => *order,
        }
    }
    fn size(&self) -> f64 {
        match self {
            Entry::Tab { .. } => 1.0,
            Entry::Split { size, .. } | Entry::Panel { size, .. } => *size,
        }
    }
    fn set_order(&mut self, v: usize) {
        match self {
            Entry::Tab { order, .. } | Entry::Split { order, .. } | Entry::Panel { order, .. } => *order = v,
        }
    }
    fn set_size(&mut self, v: f64) {
        match self {
            Entry::Tab { .. } => {}
            Entry::Split { size, .. } | Entry::Panel { size, .. } => *size = v,
        }
    }
    fn set_parent(&mut self, v: &str) {
        match self {
            Entry::Tab { .. } => {}
            Entry::Split { parent, .. } | Entry::Panel { parent, .. } => *parent = v.to_string(),
        }
    }
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
    /// Its old siblings, nearest first. The first one still standing is what it lands beside, which
    /// puts the old pairing back whether or not the old split survived.
    siblings: Vec<Id>,
    /// What every entry's slot was worth before the move, by id. `siblings` decides WHERE the subtree
    /// lands; this is what gives the arrangement its geometry back. A move renormalizes the split it
    /// leaves AND the split it enters — and a promote pushes that another level up — so the set of
    /// slices it disturbs is not knowable from one level of siblings.
    shares: BTreeMap<Id, f64>,
    /// The old parent's id and axis. The id is handed back only to a wrapper that has to be minted
    /// anyway, and only while it is free — an absent split is referenced by nothing, so reusing its
    /// id strands nobody, where restoring its SLOT would.
    parent: Id,
    axis: Axis,
    /// It sat before its nearest sibling, and held this share of its parent.
    before: bool,
    size: f64,
    /// Its tab's name and tab index — the last resort, for the frozen drag where the tab went with
    /// its last panel and there is neither a sibling nor a tab left to land on.
    tab: (String, usize),
}

impl Home {
    /// The share `id` held before the move, if this home remembers it.
    fn share(&self, id: &str) -> Option<f64> {
        self.shares.get(id).copied()
    }
}

/// The whole arrangement. Keyed by id, so a duplicate id is structurally impossible rather than a
/// validation class — the flat model's one free correctness win.
#[derive(Clone, Debug)]
pub struct Layout {
    entries: BTreeMap<Id, Entry>,
    /// The id counter, monotone: [`Self::insert`] raises it past every id it admits and nothing ever
    /// lowers it, so a closed panel's id is never handed out again — `model.ts`'s `_seq`, which
    /// likewise only counts up. It rides [`Self::to_json`] so a reopened patch keeps counting
    /// forward. Recycling would silently give a fresh panel a dead one's client-side state, the
    /// viewpoint's `subpatchPath` among it.
    seq: u64,
}

/// Two arrangements are the same when they DRAW the same. The id counter is bookkeeping that only
/// counts up — an undo deliberately does NOT wind it back, which is the whole point of it.
impl PartialEq for Layout {
    fn eq(&self, other: &Layout) -> bool {
        self.entries == other.entries
    }
}

impl Default for Layout {
    /// The arrangement a fresh patch opens with, matching `defaultWorkspaceState()`: one tab
    /// holding one node-editor panel. Also the fallback a corrupt stored arrangement lands on.
    fn default() -> Layout {
        let mut l = Layout { entries: BTreeMap::new(), seq: 0 };
        let tab = l.mint("tab");
        l.insert(tab.clone(), Entry::Tab { name: DEFAULT_TAB_NAME.into(), order: 0 });
        let panel = l.mint("panel");
        l.insert(
            panel,
            Entry::Panel {
                parent: tab,
                order: 0,
                size: 1.0,
                panel_type: DEFAULT_PANEL_TYPE.into(),
                state: Value::Null,
            },
        );
        l
    }
}

impl Layout {
    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.entries.get(id)
    }
    /// A fresh id — one counter across all three kinds, like `model.ts`'s `_seq`. Only [`Self::insert`]
    /// advances it, so minting twice before inserting either would collide; every planner inserts
    /// through it.
    fn mint(&self, prefix: &str) -> Id {
        format!("{prefix}-{}", self.seq + 1)
    }

    /// This parent's children, in order. Ties break by id so document order is total even mid-edit.
    pub fn children(&self, parent: &str) -> Vec<Id> {
        let mut c: Vec<(usize, &Id)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.parent() == Some(parent))
            .map(|(id, e)| (e.order(), id))
            .collect();
        c.sort();
        c.into_iter().map(|(_, id)| id.clone()).collect()
    }

    /// Every tab, in order.
    pub fn tabs(&self) -> Vec<Id> {
        let mut p: Vec<(usize, &Id)> = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(e, Entry::Tab { .. }))
            .map(|(id, e)| (e.order(), id))
            .collect();
        p.sort();
        p.into_iter().map(|(_, id)| id.clone()).collect()
    }

    pub fn tab_named(&self, name: &str) -> Option<Id> {
        self.entries
            .iter()
            .find(|(_, e)| matches!(e, Entry::Tab { name: n, .. } if n == name))
            .map(|(id, _)| id.clone())
    }

    pub fn name_of(&self, tab: &str) -> Option<&str> {
        match self.entries.get(tab) {
            Some(Entry::Tab { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// The tab an entry belongs to (itself, if it is one). `None` on a dangling parent or a cycle —
    /// the walk is bounded by the entry count, which is what makes [`Self::validate`] catch both in
    /// one step.
    pub fn tab_of(&self, id: &str) -> Option<Id> {
        let mut cur = id;
        for _ in 0..=self.entries.len() {
            match self.entries.get(cur)? {
                Entry::Tab { .. } => return Some(cur.to_string()),
                e => cur = e.parent()?,
            }
        }
        None
    }

    /// `root` and every descendant, in document order (depth-first, parents before children).
    fn subtree(&self, root: &str) -> Vec<Id> {
        let mut out = Vec::new();
        self.walk(root, &mut out);
        out
    }

    fn walk(&self, id: &str, out: &mut Vec<Id>) {
        // The length guard is the cycle stop: a valid arrangement visits each entry at most once.
        if !self.entries.contains_key(id) || out.len() > self.entries.len() {
            return;
        }
        out.push(id.to_string());
        for c in self.children(id) {
            self.walk(&c, out);
        }
    }

    /// Upsert one entry, handing back what it displaced — the primitive a layout command inverts
    /// (`None` back means the inverse of this write is a removal). Also the ONE place the id counter
    /// advances: every id this arrangement has ever admitted stays spent, whether or not it is still
    /// here.
    pub fn insert(&mut self, id: Id, entry: Entry) -> Option<Entry> {
        if let Some(n) = id.rsplit_once('-').and_then(|(_, n)| n.parse::<u64>().ok()) {
            self.seq = self.seq.max(n);
        }
        self.entries.insert(id, entry)
    }

    /// Remove one entry, handing it back for the same reason.
    pub fn remove(&mut self, id: &str) -> Option<Entry> {
        self.entries.remove(id)
    }

    /// Apply a planner's writes wholesale — what `LayoutBirth`/`LayoutClose` land, since those two
    /// carry ONE re-planned inverse between them rather than one per slot. The commands that do
    /// invert slot by slot apply the writes one at a time instead.
    pub fn apply(&mut self, writes: Vec<Write>) {
        for (id, entry) in writes {
            match entry {
                Some(e) => self.insert(id, e),
                None => self.remove(&id),
            };
        }
    }

    /// The per-entry writes that turn `self` into `next` — every planner's return value, and thus
    /// the shape a layout command inverts entry by entry.
    fn diff(&self, next: &Layout) -> Vec<Write> {
        let mut w: Vec<Write> = next
            .entries
            .iter()
            .filter(|(id, e)| self.entries.get(*id) != Some(e))
            .map(|(id, e)| (id.clone(), Some(e.clone())))
            .collect();
        w.extend(
            self.entries.keys().filter(|id| !next.entries.contains_key(*id)).map(|id| (id.clone(), None)),
        );
        w
    }

    fn order_children(&mut self, ids: &[Id]) {
        for (i, id) in ids.iter().enumerate() {
            if let Some(e) = self.entries.get_mut(id) {
                e.set_order(i);
            }
        }
    }

    /// Scale a parent's children so their sizes sum to 1.
    fn normalize(&mut self, parent: &str) {
        let kids = self.children(parent);
        let total: f64 = kids.iter().filter_map(|k| self.entries.get(k)).map(Entry::size).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        for k in kids {
            if let Some(e) = self.entries.get_mut(&k) {
                let s = e.size() / total;
                e.set_size(s);
            }
        }
    }

    /// Lift `id` out of its parent's child list and hand it back — the shared half of a close and a
    /// move (`closePanel`). The freed slice goes to the siblings in proportion, and a split left
    /// with ONE child is replaced by that child in its own slot, so the tree never keeps a one-armed
    /// wrapper. The subtree hanging off `id` is untouched (a move re-attaches it whole).
    fn detach(&mut self, id: &str) -> Result<Entry, String> {
        let e = self.entries.get(id).cloned().ok_or_else(|| format!("no such panel `{id}`"))?;
        let parent = e.parent().ok_or("a tab is not inside anything")?.to_string();
        if matches!(self.entries.get(&parent), Some(Entry::Tab { .. })) {
            return Err(format!(
                "`{id}` is tab `{}`'s only root — a tab always keeps one",
                self.name_of(&parent).unwrap_or(&parent)
            ));
        }
        self.entries.remove(id);
        let sibs = self.children(&parent);
        let total: f64 = sibs.iter().filter_map(|s| self.entries.get(s)).map(Entry::size).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        for s in &sibs {
            if let Some(x) = self.entries.get_mut(s) {
                let v = x.size();
                x.set_size(v + e.size() * v / total);
            }
        }
        self.normalize(&parent);
        if sibs.len() == 1 {
            let (order, size, grand) = match self.entries.get(&parent) {
                Some(p) => (p.order(), p.size(), p.parent().map(str::to_string)),
                None => return Ok(e),
            };
            if let (Some(g), Some(c)) = (grand, self.entries.get_mut(&sibs[0])) {
                c.set_parent(&g);
                c.set_order(order);
                c.set_size(size);
            }
            self.entries.remove(&parent);
        } else {
            self.order_children(&sibs);
        }
        Ok(e)
    }

    /// Put `entry` back under `parent` at `index`. The newcomer takes an EQUAL share (the average of
    /// what is there) and the siblings keep their relative proportions.
    fn attach(&mut self, id: &str, mut entry: Entry, parent: &str, index: usize) {
        let mut kids = self.children(parent);
        entry.set_parent(parent);
        entry.set_size(if kids.is_empty() { 1.0 } else { 1.0 / kids.len() as f64 });
        self.insert(id.to_string(), entry);
        kids.insert(index.min(kids.len()), id.to_string());
        self.order_children(&kids);
        self.normalize(parent);
    }

    /// Add a tab and return its id. It holds one fresh node-editor panel — unless `subtree` names
    /// an existing one, in which case the tab is built AROUND it: the frozen drop-onto-the-tab-bar
    /// gesture, which `add_tab` + `move_panel` cannot express (a move needs a split to
    /// land in, and a fresh tab has none). `index` places it in the tab strip.
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
            Some(s) => Some((s.to_string(), next.take(s)?)),
            None => None,
        };
        let order = next.tabs().len();
        let tab = next.mint("tab");
        next.insert(tab.clone(), Entry::Tab { name: name.to_string(), order });
        let (root, mut entry) = adopted.unwrap_or_else(|| {
            (
                next.mint("panel"),
                Entry::Panel {
                    parent: String::new(),
                    order: 0,
                    size: 1.0,
                    panel_type: DEFAULT_PANEL_TYPE.into(),
                    state: Value::Null,
                },
            )
        });
        entry.set_parent(&tab);
        entry.set_order(0);
        entry.set_size(1.0);
        next.insert(root, entry);
        if let Some(i) = index {
            let mut order = next.tabs();
            order.retain(|p| *p != tab);
            order.insert(i.min(order.len()), tab.clone());
            next.order_children(&order);
        }
        Ok((self.diff(&next), tab))
    }

    /// Lift a subtree out for re-homing. Normally a [`Self::detach`] — but when it is its tab's ONLY
    /// root the TAB goes with it, which is the frozen "the panel was the tab's only node → the tab
    /// goes with it" branch of `_takeNode`. The last tab never goes.
    fn take(&mut self, root: &str) -> Result<Entry, String> {
        let parent = match self.entries.get(root) {
            Some(e) => e.parent().map(str::to_string),
            None => return Err(format!("no such panel `{root}`")),
        };
        let Some(parent) = parent else {
            return Err("a tab is not a subtree — reorder it with reorder_tab".into());
        };
        if !matches!(self.entries.get(&parent), Some(Entry::Tab { .. })) {
            return self.detach(root);
        }
        if self.tabs().len() <= 1 {
            return Err(format!("`{root}` is the only panel on the only tab — it has nowhere to go"));
        }
        let e = self.entries.remove(root).expect("looked up above");
        self.entries.remove(&parent);
        let rest = self.tabs();
        self.order_children(&rest);
        Ok(e)
    }

    /// Put `entry` beside `target` along `axis` — the ONE place split-or-wrap lives. A parent
    /// already running along `axis` gains a sibling; otherwise the target is wrapped in a fresh
    /// split inheriting its slot. `entry`'s `size` is READ as the share it asks for, so a caller
    /// hands over a lifted subtree or a new panel indifferently. `wrap` names the id a minted
    /// wrapper takes if still free — how an undo gives a promoted-away split its id back without
    /// restoring its slot.
    fn insert_at(
        &mut self,
        id: &str,
        mut entry: Entry,
        target: &str,
        axis: Axis,
        before: bool,
        wrap: Option<&str>,
    ) {
        let f = entry.size();
        let Some((parent, slot, order)) = self
            .entries
            .get(target)
            .map(|t| (t.parent().unwrap_or_default().to_string(), t.size(), t.order()))
        else {
            return;
        };
        let same = matches!(self.entries.get(&parent), Some(Entry::Split { axis: a, .. }) if *a == axis);
        entry.set_parent(&parent);
        entry.set_order(order);
        entry.set_size(slot * f);
        self.insert(id.to_string(), entry);
        if same {
            if let Some(t) = self.entries.get_mut(target) {
                t.set_size(slot - slot * f);
            }
            let mut kids = self.children(&parent);
            kids.retain(|k| k != id);
            let at =
                kids.iter().position(|k| k == target).map_or(0, |i| if before { i } else { i + 1 });
            kids.insert(at, id.to_string());
            self.order_children(&kids);
        } else {
            let wrap = wrap
                .filter(|w| !self.entries.contains_key(*w))
                .map(str::to_string)
                .unwrap_or_else(|| self.mint("split"));
            self.insert(wrap.clone(), Entry::Split { parent, order, size: slot, axis });
            // The newcomer always takes `f` and the target the rest; only their ORDER flips.
            for (child, o, size) in [(target, before as usize, 1.0 - f), (id, 1 - before as usize, f)]
            {
                if let Some(e) = self.entries.get_mut(child) {
                    e.set_parent(&wrap);
                    e.set_order(o);
                    e.set_size(size);
                }
            }
        }
    }

    /// Re-home the subtree rooted at `subtree` beside `target`, splitting along `axis` — `dropOnPanel`
    /// as ONE plan. Three ops would cost the user three ctrl-Z for one drag and show every peer two
    /// arrangements that were never on screen.
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
        match self.entries.get(target) {
            Some(Entry::Panel { .. }) => {}
            Some(e) => return Err(format!("`{target}` is a {} — a drop lands on a panel", e.kind())),
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
        next.insert_at(subtree, moved, target, axis, before, None);
        Ok(self.diff(&next))
    }

    /// Where `root` sits now, in the terms [`Self::re_home`] needs to put it back. `None` for a tab,
    /// which is reordered rather than moved.
    pub fn home_of(&self, root: &str) -> Option<Home> {
        let e = self.entries.get(root)?;
        let parent = e.parent()?.to_string();
        let tab = self.tab_of(root)?;
        let kids = self.children(&parent);
        let i = kids.iter().position(|k| k == root)?;
        // Nearest first: the neighbour that shared an edge with `root` is the one whose survival
        // reconstructs the old pairing exactly.
        let mut sibs: Vec<(usize, Id)> =
            kids.into_iter().enumerate().filter(|(_, k)| k != root).collect();
        sibs.sort_by_key(|(j, _)| j.abs_diff(i));
        Some(Home {
            siblings: sibs.into_iter().map(|(_, k)| k).collect(),
            shares: self.entries.iter().map(|(id, e)| (id.clone(), e.size())).collect(),
            axis: match self.entries.get(&parent) {
                Some(Entry::Split { axis, .. }) => *axis,
                _ => Axis::Row,
            },
            parent,
            before: i == 0,
            size: e.size(),
            tab: (self.name_of(&tab)?.to_string(), self.tabs().iter().position(|p| *p == tab)?),
        })
    }

    /// Plan a move of `root` back to `home`, against the arrangement AS IT STANDS — the inverse of
    /// every layout op that moves something. It lands beside the first old sibling still standing,
    /// else beside its old tab's current root, else inside a tab re-born around it. What it never
    /// does is restore its old parent's slot: the move may have promoted that split away, and a peer
    /// may have built on whatever took its place, which a restore would strand.
    pub fn re_home(&self, root: &str, home: &Home) -> Result<Vec<Write>, String> {
        let mut next = self.clone();
        // Lifted FIRST, so the landing is chosen among what survives closing up behind it.
        let mut e = next.take(root)?;
        let inside = self.subtree(root);
        let landing = home
            .siblings
            .iter()
            .find(|s| next.entries.contains_key(*s) && !inside.contains(s))
            .cloned()
            .or_else(|| {
                next.tab_named(&home.tab.0).and_then(|p| next.children(&p).into_iter().next())
            });
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
        e.set_size(home.size);
        next.insert_at(root, e, &landing, home.axis, home.before, Some(&home.parent));
        next.give_back_shares(self, home);
        Ok(self.diff(&next))
    }

    /// Re-assert the shares `home` remembers wherever this plan disturbed them — what makes an
    /// undisturbed undo exact to the pixel. SHARES only: where an entry sits is still re-planned
    /// and never restored. A split the plan left alone keeps what it holds, so a peer's resize
    /// elsewhere survives.
    fn give_back_shares(&mut self, before: &Layout, home: &Home) {
        let disturbed: std::collections::BTreeSet<Id> = self
            .entries
            .values()
            .filter_map(Entry::parent)
            .filter(|p| self.children(p) != before.children(p))
            .map(str::to_string)
            .collect();
        for p in disturbed {
            for k in self.children(&p) {
                if let (Some(s), Some(x)) = (home.share(&k), self.entries.get_mut(&k)) {
                    x.set_size(s);
                }
            }
            self.normalize(&p);
        }
    }

    /// Every entry of the subtree rooted at `root` — what a close carries into its own inverse. The
    /// ids are dead the moment the close lands and nothing ever mints one again, so putting them back
    /// strands nobody; WHERE the root lands is [`Self::revive`]'s question, not theirs.
    pub fn dead_subtree(&self, root: &str) -> Vec<(Id, Entry)> {
        self.subtree(root).into_iter().filter_map(|id| Some((id.clone(), self.get(&id)?.clone()))).collect()
    }

    /// Plan the inverse of a close: put `dead` back, then RE-PLAN where its root belongs —
    /// [`Self::re_home`] for a subtree, the tab strip for a tab. What it never does is pin the root
    /// into the slot it held: the close promoted that split away, a peer may have built where it
    /// stood, and a later undo may even have handed its id to a live wrapper.
    pub fn revive(&self, dead: &[(Id, Entry)], root: &str, home: Option<&Home>) -> Result<Vec<Write>, String> {
        let mut back = self.clone();
        for (id, e) in dead {
            let mut e = e.clone();
            // The root's old parent is deliberately NOT restored, for the same reason its slot is not.
            if id == root {
                e.set_parent("");
            }
            back.insert(id.clone(), e);
        }
        let writes = match (back.get(root).cloned(), home) {
            // A tab hangs off nothing, so only its place in the tab strip needs re-planning — a
            // peer's new tab has taken an index since, and restoring the old one collides with it.
            (Some(Entry::Tab { order, .. }), _) => back.reorder_tab(root, order)?,
            (Some(_), Some(h)) => back.re_home(root, h)?,
            _ => return Err(format!("`{root}` is not something a close can give back")),
        };
        back.apply(writes);
        Ok(self.diff(&back))
    }

    /// Land `writes` as CONTENTS edits: what each entry HOLDS arrives, but WHERE it sits is read
    /// off the arrangement as it stands and never off the write. That is what makes the inverse of
    /// a type change safe — the slot an entry held at plan time may be a peer's by undo time. An id
    /// that has since gone is skipped, so a stale replay degrades instead of resurrecting it.
    pub fn set_contents(&self, writes: &[Write]) -> Vec<Write> {
        let mut next = self.clone();
        let mut parents = std::collections::BTreeSet::new();
        for (id, entry) in writes {
            let (Some(mut e), Some(live)) = (entry.clone(), next.get(id).cloned()) else {
                continue;
            };
            if let Some(p) = live.parent() {
                e.set_parent(p);
                parents.insert(p.to_string());
            }
            e.set_order(live.order());
            next.insert(id.clone(), e);
        }
        for p in parents {
            next.normalize(&p);
        }
        self.diff(&next)
    }

    /// Set every child of `split` at once — what a resize drag commits on pointer-up, and the only
    /// op that sizes anything. Scaling ONE child and renormalizing its siblings would make N of them
    /// chase a moving target and never land on the fraction set the user drew.
    pub fn resize_split(
        &self,
        split: &str,
        fractions: &[f64],
    ) -> Result<Vec<Write>, String> {
        if !matches!(self.entries.get(split), Some(Entry::Split { .. })) {
            let kind = self.entries.get(split).map_or("entry", Entry::kind);
            return Err(format!("`{split}` is a {kind} — only a split divides its slot"));
        }
        let kids = self.children(split);
        if kids.len() != fractions.len() {
            return Err(format!(
                "`{split}` has {} children, so it needs {} fractions, not {}",
                kids.len(),
                kids.len(),
                fractions.len()
            ));
        }
        if fractions.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err("every fraction must be a positive number".into());
        }
        let mut next = self.clone();
        for (k, f) in kids.iter().zip(fractions) {
            if let Some(e) = next.entries.get_mut(k) {
                e.set_size(f.max(MIN_FRACTION));
            }
        }
        next.normalize(split);
        Ok(self.diff(&next))
    }

    /// Refuse an id that is not a tab. Every tab op addresses BY ID — a name is what the tab holds,
    /// not how it is found, so renaming one cannot make a caller's next op miss.
    fn is_tab(&self, tab: &str) -> Result<(), String> {
        match self.entries.get(tab) {
            Some(Entry::Tab { .. }) => Ok(()),
            Some(e) => Err(format!("`{tab}` is a {} — not a tab", e.kind())),
            None => Err(format!("no such tab `{tab}`")),
        }
    }

    pub fn remove_tab(&self, tab: &str) -> Result<Vec<Write>, String> {
        self.is_tab(tab)?;
        if self.tabs().len() <= 1 {
            return Err("the last tab cannot be removed".into());
        }
        let mut next = self.clone();
        for id in self.subtree(tab) {
            next.entries.remove(&id);
        }
        let rest = next.tabs();
        next.order_children(&rest);
        Ok(self.diff(&next))
    }

    /// Rename a tab. A field edit, so the id — and therefore every descendant's `parent` — stands.
    pub fn rename_tab(&self, tab: &str, to: &str) -> Result<Vec<Write>, String> {
        self.is_tab(tab)?;
        let to = to.trim();
        if to.is_empty() {
            return Err("a tab needs a name".into());
        }
        if self.tab_named(to).is_some_and(|other| other != tab) {
            return Err(format!("a tab named `{to}` already exists"));
        }
        let mut next = self.clone();
        if let Some(Entry::Tab { name, .. }) = next.entries.get_mut(tab) {
            *name = to.to_string();
        }
        Ok(self.diff(&next))
    }

    pub fn reorder_tab(&self, tab: &str, to_index: usize) -> Result<Vec<Write>, String> {
        self.is_tab(tab)?;
        let mut order = self.tabs();
        order.retain(|p| p != tab);
        order.insert(to_index.min(order.len()), tab.to_string());
        let mut next = self.clone();
        next.order_children(&order);
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
        match self.entries.get(panel) {
            Some(Entry::Panel { .. }) => {}
            Some(e) => return Err(format!("`{panel}` is a {} — only a panel splits", e.kind())),
            None => return Err(format!("no such panel `{panel}`")),
        }
        let f = fraction(ratio)?;
        let mut next = self.clone();
        let fresh = next.mint("panel");
        // parent/order are `insert_at`'s to set and `size` is the share it asks for; only the type
        // and state are this op's.
        let born = Entry::Panel {
            parent: String::new(),
            order: 0,
            size: f,
            panel_type: EMPTY_PANEL_TYPE.into(),
            state: Value::Null,
        };
        next.insert_at(&fresh, born, panel, axis, place_before, None);
        Ok((self.diff(&next), fresh))
    }

    /// Clear the node binding of every panel naming a uid in `gone`, as the writes a
    /// [`crate::Command::LayoutContents`] lands. A panel's `state` is opaque here save for this one
    /// key, which the frontend and the bind validation already share (`set_panel`'s
    /// `state.node`) — a panel pointing at a deleted node is the one arrangement the manager can
    /// know is wrong.
    pub fn unbind(&self, gone: &std::collections::HashSet<crate::Uid>) -> Vec<Write> {
        let mut writes = Vec::new();
        for (id, e) in &self.entries {
            let Entry::Panel { state, .. } = e else { continue };
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
            writes.push((id.clone(), Some(next)));
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
        let Some(Entry::Panel { panel_type: pt, state: st, .. }) = next.entries.get_mut(panel) else {
            let kind = self.entries.get(panel).map_or("entry", Entry::kind);
            return Err(format!("`{panel}` is a {kind} — only a panel carries a type and state"));
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

    /// Move the subtree rooted at `root` under `new_parent` at `order_index`. A panel is a subtree
    /// of one, so this covers both the panel case and the tab-onto-panel merge that carries an
    /// arbitrary subtree across tabs — identity, state and every descendant preserved.
    pub fn move_subtree(
        &self,
        root: &str,
        new_parent: &str,
        order_index: usize,
    ) -> Result<Vec<Write>, String> {
        let e = &self.entries[root];
        if matches!(e, Entry::Tab { .. }) {
            return Err("a tab is not a subtree — reorder it with reorder_tab".into());
        }
        match self.entries.get(new_parent) {
            Some(Entry::Split { .. }) => {}
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
        if e.parent() == Some(new_parent) {
            // A pure reorder inside one split: detaching would renormalize twice and could promote
            // away the very split being reordered, so only the order changes.
            let mut kids = next.children(new_parent);
            kids.retain(|k| k != root);
            kids.insert(order_index.min(kids.len()), root.to_string());
            next.order_children(&kids);
        } else {
            let moved = next.detach(root)?;
            next.attach(root, moved, new_parent, order_index);
        }
        Ok(self.diff(&next))
    }

    /// Remove the subtree rooted at `root`, promoting and renormalizing what is left.
    pub fn remove_subtree(&self, root: &str) -> Result<Vec<Write>, String> {
        if matches!(self.entries.get(root), Some(Entry::Tab { .. })) {
            return Err("a tab is removed with remove_tab".into());
        }
        let doomed = self.subtree(root);
        let mut next = self.clone();
        next.detach(root)?;
        for d in doomed {
            next.entries.remove(&d);
        }
        Ok(self.diff(&next))
    }

    /// The arrangement as plain JSON, one object per entry keyed by id. The `.gfi` section and the
    /// CRDT root share this ONE shape, so the two projections cannot drift.
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        for (id, e) in &self.entries {
            let mut o = serde_json::Map::new();
            o.insert("kind".into(), Value::from(e.kind()));
            o.insert("order".into(), Value::from(e.order()));
            match e {
                Entry::Tab { name, .. } => {
                    o.insert("name".into(), Value::from(name.as_str()));
                }
                Entry::Split { parent, size, axis, .. } => {
                    o.insert("parent".into(), Value::from(parent.as_str()));
                    o.insert("size".into(), Value::from(*size));
                    o.insert("axis".into(), Value::from(axis.name()));
                }
                Entry::Panel { parent, size, panel_type, state, .. } => {
                    o.insert("parent".into(), Value::from(parent.as_str()));
                    o.insert("size".into(), Value::from(*size));
                    o.insert("panel_type".into(), Value::from(panel_type.as_str()));
                    // The state rides as a JSON STRING leaf, like a node's `viewers`: the CRDT
                    // reconciler mirrors nested maps but ERASES nested arrays, and a viewer's
                    // settings can hold one.
                    o.insert("state".into(), Value::from(state.to_string()));
                }
            }
            m.insert(id.clone(), Value::Object(o));
        }
        m.insert(SEQ_KEY.into(), Value::from(self.seq));
        Value::Object(m)
    }

    /// Parse a stored arrangement, refusing every failure class flattening admits. The caller falls
    /// back to the default rather than refusing the patch — the graph is the value, the arrangement
    /// is chrome.
    pub fn from_json(v: &Value) -> Result<Layout, String> {
        let obj = v.as_object().ok_or("arrangement: not an object")?;
        let mut l = Layout {
            entries: BTreeMap::new(),
            seq: obj.get(SEQ_KEY).and_then(|v| v.as_u64()).unwrap_or(0),
        };
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
            l.insert(id.clone(), e);
        }
        l.validate()?;
        Ok(l)
    }

    /// Every invariant the flat model can violate but the nested tree could not. A duplicate ID is
    /// absent from the list because the id-keyed map makes it impossible to express.
    fn validate(&self) -> Result<(), String> {
        let tabs = self.tabs();
        if tabs.is_empty() {
            return Err("arrangement: no tabs".into());
        }
        let mut names = std::collections::HashSet::new();
        let mut indices = std::collections::HashSet::new();
        for p in &tabs {
            let n = self.name_of(p).unwrap_or_default();
            if !names.insert(n) {
                return Err(format!("arrangement: two tabs are both named `{n}`"));
            }
            // The tab strip is the one child list no parent owns, so the per-entry order check
            // below never reaches it — and a tab restored into the slot it held collides here.
            let i = self.entries[p].order();
            if !indices.insert(i) {
                return Err(format!("arrangement: two tabs share strip index {i}"));
            }
        }
        // The per-entry checks run FIRST, because an entry that reaches no tab is the CAUSE of the
        // tab-root count being wrong, and a caller shown only the symptom cannot find the entry.
        for (id, e) in &self.entries {
            // Walking up to a tab refuses a dangling parent AND a cycle in the same step.
            if self.tab_of(id).is_none() {
                return Err(format!(
                    "arrangement: `{id}` reaches no tab — a missing parent, or a cycle"
                ));
            }
            if let Some(p) = e.parent() {
                if matches!(self.entries.get(p), Some(Entry::Panel { .. })) {
                    return Err(format!("arrangement: `{id}` hangs off panel `{p}`, which is a leaf"));
                }
            }
            let mut seen = std::collections::HashSet::new();
            for k in self.children(id) {
                if !seen.insert(self.entries[&k].order()) {
                    return Err(format!(
                        "arrangement: two children of `{id}` share order {}",
                        self.entries[&k].order()
                    ));
                }
            }
        }
        for p in &tabs {
            // A tab holds exactly one root — the nested tree's `Workspace.root`. Zero or many is a
            // shape flattening admits and rendering cannot.
            let roots = self.children(p).len();
            if roots != 1 {
                let n = self.name_of(p).unwrap_or_default();
                return Err(format!("arrangement: tab `{n}` has {roots} roots, not 1"));
            }
        }
        Ok(())
    }
}
