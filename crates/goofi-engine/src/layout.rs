//! The editor's panel arrangement, held FLAT and id-keyed — every page, split and panel is one
//! entry naming its parent and its order among that parent's children. The tree is reconstructed
//! from those pointers at render time; `parent` always names a stable id, never a name.
//!
//! Flat is what lets the arrangement ride the SAME machinery as the graph. The CRDT reconciler
//! mirrors nested maps and scalars but ERASES nested arrays (which is why `links` has its own flat
//! path), so an id-keyed map is exactly the `nodes` shape it already handles — no new CRDT
//! machinery. Flattening also turns move/reorder/reparent into field edits with panel identity
//! preserved, instead of identity-losing tree surgery.
//!
//! The tree algebra that survives flattening — split-or-wrap, close-with-promote and size
//! renormalization — is ported from `frontend/src/lib/workspace/model.ts`, which is its source of
//! truth and carries its own tests. Matching it is deliberate, not incidental.
//!
//! Every mutation is a PLANNER: it mutates a clone and returns the per-entry writes that turn this
//! arrangement into that one. That is what makes a layout op an ordinary `Command` compound with an
//! exact per-entry inverse, and it keeps the genuinely tricky cases (a promote whose survivor is
//! also the move's destination) out of hand-rolled delta bookkeeping.

use serde_json::Value;
use std::collections::BTreeMap;

/// A stable page/split/panel id. Minted here, never by a client.
pub type Id = String;

/// One entry's new value, or `None` to remove it — the unit a layout command applies and inverts.
pub type Write = (Id, Option<Entry>);

/// The panel type a fresh page starts with, and the placeholder a split births (which is what keeps
/// a split from assuming content). Both mirror `model.ts`.
pub const DEFAULT_PANEL_TYPE: &str = "node-editor";
pub const EMPTY_PANEL_TYPE: &str = "empty";
const DEFAULT_PAGE_NAME: &str = "Layout";

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

/// One node of the arrangement. A `Page` is a root (no parent, exactly one child); a `Split` divides
/// its slot between its children along one axis; a `Panel` is a leaf hosting one registered type.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Page {
        /// The unique human alias every session/page op addresses. Renaming it preserves the id, so
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
            Entry::Page { .. } => None,
            Entry::Split { parent, .. } | Entry::Panel { parent, .. } => Some(parent),
        }
    }
    fn order(&self) -> usize {
        match self {
            Entry::Page { order, .. } | Entry::Split { order, .. } | Entry::Panel { order, .. } => *order,
        }
    }
    fn size(&self) -> f64 {
        match self {
            Entry::Page { .. } => 1.0,
            Entry::Split { size, .. } | Entry::Panel { size, .. } => *size,
        }
    }
    fn set_order(&mut self, v: usize) {
        match self {
            Entry::Page { order, .. } | Entry::Split { order, .. } | Entry::Panel { order, .. } => *order = v,
        }
    }
    fn set_size(&mut self, v: f64) {
        match self {
            Entry::Page { .. } => {}
            Entry::Split { size, .. } | Entry::Panel { size, .. } => *size = v,
        }
    }
    fn set_parent(&mut self, v: &str) {
        match self {
            Entry::Page { .. } => {}
            Entry::Split { parent, .. } | Entry::Panel { parent, .. } => *parent = v.to_string(),
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            Entry::Page { .. } => "page",
            Entry::Split { .. } => "split",
            Entry::Panel { .. } => "panel",
        }
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
    /// The arrangement a fresh patch opens with, matching `defaultWorkspaceState()`: one page
    /// holding one node-editor panel. Also the fallback a corrupt stored arrangement lands on.
    fn default() -> Layout {
        let mut l = Layout { entries: BTreeMap::new(), seq: 0 };
        let page = l.mint("page");
        l.insert(page.clone(), Entry::Page { name: DEFAULT_PAGE_NAME.into(), order: 0 });
        let panel = l.mint("panel");
        l.insert(
            panel,
            Entry::Panel {
                parent: page,
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

    /// Every page, in order.
    pub fn pages(&self) -> Vec<Id> {
        let mut p: Vec<(usize, &Id)> = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(e, Entry::Page { .. }))
            .map(|(id, e)| (e.order(), id))
            .collect();
        p.sort();
        p.into_iter().map(|(_, id)| id.clone()).collect()
    }

    pub fn page_named(&self, name: &str) -> Option<Id> {
        self.entries
            .iter()
            .find(|(_, e)| matches!(e, Entry::Page { name: n, .. } if n == name))
            .map(|(id, _)| id.clone())
    }

    pub fn name_of(&self, page: &str) -> Option<&str> {
        match self.entries.get(page) {
            Some(Entry::Page { name, .. }) => Some(name),
            _ => None,
        }
    }

    /// The page an entry belongs to (itself, if it is one). `None` on a dangling parent or a cycle —
    /// the walk is bounded by the entry count, which is what makes [`Self::validate`] catch both in
    /// one step.
    fn page_of(&self, id: &str) -> Option<Id> {
        let mut cur = id;
        for _ in 0..=self.entries.len() {
            match self.entries.get(cur)? {
                Entry::Page { .. } => return Some(cur.to_string()),
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

    /// The panel ids inside `page`, in document order.
    pub fn panels(&self, page: &str) -> Vec<Id> {
        self.subtree(page)
            .into_iter()
            .filter(|i| matches!(self.entries.get(i), Some(Entry::Panel { .. })))
            .collect()
    }

    /// An entry's fractional (width, height) within its page — the product of the axis-relative
    /// sizes along the path up to the page. What `page_list_panels` reports.
    pub fn extent(&self, id: &str) -> (f64, f64) {
        let (mut w, mut h) = (1.0, 1.0);
        let mut cur = id.to_string();
        for _ in 0..=self.entries.len() {
            let Some(e) = self.entries.get(&cur) else { break };
            let Some(p) = e.parent() else { break };
            match self.entries.get(p) {
                Some(Entry::Split { axis: Axis::Row, .. }) => w *= e.size(),
                Some(Entry::Split { axis: Axis::Column, .. }) => h *= e.size(),
                _ => {} // a page: its single root fills it on both axes
            }
            cur = p.to_string();
        }
        (w, h)
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

    /// Refuse an op that addresses a panel through the wrong page — the `page` argument is what
    /// makes a mistyped panel id a teachable error instead of a silent edit somewhere else.
    fn in_page(&self, page: &str, id: &str) -> Result<(), String> {
        match self.page_of(id) {
            Some(p) if p == page => Ok(()),
            Some(p) => Err(format!(
                "`{id}` is on page `{}`, not `{}`",
                self.name_of(&p).unwrap_or(&p),
                self.name_of(page).unwrap_or(page)
            )),
            None => Err(format!("no such panel `{id}`")),
        }
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
        let parent = e.parent().ok_or("a page is not inside anything")?.to_string();
        if matches!(self.entries.get(&parent), Some(Entry::Page { .. })) {
            return Err(format!(
                "`{id}` is page `{}`'s only root — a page always keeps one",
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

    /// Add a page and return its id. It holds one fresh node-editor panel — unless `subtree` names
    /// an existing one, in which case the page is built AROUND it: the frozen drop-onto-the-tab-bar
    /// gesture, which `session_add_page` + `page_move_panel` cannot express (a move needs a split to
    /// land in, and a fresh page has none). `index` places it in the tab strip.
    pub fn add_page(
        &self,
        name: &str,
        index: Option<usize>,
        subtree: Option<&str>,
    ) -> Result<(Vec<Write>, Id), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("a page needs a name".into());
        }
        if self.page_named(name).is_some() {
            return Err(format!("a page named `{name}` already exists"));
        }
        let mut next = self.clone();
        // Lifted FIRST, because taking a page's last panel takes the page — which is what the new
        // page's own position is counted against.
        let adopted = match subtree {
            Some(s) => Some((s.to_string(), next.take(s)?)),
            None => None,
        };
        let order = next.pages().len();
        let page = next.mint("page");
        next.insert(page.clone(), Entry::Page { name: name.to_string(), order });
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
        entry.set_parent(&page);
        entry.set_order(0);
        entry.set_size(1.0);
        next.insert(root, entry);
        if let Some(i) = index {
            let mut order = next.pages();
            order.retain(|p| *p != page);
            order.insert(i.min(order.len()), page.clone());
            next.order_children(&order);
        }
        Ok((self.diff(&next), page))
    }

    /// Lift a subtree out for re-homing. Normally a [`Self::detach`] — but when it is its page's ONLY
    /// root the PAGE goes with it, which is the frozen "the panel was the tab's only node → the tab
    /// goes with it" branch of `_takeNode`. The last page never goes.
    fn take(&mut self, root: &str) -> Result<Entry, String> {
        let parent = match self.entries.get(root) {
            Some(e) => e.parent().map(str::to_string),
            None => return Err(format!("no such panel `{root}`")),
        };
        let Some(parent) = parent else {
            return Err("a page is not a subtree — reorder it with session_reorder_page".into());
        };
        if !matches!(self.entries.get(&parent), Some(Entry::Page { .. })) {
            return self.detach(root);
        }
        if self.pages().len() <= 1 {
            return Err(format!("`{root}` is the only panel on the only page — it has nowhere to go"));
        }
        let e = self.entries.remove(root).expect("looked up above");
        self.entries.remove(&parent);
        let rest = self.pages();
        self.order_children(&rest);
        Ok(e)
    }

    /// Put `entry` beside `target`, splitting along `axis` — `insertNodeAtPanel`, and the ONE place
    /// split-or-wrap lives. A parent already running along `axis` gains an adjacent sibling taking
    /// `f` of the target's slice; otherwise the target is wrapped in a fresh split inheriting its
    /// slot. `before` puts the newcomer left/top. `entry`'s own parent/order/size are overwritten,
    /// so a caller hands over a lifted subtree root or a brand-new panel indifferently.
    fn insert_at(&mut self, id: &str, mut entry: Entry, target: &str, axis: Axis, before: bool, f: f64) {
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
            let wrap = self.mint("split");
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
        page: &str,
        subtree: &str,
        target: &str,
        axis: Axis,
        before: bool,
        ratio: f64,
    ) -> Result<Vec<Write>, String> {
        self.in_page(page, target)?;
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
        let moved = next.take(subtree)?;
        next.insert_at(subtree, moved, target, axis, before, f);
        Ok(self.diff(&next))
    }

    /// Set every child of `split` at once — what a resize drag commits on pointer-up. `set_panel`'s
    /// `size_mult` scales ONE child and renormalizes its siblings, so N of them chase a moving
    /// target and never land on the fraction set the user drew.
    pub fn resize_split(
        &self,
        page: &str,
        split: &str,
        fractions: &[f64],
    ) -> Result<Vec<Write>, String> {
        self.in_page(page, split)?;
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

    pub fn remove_page(&self, name: &str) -> Result<Vec<Write>, String> {
        let page = self.page_named(name).ok_or_else(|| format!("no page named `{name}`"))?;
        if self.pages().len() <= 1 {
            return Err("the last page cannot be removed".into());
        }
        let mut next = self.clone();
        for id in self.subtree(&page) {
            next.entries.remove(&id);
        }
        let rest = next.pages();
        next.order_children(&rest);
        Ok(self.diff(&next))
    }

    /// Rename a page. A field edit, so the id — and therefore every descendant's `parent` — stands.
    pub fn rename_page(&self, from: &str, to: &str) -> Result<Vec<Write>, String> {
        let page = self.page_named(from).ok_or_else(|| format!("no page named `{from}`"))?;
        let to = to.trim();
        if to.is_empty() {
            return Err("a page needs a name".into());
        }
        if self.page_named(to).is_some_and(|other| other != page) {
            return Err(format!("a page named `{to}` already exists"));
        }
        let mut next = self.clone();
        if let Some(Entry::Page { name, .. }) = next.entries.get_mut(&page) {
            *name = to.to_string();
        }
        Ok(self.diff(&next))
    }

    pub fn reorder_page(&self, name: &str, to_index: usize) -> Result<Vec<Write>, String> {
        let page = self.page_named(name).ok_or_else(|| format!("no page named `{name}`"))?;
        let mut order = self.pages();
        order.retain(|p| *p != page);
        order.insert(to_index.min(order.len()), page);
        let mut next = self.clone();
        next.order_children(&order);
        Ok(self.diff(&next))
    }

    /// Split `panel` along `axis`, birthing an EMPTY panel that takes `ratio` of its slot — the same
    /// [`Self::insert_at`] a drop uses, handed a brand-new panel instead of a lifted subtree.
    pub fn split_panel(
        &self,
        page: &str,
        panel: &str,
        axis: Axis,
        ratio: f64,
    ) -> Result<(Vec<Write>, Id), String> {
        self.in_page(page, panel)?;
        match self.entries.get(panel) {
            Some(Entry::Panel { .. }) => {}
            Some(e) => return Err(format!("`{panel}` is a {} — only a panel splits", e.kind())),
            None => return Err(format!("no such panel `{panel}`")),
        }
        let f = fraction(ratio)?;
        let mut next = self.clone();
        let fresh = next.mint("panel");
        // parent/order/size are `insert_at`'s to set; only the type and state are this op's.
        let born = Entry::Panel {
            parent: String::new(),
            order: 0,
            size: 1.0,
            panel_type: EMPTY_PANEL_TYPE.into(),
            state: Value::Null,
        };
        next.insert_at(&fresh, born, panel, axis, false, f);
        Ok((self.diff(&next), fresh))
    }

    /// Set a panel's type, state and/or size. `panel_type` is applied FIRST because changing it
    /// clears the old type's state (a new type cannot interpret it) — which is exactly why a
    /// combined `{type, state}` has to land the state afterwards, and why re-asserting the SAME
    /// type must not wipe (an agent passing `type` redundantly beside `size_mult` would otherwise
    /// destroy a live binding).
    pub fn set_panel(
        &self,
        page: &str,
        panel: &str,
        panel_type: Option<&str>,
        state: Option<Value>,
        size_mult: Option<f64>,
    ) -> Result<Vec<Write>, String> {
        self.in_page(page, panel)?;
        if let Some(m) = size_mult {
            if !m.is_finite() || m <= 0.0 {
                return Err("size_mult must be a positive number".into());
            }
        }
        let mut next = self.clone();
        let Some(Entry::Panel { panel_type: pt, state: st, .. }) = next.entries.get_mut(panel) else {
            let kind = self.entries.get(panel).map_or("entry", Entry::kind);
            return Err(format!("`{panel}` is a {kind} — only a panel carries a type and state"));
        };
        if let Some(t) = panel_type.filter(|t| *t != pt) {
            *pt = t.to_string();
            *st = Value::Null;
        }
        if let Some(s) = state {
            *st = s;
        }
        if let Some(m) = size_mult {
            let parent = next.entries[panel].parent().unwrap_or_default().to_string();
            let scaled = (next.entries[panel].size() * m).max(MIN_FRACTION);
            if let Some(e) = next.entries.get_mut(panel) {
                e.set_size(scaled);
            }
            next.normalize(&parent);
        }
        Ok(self.diff(&next))
    }

    /// Move the subtree rooted at `root` under `new_parent` at `order_index`. A panel is a subtree
    /// of one, so this covers both the panel case and the tab-onto-panel merge that carries an
    /// arbitrary subtree across pages — identity, state and every descendant preserved.
    pub fn move_subtree(
        &self,
        page: &str,
        root: &str,
        new_parent: &str,
        order_index: usize,
    ) -> Result<Vec<Write>, String> {
        self.in_page(page, root)?;
        let e = &self.entries[root];
        if matches!(e, Entry::Page { .. }) {
            return Err("a page is not a subtree — reorder it with session_reorder_page".into());
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
    pub fn remove_subtree(&self, page: &str, root: &str) -> Result<Vec<Write>, String> {
        self.in_page(page, root)?;
        if matches!(self.entries.get(root), Some(Entry::Page { .. })) {
            return Err("a page is removed with session_remove_page".into());
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
                Entry::Page { name, .. } => {
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
                "page" => Entry::Page {
                    name: text("name").ok_or_else(|| format!("arrangement: page `{id}` has no name"))?,
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
        let pages = self.pages();
        if pages.is_empty() {
            return Err("arrangement: no pages".into());
        }
        let mut names = std::collections::HashSet::new();
        for p in &pages {
            let n = self.name_of(p).unwrap_or_default();
            if !names.insert(n) {
                return Err(format!("arrangement: two pages are both named `{n}`"));
            }
        }
        // The per-entry checks run FIRST, because an entry that reaches no page is the CAUSE of the
        // page-root count being wrong, and a caller shown only the symptom cannot find the entry.
        for (id, e) in &self.entries {
            // Walking up to a page refuses a dangling parent AND a cycle in the same step.
            if self.page_of(id).is_none() {
                return Err(format!(
                    "arrangement: `{id}` reaches no page — a missing parent, or a cycle"
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
        for p in &pages {
            // A page holds exactly one root — the nested tree's `Workspace.root`. Zero or many is a
            // shape flattening admits and rendering cannot.
            let roots = self.children(p).len();
            if roots != 1 {
                let n = self.name_of(p).unwrap_or_default();
                return Err(format!("arrangement: page `{n}` has {roots} roots, not 1"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn applied(l: &Layout, w: Vec<Write>) -> Layout {
        let mut n = l.clone();
        n.apply(w);
        n
    }

    /// The default page's single panel — the handle every test splits from.
    fn root_panel(l: &Layout) -> Id {
        let page = l.pages()[0].clone();
        l.children(&page)[0].clone()
    }

    #[test]
    fn the_default_arrangement_is_one_page_holding_one_editor_panel() {
        let l = Layout::default();
        assert_eq!(l.pages().len(), 1, "one page");
        let page = &l.pages()[0];
        assert_eq!(l.name_of(page), Some(DEFAULT_PAGE_NAME));
        let kids = l.children(page);
        assert_eq!(kids.len(), 1, "a page holds exactly one root node");
        match l.get(&kids[0]) {
            Some(Entry::Panel { panel_type, size, .. }) => {
                assert_eq!(panel_type, DEFAULT_PANEL_TYPE);
                assert_eq!(*size, 1.0, "the page's root fills it");
            }
            other => panic!("expected a panel, got {other:?}"),
        }
        assert!(l.validate().is_ok(), "the default arrangement is valid");
    }

    #[test]
    fn splitting_along_the_parents_own_axis_inserts_a_sibling_and_halves_the_slot() {
        // model.ts `insertNodeAtPanel`: when the target's parent already runs along `direction`, the
        // new node is inserted ADJACENT and takes `fraction` of the target's slice.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, _b) = l.split_panel(&page, &a, Axis::Row, 0.5).expect("first split");
        let l = applied(&l, w);
        let split = l.children(&page)[0].clone();
        assert_eq!(l.get(&split).map(Entry::kind), Some("split"), "the panel got wrapped");

        // A second split of `a` along the SAME axis inserts into the existing split, no new wrapper.
        let (w, c) = l.split_panel(&page, &a, Axis::Row, 0.5).expect("second split");
        let l2 = applied(&l, w);
        assert_eq!(l2.children(&page)[0], split, "still the same split — no second wrapper");
        let kids = l2.children(&split);
        assert_eq!(kids.len(), 3, "inserted as a sibling");
        assert_eq!(kids[1], c, "the new panel lands right after its target");
        // `a` held 0.5 of the row and gives half of that away.
        assert!((l2.get(&a).unwrap().size() - 0.25).abs() < 1e-9, "target keeps half its slice");
        assert!((l2.get(&c).unwrap().size() - 0.25).abs() < 1e-9, "the new panel takes the other half");
        assert!(matches!(l2.get(&c), Some(Entry::Panel { panel_type, .. }) if panel_type == EMPTY_PANEL_TYPE));
    }

    #[test]
    fn splitting_across_the_axis_wraps_the_target_in_a_fresh_split() {
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, _) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let row = l.children(&page)[0].clone();

        let (w, c) = l.split_panel(&page, &a, Axis::Column, 0.25).unwrap();
        let l = applied(&l, w);
        // `a` is now under a NEW column split that took `a`'s slot in the row.
        let wrap = l.get(&a).unwrap().parent().unwrap().to_string();
        assert_ne!(wrap, row, "a fresh wrapper, not the row");
        assert!(matches!(l.get(&wrap), Some(Entry::Split { axis: Axis::Column, .. })));
        assert_eq!(l.get(&wrap).unwrap().parent(), Some(row.as_str()));
        assert!((l.get(&wrap).unwrap().size() - 0.5).abs() < 1e-9, "the wrapper inherits the slot");
        assert_eq!(l.children(&wrap), vec![a.clone(), c.clone()], "target first, new panel after");
        assert!((l.get(&a).unwrap().size() - 0.75).abs() < 1e-9);
        assert!((l.get(&c).unwrap().size() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn removing_a_panel_promotes_the_last_sibling_into_the_splits_place() {
        // model.ts `closePanel`: the freed space goes to the siblings, and a split left with ONE
        // child is replaced by that child — inheriting its slot, so the tree never grows a stub.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let split = l.children(&page)[0].clone();
        let (w, c) = l.split_panel(&page, &b, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);

        // Three children: removing one leaves two — the split survives and the sizes renormalize.
        let l3 = applied(&l, l.remove_subtree(&page, &c).unwrap());
        assert_eq!(l3.children(&split).len(), 2);
        let total: f64 = l3.children(&split).iter().map(|i| l3.get(i).unwrap().size()).sum();
        assert!((total - 1.0).abs() < 1e-9, "sizes renormalize to 1, got {total}");
        assert_eq!(l3.get(&c), None, "the removed panel is gone");

        // A stored arrangement is range-checked but not SUM-checked, so sizes can arrive not adding
        // up. Handing the freed slice out proportionally preserves whatever the total was; the
        // renormalize is the separate step that pulls the split back to a full page.
        let mut v = l.to_json();
        for id in [&a, &b, &c] {
            v[id]["size"] = json!(0.2);
        }
        let skewed = Layout::from_json(&v).expect("0.2 is a valid size, even three of them");
        let closed = applied(&skewed, skewed.remove_subtree(&page, &c).unwrap());
        let total: f64 = closed.children(&split).iter().map(|i| closed.get(i).unwrap().size()).sum();
        assert!((total - 1.0).abs() < 1e-9, "a skewed split renormalizes on close, got {total}");

        // Removing again leaves ONE child → the split is promoted away and `a` becomes the root.
        let l4 = applied(&l3, l3.remove_subtree(&page, &b).unwrap());
        assert_eq!(l4.get(&split), None, "the one-child split is promoted away");
        assert_eq!(l4.children(&page), vec![a.clone()], "the survivor takes the page's root slot");
        assert!((l4.get(&a).unwrap().size() - 1.0).abs() < 1e-9);

        // The page's last panel cannot be closed.
        assert!(l4.remove_subtree(&page, &a).is_err(), "a page must keep a panel");
    }

    #[test]
    fn moving_a_subtree_preserves_panel_identity_and_every_descendant() {
        // The tab-onto-panel merge gesture moves an arbitrary subtree across pages. Identity is the
        // point: the SAME panel ids arrive, with their state, under the new parent.
        let l = Layout::default();
        let p1 = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&p1, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let (w, c) = l.split_panel(&p1, &b, Axis::Column, 0.5).unwrap();
        let l = applied(&l, w);
        let sub = l.get(&b).unwrap().parent().unwrap().to_string(); // the column split holding b + c
        let l = applied(&l, l.set_panel(&p1, &c, None, Some(json!({ "node": "osc0" })), None).unwrap());

        // A second page, so the move crosses pages.
        let l = applied(&l, l.add_page("Second", None, None).unwrap().0);
        let p2 = l.page_named("Second").expect("the new page");
        let d = l.children(&p2)[0].clone();
        let (w, _e) = l.split_panel(&p2, &d, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let dest = l.children(&p2)[0].clone();

        let l2 = applied(&l, l.move_subtree(&p1, &sub, &dest, 0).unwrap());
        assert_eq!(l2.get(&sub).unwrap().parent(), Some(dest.as_str()), "the subtree re-parented");
        assert_eq!(l2.children(&dest)[0], sub, "at the index asked for");
        assert_eq!(l2.page_of(&c).as_deref(), Some(p2.as_str()), "descendants came along");
        assert!(
            matches!(l2.get(&c), Some(Entry::Panel { state, .. }) if state["node"] == json!("osc0")),
            "the moved panel kept its identity AND its binding"
        );
        // The newcomer takes an equal share, and the destination still fills exactly its own slot.
        let dest_total: f64 = l2.children(&dest).iter().map(|i| l2.get(i).unwrap().size()).sum();
        assert!((dest_total - 1.0).abs() < 1e-9, "the destination split stays full, got {dest_total}");
        // The vacated page collapses onto its survivor rather than keeping an empty split.
        assert_eq!(l2.children(&p1), vec![a.clone()]);
        assert!(l2.validate().is_ok());

        // A move into one's own subtree would make a cycle — refused, not corrupted.
        assert!(l.move_subtree(&p1, &sub, &c, 0).is_err(), "a cycle is refused");
        assert!(l.move_subtree(&p1, &sub, &sub, 0).is_err(), "so is parenting to oneself");
    }

    #[test]
    fn set_panel_applies_the_type_before_the_state() {
        // A combined `{type, state}` must land the BINDING, not a wiped one — switching type clears
        // the old type's state (`setPanelType`), so the order is the whole contract.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let l = applied(
            &l,
            l.set_panel(&page, &a, Some("viewer"), Some(json!({ "node": "osc0", "slot": "out" })), None)
                .unwrap(),
        );
        match l.get(&a) {
            Some(Entry::Panel { panel_type, state, .. }) => {
                assert_eq!(panel_type, "viewer");
                assert_eq!(state["node"], json!("osc0"), "the binding survived the type change");
            }
            other => panic!("expected a panel: {other:?}"),
        }
        // A type change ALONE wipes the old type's state (it cannot interpret it)…
        let l2 = applied(&l, l.set_panel(&page, &a, Some("console"), None, None).unwrap());
        assert!(matches!(l2.get(&a), Some(Entry::Panel { state, .. }) if state.is_null()));
        // …but re-asserting the SAME type must not, or an agent passing type redundantly alongside
        // `size_mult` would silently destroy a live binding.
        let l3 = applied(&l, l.set_panel(&page, &a, Some("viewer"), None, Some(2.0)).unwrap());
        assert!(matches!(l3.get(&a), Some(Entry::Panel { state, .. }) if state["node"] == json!("osc0")));
    }

    #[test]
    fn a_page_is_addressed_by_name_and_keeps_its_id_across_a_rename() {
        let l = Layout::default();
        let p1 = l.pages()[0].clone();
        let a = root_panel(&l);
        let l = applied(&l, l.add_page("Second", None, None).unwrap().0);
        assert!(l.add_page("Second", None, None).is_err(), "a duplicate page name is refused teachably");

        let l = applied(&l, l.rename_page(DEFAULT_PAGE_NAME, "Main").unwrap());
        assert_eq!(l.page_named("Main"), Some(p1.clone()), "the id survived the rename");
        assert_eq!(l.get(&a).unwrap().parent(), Some(p1.as_str()), "and every descendant's parent");

        // Reordering moves the page, not its contents.
        let l = applied(&l, l.reorder_page("Main", 1).unwrap());
        assert_eq!(l.pages()[1], p1, "Main is now second");
        assert_eq!(l.page_of(&a).as_deref(), Some(p1.as_str()));

        // Removing a page takes its whole subtree; the last page cannot go.
        let l = applied(&l, l.remove_page("Main").unwrap());
        assert_eq!(l.get(&a), None, "the page's panels went with it");
        assert!(l.remove_page("Second").is_err(), "the last page stays");
    }

    #[test]
    fn the_arrangement_round_trips_through_json() {
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, _b) = l.split_panel(&page, &a, Axis::Column, 0.3).unwrap();
        let l = applied(&l, w);
        let l = applied(&l, l.set_panel(&page, &a, Some("viewer"), Some(json!({ "node": "x" })), None).unwrap());
        let back = Layout::from_json(&l.to_json()).expect("its own output is valid");
        assert_eq!(back, l, "byte-for-byte the same arrangement");
    }

    #[test]
    fn from_json_refuses_every_corruption_class_flattening_admits() {
        // The nested tree made these structurally impossible; the flat map does not. Each one is a
        // load that must fall back to the default rather than render a broken arrangement.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let split = l.children(&page)[0].clone();
        let good = l.to_json();
        assert!(Layout::from_json(&good).is_ok(), "the fixture itself is valid");

        let corrupt = |f: &dyn Fn(&mut Value)| {
            let mut v = good.clone();
            f(&mut v);
            Layout::from_json(&v)
        };
        assert!(corrupt(&|v| v[&a]["parent"] = json!("nope")).is_err(), "dangling parent");
        assert!(
            corrupt(&|v| {
                v[&split]["parent"] = json!(a.clone());
                v[&a]["parent"] = json!(split.clone());
            })
            .is_err(),
            "a parent cycle never reaches a page"
        );
        assert!(corrupt(&|v| v[&b]["parent"] = json!(page.clone())).is_err(), "a page has one root, not two");
        assert!(corrupt(&|v| v[&b]["order"] = json!(0)).is_err(), "duplicate order among siblings");
        assert!(corrupt(&|v| v[&b]["size"] = json!(1.5)).is_err(), "a size outside (0, 1]");
        assert!(corrupt(&|v| v[&b]["size"] = json!(0.0)).is_err(), "a zero-size child");
        assert!(corrupt(&|v| v[&b]["panel_type"] = json!("")).is_err(), "a panel with no type");
        assert!(corrupt(&|v| v[&b]["state"] = json!("{not json")).is_err(), "malformed panel state");
        // A state written as a MAP — the natural shape of a hand-edited `.gfi` — must be refused,
        // not read as absent: silently loading it would wipe the binding it was carrying.
        assert!(
            corrupt(&|v| v[&b]["state"] = json!({ "node": "osc0" })).is_err(),
            "a panel state that is not a JSON string"
        );
        assert!(corrupt(&|v| v[&b]["kind"] = json!("gadget")).is_err(), "an unknown entry kind");
        assert!(
            corrupt(&|v| {
                v.as_object_mut().unwrap().remove(&page);
            })
            .is_err(),
            "removing the page orphans everything"
        );
        assert!(Layout::from_json(&json!({})).is_err(), "an arrangement with no page at all");
        assert!(Layout::from_json(&json!([])).is_err(), "not even an object");
        // Two pages sharing a name would make the name-addressed ops ambiguous.
        let two = applied(&l, l.add_page("Second", None, None).unwrap().0);
        let dup = two.page_named("Second").unwrap();
        let mut v = two.to_json();
        v[&dup]["name"] = json!(DEFAULT_PAGE_NAME);
        assert!(Layout::from_json(&v).is_err(), "duplicate page names");
    }

    #[test]
    fn a_drop_onto_a_panel_re_homes_a_subtree_in_one_plan() {
        // `dropOnPanel` is ONE frozen gesture, so it is one plan: the source is lifted (`_takeNode`)
        // and re-homed beside the target in the same step. Composed from split + move + close it
        // would be three undo steps for one drag, and every peer would see two arrangements that
        // were never on screen.
        let l = Layout::default();
        let p1 = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&p1, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let l = applied(&l, l.set_panel(&p1, &b, None, Some(json!({ "node": "osc0" })), None).unwrap());
        let (w, p2) = l.add_page("Second", None, None).unwrap();
        let l = applied(&l, w);
        let target = l.children(&p2)[0].clone();

        let l2 = applied(&l, l.insert_at_panel(&p2, &b, &target, Axis::Column, false, 0.25).unwrap());
        assert_eq!(l2.page_of(&b).as_deref(), Some(p2.as_str()), "the panel crossed pages");
        assert!(
            matches!(l2.get(&b), Some(Entry::Panel { state, .. }) if state["node"] == json!("osc0")),
            "with its identity and its binding"
        );
        let wrap = l2.get(&b).unwrap().parent().unwrap().to_string();
        assert!(matches!(l2.get(&wrap), Some(Entry::Split { axis: Axis::Column, .. })), "wrapped");
        assert_eq!(l2.children(&wrap), vec![target.clone(), b.clone()], "landing AFTER the target");
        assert!((l2.get(&b).unwrap().size() - 0.25).abs() < 1e-9, "taking the fraction asked for");
        assert_eq!(l2.children(&p1), vec![a.clone()], "the page it left promoted its survivor");
        assert!(l2.validate().is_ok());

        // `placeBefore`, and the branch where a page's LAST panel leaves: the tab goes with it,
        // which is why this cannot be a plain move (a move refuses to empty a page).
        let l3 = applied(&l, l.insert_at_panel(&p1, &target, &a, Axis::Row, true, 0.5).unwrap());
        assert_eq!(l3.pages(), vec![p1.clone()], "the emptied page went with its panel");
        let row = l3.get(&a).unwrap().parent().unwrap().to_string();
        assert_eq!(l3.children(&row), vec![target.clone(), a.clone(), b.clone()], "placed BEFORE");
        assert!(l3.validate().is_ok());

        assert!(l.insert_at_panel(&p2, &target, &target, Axis::Row, false, 0.5).is_err(), "onto itself");
        let split = l.get(&a).unwrap().parent().unwrap().to_string();
        assert!(l.insert_at_panel(&p1, &split, &a, Axis::Row, false, 0.5).is_err(), "a cycle is refused");
    }

    #[test]
    fn a_drop_onto_the_tab_bar_builds_the_page_around_the_panel_it_carries() {
        // `dropPanelOnTabBar`: a new tab whose root IS the dragged subtree. `session_add_page` plus a
        // move cannot express it — a move needs a split to land in, and a fresh page has none.
        let l = Layout::default();
        let p1 = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&p1, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);

        let (w, p2) = l.add_page("Second", Some(0), Some(&b)).unwrap();
        let l2 = applied(&l, w);
        assert_eq!(l2.children(&p2), vec![b.clone()], "the page was built AROUND the dragged panel");
        assert_eq!(l2.pages()[0], p2, "at the tab index asked for");
        assert!((l2.get(&b).unwrap().size() - 1.0).abs() < 1e-9, "filling its new page");
        assert_eq!(l2.children(&p1), vec![a.clone()], "the page it left promoted its survivor");
        assert!(l2.validate().is_ok());

        // The only panel of the only page has nowhere to go — there would be no page left.
        let one = Layout::default();
        assert!(one.add_page("Second", None, Some(&root_panel(&one))).is_err());
        // With another page standing it may, and its emptied page goes with it.
        let l3 = applied(&l2, l2.add_page("Third", None, Some(&a)).unwrap().0);
        assert_eq!(l3.pages().len(), 2, "the emptied page went with its panel");
        assert!(l3.validate().is_ok());
    }

    #[test]
    fn a_resize_sets_every_child_of_a_split_at_once() {
        // A drag commits FINAL fractions on pointer-up. `size_mult` scales one child and renormalizes
        // its siblings, so N of them chase a moving target and never land on the set the user drew.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let (w, c) = l.split_panel(&page, &b, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let split = l.children(&page)[0].clone();
        assert_eq!(l.children(&split), vec![a.clone(), b.clone(), c.clone()]);

        let l2 = applied(&l, l.resize_split(&page, &split, &[0.2, 0.3, 0.5]).unwrap());
        for (id, want) in [(&a, 0.2), (&b, 0.3), (&c, 0.5)] {
            assert!((l2.get(id).unwrap().size() - want).abs() < 1e-9, "`{id}` got the share asked for");
        }
        assert!(l2.validate().is_ok());
        // Fractions that do not add up are RATIOS: the drag hands over what it drew and the split
        // still fills exactly its own slot.
        let l3 = applied(&l, l.resize_split(&page, &split, &[1.0, 1.0, 2.0]).unwrap());
        for (id, want) in [(&a, 0.25), (&b, 0.25), (&c, 0.5)] {
            assert!((l3.get(id).unwrap().size() - want).abs() < 1e-9, "`{id}` renormalized");
        }
        assert!(l.resize_split(&page, &split, &[0.5, 0.5]).is_err(), "one fraction per child");
        assert!(l.resize_split(&page, &a, &[1.0]).is_err(), "a panel has no children to size");
    }

    #[test]
    fn a_closed_panels_id_is_never_minted_again() {
        // `model.ts`'s `_seq` only counts up, and a client keys per-panel state by panel id — its
        // viewpoint's `subpatchPath` among them. Handing a fresh panel a dead one's id silently
        // gives it the dead one's navigation.
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let closed = applied(&l, l.remove_subtree(&page, &b).unwrap());
        assert_eq!(closed.get(&b), None, "the fixture really did close it");
        let (_w, next) = closed.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        assert_ne!(next, b, "the closed panel's id is not handed out again");

        // …and the counter rides the file, so a reopened patch keeps counting FORWARD rather than
        // restarting from whatever survived.
        let back = Layout::from_json(&closed.to_json()).expect("its own output is valid");
        let (_w, later) = back.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        assert_ne!(later, b, "a reopened arrangement does not restart the counter");
    }

    #[test]
    fn extent_reports_a_panels_share_of_its_page_on_both_axes() {
        let l = Layout::default();
        let page = l.pages()[0].clone();
        let a = root_panel(&l);
        let (w, b) = l.split_panel(&page, &a, Axis::Row, 0.5).unwrap();
        let l = applied(&l, w);
        let (w, c) = l.split_panel(&page, &b, Axis::Column, 0.5).unwrap();
        let l = applied(&l, w);
        let (bw, bh) = l.extent(&b);
        assert!((bw - 0.5).abs() < 1e-9 && (bh - 0.5).abs() < 1e-9, "half the width, half the height");
        let (cw, ch) = l.extent(&c);
        assert!((cw - 0.5).abs() < 1e-9 && (ch - 0.5).abs() < 1e-9);
        assert_eq!(l.extent(&a), (0.5, 1.0), "a spans the full height of its column");
        assert_eq!(l.panels(&page).len(), 3);
    }
}
