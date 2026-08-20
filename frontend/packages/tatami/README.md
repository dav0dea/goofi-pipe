# tatami

A tab + panel workspace for Svelte 5. Modular mats laid out in fixed arrangements.

Split a panel, drag one onto another's edge, tear one off onto the tab bar, drag the seam between
two. What tatami does not do is decide what any of that MEANS: it holds no tree and writes nothing.
It draws the arrangement you hand it, recognises the gesture over it, and raises the gesture at a
host you implement — so persistence, concurrency, undo and what a tab is called are yours.

```svelte
<script lang="ts">
	import { Tabs, Panels, workspace, memoryHost, registerPanel } from 'tatami';
	import 'tatami/tokens.css';

	registerPanel({ id: 'notes', title: 'Notes', component: Notes });
	workspace().configureHost(memoryHost({ defaultPanelType: 'notes' }));
</script>

<Tabs />
<Panels />
```

## The three shapes

`<Tabs>` alone is a tab strip. `<Panels>` alone is a splittable panel tree. `<Panels>` inside
`<Tabs>` is a workspace — and only there do the cross-boundary drags exist, because the drag bus one
publishes is what the other looks for. There is no `enableTabDragging` flag: a gesture that spans
two components you did not compose is not disabled, it is unexpressible.

## The host

Two contracts, each naming only what its own component can trigger. `TabHost` is panel-free by
construction; `PanelHost` is tab-free. Every method answers whether the write LANDED — a refusal is
ordinary (closing a tab's last panel, a peer having taken what the gesture names) and the panel
system uses the answer to decide where to put the focus, never to retry.

`memoryHost()` is one that keeps the tree in memory and hands it straight back. It is the whole of
the contract implemented over a plain tree — useful on its own, and the thing that proves the port
is complete.

## The panels

Yours. `registerPanel({ id, title, icon, component })` takes any component honouring `PanelProps`;
nothing in here knows what one draws. A panel gets its id, its persisted state bag, and a
`setState` that routes back through your host.

## Styling

Every rule reads `var(--tatami-x, var(--tatami-x-default))`. Import `tokens.css` for the look it
ships with, then set `--tatami-*` to wear your own — on `:root`, on a subtree, or on one instance.
See `src/lib/ui/tokens.css`: it is the contract, and it documents itself.

Icons work the same way: the chrome vendors the ten glyphs it draws, and `registerIcons(table)`
hands over yours for the panel types and menu rows you name.

MIT.
