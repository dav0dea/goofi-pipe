/**
 * The agent-harness roster, and which panel is looking at which instance.
 *
 * The roster is the manager's: it rides the `hello`/`graph_replaced` snapshot and every
 * `harness_changed` carries the WHOLE shape, so a tab that connects to a running patch is correct
 * immediately and one that stays connected never has to diff transitions.
 *
 * **Which instance a panel shows is not.** It is client-local viewpoint, the same class of fact as
 * a node editor's `subpatchPath`: machine-local, dead on unload, never shared layout state — so it
 * cannot dirty the patch, converge to a peer or enter undo. Sub-project A made that separation
 * structural (a layout write is a `set_panel` COMMAND; viewpoint is not), and the way to use
 * that structure rather than re-derive it is to send nothing at all. There is no `setPanelState`
 * below, no `LayoutIntent`, and `harness.test.ts` asserts the whole store records zero calls for a
 * mount/choose/unmount round.
 *
 * A binding is released on unmount, because it says which LIVE panel is showing what. What
 * survives a panel is the terminal itself — see `termSession.ts`, which is keyed by instance.
 */
import { getControl } from '$lib/api/control';
import type {
	Control,
	ControlEvent,
	DetectedHarness,
	HarnessInstanceInfo,
	HarnessRoster
} from '$lib/api/control';
import { endTermSession, liveTermSessions } from './termSession';
import { notify } from './notify.svelte';

export type { HarnessRoster };

/** How an instance is named wherever it is offered: the switcher, the launcher's attach button,
 * the TopBar's menu and the close question. One spelling, so the four never drift. */
export function harnessLabel(i: { harness: string; id: string }): string {
	return `${i.harness} · ${i.id.slice(0, 6)}`;
}

export class HarnessStore {
	instances = $state<HarnessInstanceInfo[]>([]);
	detected = $state<DetectedHarness[]>([]);
	/** The instance a close was asked about, and the panel to close once it is answered (the
	 * header's ✕ closes; the TopBar's badge does not). Null when nothing is being asked. */
	closing = $state<{ id: string; closePanel: string | null } | null>(null);
	/** Mounted agent panels → the instance each is showing. `undefined` is a panel that has not
	 * chosen yet (it auto-claims); `null` is one the user deliberately let go of, which must stay
	 * on its launcher rather than re-claiming the instance it just detached from. */
	private panels = $state<Record<string, string | null | undefined>>({});
	private ctl: Control;

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.on((ev) => this.handle(ev));
	}

	/** What the badge counts. Every instance here can still answer — see `adopt`. */
	get running(): number {
		return this.instances.length;
	}

	/** Any mounted agent panel — where a question about an instance nothing is showing can be
	 * asked, rather than opening a second panel for it. */
	get firstPanel(): string | null {
		return Object.keys(this.panels)[0] ?? null;
	}

	private handle(ev: ControlEvent): void {
		if (ev.event === 'hello' || ev.event === 'graph_replaced') this.adopt(ev.payload.harnesses);
		else if (ev.event === 'harness_changed') this.adopt(ev.payload);
	}

	private adopt(r: HarnessRoster | undefined): void {
		const seen = r?.instances ?? [];
		// A dead agent is GONE as far as this app is concerned (user, 2026-08-10): it has no
		// interactivity and no state left, so the panel showing it goes back to its launcher rather
		// than freezing on a corpse, and there is no dismissal to ask for. Everything below then
		// follows from the roster being live-only — the badge counts it, the switcher offers it, a
		// panel claims it.
		for (const i of seen) if (i.state === 'exited') this.bury(i);
		this.instances = seen.filter((i) => i.state !== 'exited');
		this.detected = r?.detected ?? [];
		// An instance that LEFT the roster (dismissed, torn down with its patch, or just buried)
		// takes its terminal with it.
		const known = new Set(this.instances.map((i) => i.id));
		for (const id of liveTermSessions()) if (!known.has(id)) endTermSession(id);
		if (this.closing && !known.has(this.closing.id)) this.closing = null;
		for (const p of Object.keys(this.panels)) this.claim(p);
	}

	/** An instance that has just died, seen once: it is on the roster we are adopting and was on the
	 * one before, alive. Only a death NOBODY ASKED FOR is worth interrupting for — and `stopping` is
	 * exactly that distinction, broadcast the moment a stop is asked for, a whole grace before the
	 * child goes. A clean exit is the user typing `exit`; the panel returning to its launcher is the
	 * whole of that news. Then the manager is asked to drop it, so the two rosters agree; another
	 * tab asking the same thing is answered with an error, which is nothing to report. */
	private bury(i: HarnessInstanceInfo): void {
		const was = this.instances.find((o) => o.id === i.id);
		if (!was) return;
		if (was.state === 'running' && i.exit_code)
			notify().raise(`${harnessLabel(i)} exited unexpectedly (code ${i.exit_code})`);
		void this.ctl.call('stop_harness', { instance: i.id }).catch(() => {});
	}

	mount(panelId: string): void {
		if (!(panelId in this.panels)) this.panels[panelId] = undefined;
		this.claim(panelId);
	}

	unmount(panelId: string): void {
		delete this.panels[panelId];
	}

	/** The switcher's explicit pick. */
	show(panelId: string, id: string | null): void {
		this.panels[panelId] = id;
	}

	/** The instance this panel is showing, or null — a binding to an instance that has left the
	 * roster reads as none, so every stale one heals itself. */
	instanceFor(panelId: string): string | null {
		const id = this.panels[panelId];
		return id && this.instances.some((i) => i.id === id) ? id : null;
	}

	panelShowing(id: string): string | null {
		return Object.keys(this.panels).find((p) => this.instanceFor(p) === id) ?? null;
	}

	/** A panel with no live instance takes one no OTHER panel is showing. Two agent panels
	 * defaulting to the same instance would fight over one terminal — `term.open` moves it — so
	 * the second shows the next instance, or its own launcher. The instance a close was asked
	 * about is preferred, since a panel may have just been opened to hold that question. */
	claim(panelId: string): void {
		if (this.instanceFor(panelId) || this.panels[panelId] === null) return;
		const taken = new Set(
			Object.keys(this.panels)
				.filter((p) => p !== panelId)
				.map((p) => this.instanceFor(p))
		);
		const free = this.instances.filter((i) => !taken.has(i.id));
		const pick = free.find((i) => i.id === this.closing?.id) ?? free[0];
		if (pick) this.show(panelId, pick.id);
	}

	/** Ask whether to detach or kill. Raised in the panel showing the instance — that is where the
	 * terminal the question is about is — so the caller focuses (or opens) that panel; this store
	 * holds no opinion about layout. */
	requestClose(id: string, closePanel: string | null = null): void {
		this.closing = { id, closePanel };
	}

	cancelClose(): void {
		this.closing = null;
	}

	/** The Detach half's binding: the harness keeps running and keeps its terminal; this panel just
	 * stops showing it. Closing the socket is the view's own act (`TermSession.detach`), and it is
	 * what makes the manager re-arbitrate the size among the views that are left. */
	release(panelId: string): void {
		this.show(panelId, null);
		this.closing = null;
	}

	/** Launch a harness into the panel that asked. The roster arrives on its own event; the
	 * BINDING is what this decides, so the panel shows the terminal it just asked for rather than
	 * whichever instance happened to be free. */
	async launch(panelId: string, harness: string): Promise<void> {
		const born = await this.ctl.call<{ instance_id: string }>('spawn_harness', { harness });
		if (born?.instance_id) this.show(panelId, born.instance_id);
	}

	/** The Kill half — the manager's full stop path (route dropped, SIGTERM, grace, SIGKILL). */
	kill(id: string): void {
		void this.ctl.call('stop_harness', { instance: id });
		this.closing = null;
	}
}

let instance: HarnessStore | null = null;

export function harnesses(): HarnessStore {
	if (!instance) instance = new HarnessStore();
	return instance;
}
