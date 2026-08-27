/** The manager's agent-harness roster, plus which panel shows which instance. The latter is
 * client-local viewpoint, so nothing here writes panel state or enters undo. */
import { getControl } from '$lib/api/control';
import type {
	Control,
	ControlEvent,
	AgentEntry,
	HarnessInstanceInfo,
	HarnessRoster
} from '$lib/api/control';
import { endTermSession, liveTermSessions } from './termSession';
import { notify } from './notify.svelte';

export type { HarnessRoster };

/** How an instance is named wherever it is offered. */
export function harnessLabel(i: { harness: string; id: string }): string {
	return `${i.harness} · ${i.id.slice(0, 6)}`;
}

export class HarnessStore {
	instances = $state<HarnessInstanceInfo[]>([]);
	agents = $state<AgentEntry[]>([]);
	configError = $state<string | null>(null);
	/** The instance a close was asked about, and the panel to close once it is answered. */
	closing = $state<{ id: string; closePanel: string | null } | null>(null);
	/** Mounted agent panels → the instance each shows. `undefined` has not chosen yet and
	 * auto-claims; `null` was deliberately let go and must stay on its launcher. */
	private panels = $state<Record<string, string | null | undefined>>({});
	private ctl: Control;

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.on((ev) => this.handle(ev));
	}

	/** What the badge counts. */
	get running(): number {
		return this.instances.length;
	}

	/** Any mounted agent panel — where a question about an unshown instance can be asked. */
	get firstPanel(): string | null {
		return Object.keys(this.panels)[0] ?? null;
	}

	private handle(ev: ControlEvent): void {
		if (ev.event === 'hello' || ev.event === 'graph_replaced') this.adopt(ev.payload.harnesses);
		else if (ev.event === 'harness_changed') this.adopt(ev.payload);
	}

	private adopt(r: HarnessRoster | undefined): void {
		const seen = r?.instances ?? [];
		// The roster this store keeps is live-only: a dead agent is GONE, with no dismissal to ask.
		for (const i of seen) if (i.state === 'exited') this.bury(i);
		this.instances = seen.filter((i) => i.state !== 'exited');
		this.agents = r?.agents ?? [];
		this.configError = r?.config_error ?? null;
		// An instance that LEFT the roster takes its terminal with it.
		const known = new Set(this.instances.map((i) => i.id));
		for (const id of liveTermSessions()) if (!known.has(id)) endTermSession(id);
		if (this.closing && !known.has(this.closing.id)) this.closing = null;
		for (const p of Object.keys(this.panels)) this.claim(p);
	}

	/** An instance seen dying: report only a death nobody asked for, then have the manager drop it. */
	private bury(i: HarnessInstanceInfo): void {
		const was = this.instances.find((o) => o.id === i.id);
		if (!was) return;
		if (was.state === 'running' && i.exit_code)
			notify().raise(`${harnessLabel(i)} exited unexpectedly (code ${i.exit_code})`);
		void this.ctl.call('agent stop', { instance: i.id }).catch(() => {});
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

	/** The instance this panel is showing, or null; a binding to one off the roster reads as none. */
	instanceFor(panelId: string): string | null {
		const id = this.panels[panelId];
		return id && this.instances.some((i) => i.id === id) ? id : null;
	}

	panelShowing(id: string): string | null {
		return Object.keys(this.panels).find((p) => this.instanceFor(p) === id) ?? null;
	}

	/** A panel with no live instance takes one no OTHER panel shows: two panels on one terminal
	 * would fight over it, since `term.open` moves it. */
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

	/** Ask whether to detach or kill; the caller focuses the panel that shows the instance. */
	requestClose(id: string, closePanel: string | null = null): void {
		this.closing = { id, closePanel };
	}

	cancelClose(): void {
		this.closing = null;
	}

	/** The Detach half's binding: the harness keeps running; this panel just stops showing it. */
	release(panelId: string): void {
		this.show(panelId, null);
		this.closing = null;
	}

	/** Launch a harness and bind it to the panel that asked, so that panel shows what it asked for. */
	async launch(panelId: string, harness: string): Promise<void> {
		const born = await this.ctl.call<{ instance_id: string }>('agent start', { name: harness });
		if (born?.instance_id) this.show(panelId, born.instance_id);
	}

	/** The Kill half — the manager's full stop path. */
	kill(id: string): void {
		void this.ctl.call('agent stop', { instance: id });
		this.closing = null;
	}
}

let instance: HarnessStore | null = null;

export function harnesses(): HarnessStore {
	if (!instance) instance = new HarnessStore();
	return instance;
}
