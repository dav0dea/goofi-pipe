# Demo mode — goofi in a public container

goofi runs on Railway as one container, reached by anyone who holds the URL. `GOOFI_DEMO` is what
makes that deployment coherent: it is the mode a PUBLIC goofi runs in.

## What it is not

It is not a sandbox, and it must never be described as one. A param expression is Python and a
Python node is Python, so a visitor executes arbitrary code by design — that is the product. Demo
mode removes the CONVENIENT doors, never the capability.

What the deployment therefore accepts, deliberately: outbound abuse from the container carries the
host account's address and its bill. The container holds no secret, and the service is kept free of
variables, so there is nothing there to steal.

Every visitor shares ONE patch, because one process holds one graph and one document. Two strangers
see each other's nodes. That is the architecture, and the reset is a restart.

## Decisions

- **Runtime, not compile time.** One boolean folded beside `headless`, exactly as the three headless
  doors fold. A compile-time cut would need `cfg` through the composition root and a second binary
  shape, which costs the one-artifact rule for nothing.
- **One flag, one owner.** The mode is decided once at start, rides the document as ONE field, and
  the frontend reads that field. A per-feature list of switches is what this must not become.
- **An op is DROPPED, not refused.** `ops::table` builds the vocabulary from the mode, so a demo
  server has no `session save` to reject.
- **A route is NOT MOUNTED.** `/exec`, `/mcp`, `/term/{instance}` and `/patch.gfi` are absent.
- **ONE new environment variable, `GOOFI_DEMO`.** It also decides the origin question: the guard
  admits an `Origin` whose host equals the `Host` header, rather than an operator naming the public
  host in a second variable. Rebinding wins an attacker nothing that opening a public URL already
  gives them, and a DNS name is the only way anyone reaches this deployment. goofi reads no `PORT`
  either — that is a hosting platform's spelling, and the container passes `--port`.
- **An idle demo goes QUIET, and never exits.** Untouched for ten minutes, it announces a one-minute
  countdown and then closes every socket. Going quiet is the whole saving: the host platform bills
  for as long as the container talks, and its own idle sleep cannot engage while goofi pings each
  `/data` socket every ten seconds. Exiting would be worse than doing nothing — a stopped container
  does not wake on the next visitor's request, so the demo would stay down until someone redeployed
  it. This is what answers the old open question about a reset on a timer: the sleep IS the reset.
- **The clock is ONE stamp, on the `/control` envelope.** Not a viewer count, not a per-socket timer:
  the instance is idle when nobody at all has spoken, so one visitor at work keeps it up for
  everyone — which is the only reading that matches one shared patch. The envelope rather than
  `AppState::call`, because the status worker calls ops of its own and a machine's edit is not a
  visitor's touch.
- **No audio.** `fresh_graph` registers the signal engine alone. A container has no sound server, so
  the engine would open nothing; leaving it out also takes every audio node out of the catalog, at
  one line and with no second list to keep in step.

## Open

- Whether a visitor who returns to a slept demo should be told the patch is gone, or simply find an
  empty canvas. Today it is the second.
- The `agent` panel type stays in the panel dropdown, because `panelty` registers a type for the
  session and offers no way to withdraw one. The panel answers for itself instead. Withdrawing it
  is a panelty release, never a patch here.
