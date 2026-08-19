//! The drive-by guard: one layer over the whole router, so a page the user merely VISITED cannot
//! reach goofi.
//!
//! **What this is not.** goofi is deliberately single-user, local or trusted-LAN, and
//! unauthenticated (CLAUDE.md). This is not authentication and must not grow into one — there is no
//! token, no session, no credential and no rate limit here, and anything that can reach the port
//! *without a browser* is served exactly as before. What it stops is the one attack an
//! unauthenticated local server is actually exposed to: a page on some other site, open in the
//! user's browser, using that browser as a proxy onto localhost.
//!
//! **Why it had to be a layer and not a check in the handlers.** `/control` and `/term` are
//! WebSocket upgrades, and a WS handshake is exempt from CORS — no preflight, no
//! `Access-Control-Allow-Origin` to withhold — so the browser will happily complete a
//! cross-origin socket and hand the page a full read/write channel onto the patch. That is
//! strictly more exposure than `/mcp`, which was at least measured taking only a `text/plain`
//! POST. `/term` raises the stakes again: it is a PTY running the user's own shell with the user's
//! own environment, which makes a drive-by page an RCE rather than a data leak. A guard scoped to
//! the HTTP routes would have shut the smallest of the three doors.
//!
//! **Why an allowlist and not "does Origin match Host".** DNS rebinding: an attacker who points
//! `evil.example` at 127.0.0.1 sends `Origin: http://evil.example` *and* `Host: evil.example`, so
//! a same-origin comparison agrees with itself and lets the page in. Only a host the browser could
//! not have been handed by DNS trickery is admissible, which means a loopback address or an IP
//! LITERAL: to hold an origin of `http://192.168.7.5:8000` a page must have been SERVED from
//! 192.168.7.5:8000, and on this app's trusted LAN that is goofi.

use axum::extract::Request;
use axum::http::{header, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// What the browser said this request IS — the three `Sec-Fetch-*` headers, which script cannot
/// set (they are forbidden header names) and nothing that is not a browser sends at all.
struct Fetch<'a> {
    /// `none` is the user typing the address, `same-origin` a page's own requests; the other two
    /// mean another site asked for this.
    site: Option<&'a str>,
    /// `navigate` is the address bar and every link and reload; anything else is a subresource.
    mode: Option<&'a str>,
    /// `document` is the tab itself; `iframe` is a page putting goofi inside itself.
    dest: Option<&'a str>,
}

/// Whether a request bearing these headers may be served.
///
/// A missing `Origin` is a client that is not a browser — curl, an MCP client, a harness goofi
/// spawned itself — and is served: it is not reachable from a web page, so there is nothing here
/// to stop. Everything else must name a host the browser could only have reached deliberately.
fn allowed(method: &Method, origin: Option<&str>, host: Option<&str>, fetch: Fetch<'_>) -> bool {
    // The user ARRIVING: a link, a bookmark, a reload of the tab that is already here. It carries
    // no `Origin`, a GET writes nothing (every write here is a POST or a socket), and the document
    // it returns is opaque to whoever linked to it — so there is nothing to refuse.
    //
    // Refusing it was this guard's own bug, and a nasty one to describe: a browser REPLAYS the
    // original navigation's `Sec-Fetch-Site` on every later reload, so a tab first opened from
    // anywhere but the address bar answered `cross-site` for ever after, and the only way out was
    // to retype the URL. Restarting goofi and hitting reload is exactly when a user meets it.
    //
    // `document` and not `iframe`: a page that FRAMES goofi is the drive-by this guard is for, and
    // `Sec-Fetch-Dest` is the only thing that tells the two apart. GET or HEAD and not POST: a
    // cross-site form submission is a navigation too, and it is the very hole the check below
    // exists to close.
    if matches!(*method, Method::GET | Method::HEAD)
        && fetch.mode == Some("navigate")
        && fetch.dest == Some("document")
    {
        return true;
    }
    // `Origin` alone would leave one hole: a cross-site FORM POST is a page driving goofi, and a
    // browser that omits `Origin` on one (Safari did until 15.4) reads as "not a browser" above.
    // `Sec-Fetch-Site` closes it. (It does nothing for `/control` and `/term`: Chromium sends no
    // `Sec-Fetch-*` at all on a WebSocket handshake, so those two rest entirely on the allowlist
    // below — which is what the module doc says they rest on.)
    if matches!(fetch.site, Some("cross-site" | "same-site")) {
        return false;
    }
    let Some(origin) = origin else { return true };
    // An `Origin` is a scheme and an authority and nothing else. `null` — a sandboxed frame, a
    // `file://` page — has no authority at all, so it falls through to the parse below and fails.
    let authority = origin.split_once("://").map_or(origin, |(_, rest)| rest);
    let name = match authority.strip_prefix('[') {
        // `[::1]:5173` — the bracketed IPv6 form, whose colons are not a port separator.
        Some(v6) => v6.split_once(']').map_or(v6, |(h, _)| h),
        None => authority.split_once(':').map_or(authority, |(h, _)| h),
    };
    if name == "localhost" {
        return true;
    }
    let Ok(ip) = name.parse::<std::net::IpAddr>() else {
        // A DNS NAME. Loopback is the one name a page cannot be served from without already
        // running code on this machine; every other name could have been rebound.
        return false;
    };
    // A loopback page is a second port on this machine (the `npm run dev` server, another goofi).
    // Any other IP literal has to BE the address this request was sent to, which is the trusted-LAN
    // case: the page came from goofi, because nothing else answers on that address and port.
    ip.is_loopback() || host == Some(authority)
}

/// The layer. Applied once, in [`crate::app`], over the router the fallback is already on.
pub(crate) async fn guard(req: Request, next: Next) -> Response {
    // Scoped so the borrow of `req` ends before it is moved into the handler.
    let ok = {
        let h = req.headers();
        let get = |n: &str| h.get(n).and_then(|v: &axum::http::HeaderValue| v.to_str().ok());
        let fetch = Fetch {
            site: get("sec-fetch-site"),
            mode: get("sec-fetch-mode"),
            dest: get("sec-fetch-dest"),
        };
        allowed(req.method(), get(header::ORIGIN.as_str()), get(header::HOST.as_str()), fetch)
    };
    if ok {
        return next.run(req).await;
    }
    // Plain text, and it names the reason: the only human who will ever read it is a developer
    // looking at a failed request in devtools, and "403" alone would send them hunting.
    (
        StatusCode::FORBIDDEN,
        "goofi refuses requests from a page it did not serve. This is a drive-by guard, not a \
         login: reach goofi at the address it printed at startup.\n",
    )
        .into_response()
}
