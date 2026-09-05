//! The drive-by guard: one layer over the whole router, so a page the user merely VISITED cannot
//! reach goofi. It is not authentication and must not grow into one.
//!
//! It is an allowlist and not an "Origin matches Host" comparison because of DNS rebinding: only a
//! host DNS trickery cannot hand a browser — loopback, or an IP literal — is admissible.

use axum::extract::Request;
use axum::http::{header, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// What the browser said this request IS — the three `Sec-Fetch-*` headers, which script cannot
/// set and nothing that is not a browser sends at all.
struct Fetch<'a> {
    site: Option<&'a str>,
    mode: Option<&'a str>,
    dest: Option<&'a str>,
}

/// Whether a request bearing these headers may be served. A missing `Origin` is not a browser and
/// is served; everything else must name a host the browser could only have reached deliberately.
fn allowed(
    method: &Method,
    origin: Option<&str>,
    host: Option<&str>,
    fetch: Fetch<'_>,
    demo: bool,
) -> bool {
    // A navigation is admitted whatever its `Sec-Fetch-Site`, because a browser REPLAYS the
    // original navigation's value on every later reload. `document`, not `iframe`, is the drive-by.
    if matches!(*method, Method::GET | Method::HEAD)
        && fetch.mode == Some("navigate")
        && fetch.dest == Some("document")
    {
        return true;
    }
    // Closes the cross-site FORM POST that Safari before 15.4 sends with no `Origin`. It does
    // nothing for the WebSocket routes: Chromium sends no `Sec-Fetch-*` on a handshake.
    if matches!(fetch.site, Some("cross-site" | "same-site")) {
        return false;
    }
    let Some(origin) = origin else { return true };
    // A `null` origin — a sandboxed frame, a `file://` page — has no authority, so the parse below
    // rejects it.
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
        // A DNS name other than `localhost` could have been rebound onto this machine — unless
        // this goofi is the PUBLIC one, where rebinding wins an attacker nothing that opening the
        // URL does not already give them, and a DNS name is the only way anyone reaches it.
        return demo && host == Some(authority);
    };
    // Any IP literal that is not loopback has to BE the address this request was sent to.
    ip.is_loopback() || host == Some(authority)
}

/// The layer, applied once over the whole router.
pub(crate) async fn guard(
    axum::extract::State(mode): axum::extract::State<crate::Mode>,
    req: Request,
    next: Next,
) -> Response {
    // Scoped so the borrow of `req` ends before it is moved into the handler.
    let ok = {
        let h = req.headers();
        let get = |n: &str| h.get(n).and_then(|v: &axum::http::HeaderValue| v.to_str().ok());
        let fetch = Fetch {
            site: get("sec-fetch-site"),
            mode: get("sec-fetch-mode"),
            dest: get("sec-fetch-dest"),
        };
        allowed(
            req.method(),
            get(header::ORIGIN.as_str()),
            get(header::HOST.as_str()),
            fetch,
            mode.demo,
        )
    };
    if ok {
        return next.run(req).await;
    }
    (
        StatusCode::FORBIDDEN,
        "goofi refuses requests from a page it did not serve. This is a drive-by guard, not a \
         login: reach goofi at the address it printed at startup.\n",
    )
        .into_response()
}
