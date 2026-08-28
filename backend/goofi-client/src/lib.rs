//! The client half of the one interface: resolve WHICH server, send command lines to its
//! `/exec`, print what comes back. Zero op knowledge lives here — parsing, help and rendering
//! are the server's, shared verbatim with the MCP tool.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use goofi_core::home::{self, Session};
use serde_json::{json, Value};

/// How a recorded session answered the identity probe.
#[derive(Debug, PartialEq)]
pub enum Probed {
    /// It answered `session status` with the id its file claims.
    Live,
    /// It did not answer in time — a busy server, kept tentatively.
    Unresponsive,
}

const PROBE: Duration = Duration::from_secs(2);
/// Generous: a `session load` provisions nodes, a `library refresh` restarts them.
const EXEC: Duration = Duration::from_secs(300);

/// Every recorded session and how it probed, sweeping the definitively dead as it goes.
pub fn list() -> Vec<(Session, Probed)> {
    home::sessions()
        .into_iter()
        .filter_map(|s| match probe(&s) {
            Some(p) => Some((s, p)),
            None => {
                home::remove_session(&s.id);
                None
            }
        })
        .collect()
}

/// The server this command drives: `GOOFI_SESSION` names one; unset, exactly one candidate is
/// unambiguous. Anything else is refused by naming what there is.
pub fn resolve_target() -> Result<Session, String> {
    let mut rows = list();
    if let Ok(id) = std::env::var("GOOFI_SESSION") {
        return rows
            .into_iter()
            .find(|(s, _)| s.id == id)
            .map(|(s, _)| s)
            .ok_or_else(|| format!("GOOFI_SESSION={id} names no running goofi — `goofi session list` shows them"));
    }
    match rows.len() {
        0 => Err("no running goofi — start one with `goofi`".into()),
        1 => Ok(rows.remove(0).0),
        _ => {
            let named: Vec<String> =
                rows.iter().map(|(s, _)| format!("{} ({})", s.id, s.url)).collect();
            Err(format!(
                "several goofis are running — set GOOFI_SESSION to one of: {}",
                named.join(", ")
            ))
        }
    }
}

/// Send `lines` to `url`'s `/exec`. `actor` names the undo stack when the caller has one —
/// absent, the server's own `"default"` stands. One line executes directly, several are one
/// batch; a refusal is the server's own message. Each entry is the wire's own `{result, text}`.
pub fn exec(url: &str, lines: &[String], actor: Option<&str>) -> Result<Vec<Value>, String> {
    let mut body = json!({ "commands": lines });
    if let Some(actor) = actor {
        body["actor"] = json!(actor);
    }
    let (status, reply) = http_post(url, "/exec", &body.to_string(), EXEC)
        .map_err(|e| format!("{url} did not answer: {e}"))?;
    let mut reply: Value =
        serde_json::from_str(&reply).map_err(|_| format!("{url} is not a goofi /exec door"))?;
    match (status, reply["results"].take()) {
        (200, Value::Array(entries)) => Ok(entries),
        (200, _) => Err(format!("{url} is not a goofi /exec door")),
        _ => Err(reply["error"].as_str().unwrap_or("refused").to_string()),
    }
}

/// What `goofi` writes for one entry: the decoded NPY when the result carries one — bytes for a
/// pipe — else the rendered text and a newline.
pub fn rendered(entry: &Value) -> Vec<u8> {
    use base64::Engine;
    if let Some(b64) = entry["result"]["npy_b64"].as_str() {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
            return bytes;
        }
    }
    let mut out = entry["text"].as_str().unwrap_or_default().as_bytes().to_vec();
    out.push(b'\n');
    out
}

/// Ask the recorded server who it is, through the same `/exec` door every command uses. `None` is
/// DEFINITIVE — nothing accepts on the address, something not goofi answered, or the id
/// contradicts the file — and only that sweeps a record the server writes once in its life. A
/// timeout, and every other post-connect failure, proves nothing and keeps the row.
fn probe(s: &Session) -> Option<Probed> {
    let body = json!({ "commands": ["session status"] }).to_string();
    match http_post(&s.url, "/exec", &body, PROBE) {
        Ok((200, reply)) => {
            let id = serde_json::from_str::<Value>(&reply)
                .ok()
                .and_then(|v| v["results"][0]["result"]["instance_id"].as_str().map(str::to_string));
            (id.as_deref() == Some(&s.id)).then_some(Probed::Live)
        }
        Ok(_) | Err(HttpErr::NoListener) => None,
        Err(_) => Some(Probed::Unresponsive),
    }
}

/// The one distinction a caller acts on: a connect nothing answered is DEFINITIVE, anything after
/// the connect proves nothing about the server.
enum HttpErr {
    NoListener,
    After(String),
}

impl std::fmt::Display for HttpErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpErr::NoListener => write!(f, "nothing is listening"),
            HttpErr::After(e) => write!(f, "{e}"),
        }
    }
}

/// A minimal HTTP/1.1 POST over one blocking loopback socket — no TLS, no pooling, one answer.
fn http_post(url: &str, path: &str, body: &str, timeout: Duration) -> Result<(u16, String), HttpErr> {
    let after = |e: std::io::Error| {
        HttpErr::After(match e.kind() {
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => "timed out".into(),
            _ => e.to_string(),
        })
    };
    let host = url.strip_prefix("http://").unwrap_or(url);
    let addr = host
        .parse::<std::net::SocketAddr>()
        .map_err(|_| HttpErr::After(format!("`{url}` is not `http://ip:port`")))?;
    // The CONNECT is always short: a listener answers a SYN at once or not at all, and only the
    // read may lawfully be slow (a `session load` provisions nodes). So a connect that fails at
    // all means the address holds nothing — decided by the STAGE, never by the error kind, because
    // Windows DROPS a SYN to a closed port where unix refuses it, and the kind then reads "timed
    // out" for the one state that is definitively dead.
    let mut s = TcpStream::connect_timeout(&addr, PROBE).map_err(|_| HttpErr::NoListener)?;
    s.set_read_timeout(Some(timeout)).map_err(after)?;
    s.set_write_timeout(Some(timeout)).map_err(after)?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(after)?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(after)?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or(HttpErr::After("a malformed HTTP reply".into()))?;
    let status = String::from_utf8_lossy(&raw[..split])
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or(HttpErr::After("a malformed HTTP status line".into()))?;
    Ok((status, String::from_utf8_lossy(&raw[split + 4..]).into_owned()))
}
