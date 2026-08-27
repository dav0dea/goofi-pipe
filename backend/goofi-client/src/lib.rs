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

/// One row of `session list`: the record, how it probed, and whether `GOOFI_SESSION` names it.
#[derive(Debug)]
pub struct Row {
    pub session: Session,
    pub probed: Probed,
    pub current: bool,
}

const PROBE: Duration = Duration::from_secs(2);
/// Generous: a `session load` provisions nodes, a `library refresh` restarts them.
const EXEC: Duration = Duration::from_secs(300);

/// Probe every recorded session, sweeping the DEFINITIVELY dead: a refused connection, an
/// answer that is not goofi's, or an id the file contradicts. A timeout — and every other local
/// failure, a reset or an exhausted fd table included — keeps its row: only a definitive wrong
/// answer may delete a record the server writes once in its life.
pub fn list() -> Vec<Row> {
    let current = std::env::var("GOOFI_SESSION").ok();
    let mut rows = Vec::new();
    for s in home::sessions() {
        let probed = match probe(&s.url) {
            Answered::Id(id) if id == s.id => Probed::Live,
            Answered::Id(_) | Answered::NotGoofi => {
                home::remove_session(&s.id);
                continue;
            }
            Answered::Silent => Probed::Unresponsive,
        };
        rows.push(Row { current: current.as_deref() == Some(&s.id), session: s, probed });
    }
    rows
}

/// The server this command drives: `GOOFI_SESSION` names one; unset, exactly one candidate is
/// unambiguous. Anything else is refused by naming what there is.
pub fn resolve_target() -> Result<Session, String> {
    let rows = list();
    if let Ok(id) = std::env::var("GOOFI_SESSION") {
        return rows
            .into_iter()
            .find(|r| r.session.id == id)
            .map(|r| r.session)
            .ok_or_else(|| format!("GOOFI_SESSION={id} names no running goofi — `goofi session list` shows them"));
    }
    match rows.len() {
        0 => Err("no running goofi — start one with `goofi`".into()),
        1 => Ok(rows.into_iter().next().unwrap().session),
        _ => {
            let named: Vec<String> =
                rows.iter().map(|r| format!("{} ({})", r.session.id, r.session.url)).collect();
            Err(format!(
                "several goofis are running — set GOOFI_SESSION to one of: {}",
                named.join(", ")
            ))
        }
    }
}

/// One `/exec` entry: the op's JSON and the server-rendered text.
#[derive(Debug)]
pub struct Entry {
    pub result: Value,
    pub text: String,
}

/// Send `lines` to `url`'s `/exec`. `actor` names the undo stack when the caller has one —
/// absent, the server's own `"default"` stands. One line executes directly, several are one
/// batch; a refusal is the server's own message.
pub fn exec(url: &str, lines: &[String], actor: Option<&str>) -> Result<Vec<Entry>, String> {
    let mut body = json!({ "commands": lines });
    if let Some(actor) = actor {
        body["actor"] = json!(actor);
    }
    let (status, reply) = http_post(url, "/exec", &body.to_string(), EXEC)
        .map_err(|e| format!("{url} did not answer: {e}"))?;
    let reply: Value =
        serde_json::from_str(&reply).map_err(|_| format!("{url} is not a goofi /exec door"))?;
    match (status, reply["results"].as_array()) {
        (200, Some(entries)) => Ok(entries
            .iter()
            .map(|e| Entry {
                result: e["result"].clone(),
                text: e["text"].as_str().unwrap_or_default().to_string(),
            })
            .collect()),
        (200, None) => Err(format!("{url} is not a goofi /exec door")),
        _ => Err(reply["error"].as_str().unwrap_or("refused").to_string()),
    }
}

/// What `goofi` writes for one entry: the decoded NPY when the result carries one — bytes for a
/// pipe — else the rendered text and a newline.
pub fn rendered(entry: &Entry) -> Vec<u8> {
    use base64::Engine;
    if let Some(b64) = entry.result.get("npy_b64").and_then(Value::as_str) {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
            return bytes;
        }
    }
    let mut out = entry.text.clone().into_bytes();
    out.push(b'\n');
    out
}

/// What a probed url said about itself.
enum Answered {
    /// A goofi answered `session status` with this instance id.
    Id(String),
    /// Something DEFINITIVELY not this record's goofi answered — another program on the port, a
    /// refused connection.
    NotGoofi,
    /// Nothing conclusive: a timeout, or a local failure that proves nothing about the server.
    Silent,
}

/// Ask `url` who it is: `session status` through the same `/exec` door every command uses.
fn probe(url: &str) -> Answered {
    let body = json!({ "commands": ["session status"] }).to_string();
    match http_post(url, "/exec", &body, PROBE) {
        Ok((200, reply)) => serde_json::from_str::<Value>(&reply)
            .ok()
            .and_then(|v| v["results"][0]["result"]["instance_id"].as_str().map(str::to_string))
            .map(Answered::Id)
            .unwrap_or(Answered::NotGoofi),
        Ok(_) => Answered::NotGoofi,
        Err(HttpErr::Refused) => Answered::NotGoofi,
        Err(_) => Answered::Silent,
    }
}

enum HttpErr {
    Refused,
    Timeout,
    Other(String),
}

impl std::fmt::Display for HttpErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpErr::Refused => write!(f, "refused the connection"),
            HttpErr::Timeout => write!(f, "timed out"),
            HttpErr::Other(e) => write!(f, "{e}"),
        }
    }
}

/// A minimal HTTP/1.1 POST over one blocking loopback socket — no TLS, no pooling, one answer.
fn http_post(url: &str, path: &str, body: &str, timeout: Duration) -> Result<(u16, String), HttpErr> {
    let host = url.strip_prefix("http://").unwrap_or(url);
    let addr = host
        .parse::<std::net::SocketAddr>()
        .map_err(|_| HttpErr::Other(format!("`{url}` is not `http://ip:port`")))?;
    // The CONNECT is always short: a listener answers a SYN at once or not at all, and only the
    // read may lawfully be slow (a `session load` provisions nodes).
    let mut s = TcpStream::connect_timeout(&addr, PROBE).map_err(io_err)?;
    s.set_read_timeout(Some(timeout)).map_err(io_err)?;
    s.set_write_timeout(Some(timeout)).map_err(io_err)?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).map_err(io_err)?;
    s.write_all(body.as_bytes()).map_err(io_err)?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(io_err)?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or(HttpErr::Other("a malformed HTTP reply".into()))?;
    let status = String::from_utf8_lossy(&raw[..split])
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or(HttpErr::Other("a malformed HTTP status line".into()))?;
    Ok((status, String::from_utf8_lossy(&raw[split + 4..]).into_owned()))
}

fn io_err(e: std::io::Error) -> HttpErr {
    match e.kind() {
        std::io::ErrorKind::ConnectionRefused => HttpErr::Refused,
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => HttpErr::Timeout,
        _ => HttpErr::Other(e.to_string()),
    }
}
