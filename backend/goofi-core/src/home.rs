//! The `$GOOFI_HOME/.goofi/` folder: path resolution, creation, and the stale-file sweep.
//! `GOOFI_HOME` is read PER CALL, so a spawned process is scoped by its environment alone.

use std::path::PathBuf;

/// One running server, as its session file records it: the id the probe verifies, and the HTTP
/// base every route hangs off.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: String,
    pub url: String,
}

/// The `.goofi` folder itself.
pub fn dir() -> PathBuf {
    std::env::var_os("GOOFI_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(std::env::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".goofi")
}

fn sessions_dir() -> PathBuf {
    dir().join("sessions")
}

fn session_file(id: &str) -> PathBuf {
    sessions_dir().join(format!("{id}.json"))
}

/// Record a running server. Written BESIDE the folder and renamed in, so a concurrent reader
/// sees a whole file or none — a torn read would be swept as stale, silently unlisting a live
/// server. A failure is said once and served through: recording is not what serving needs.
pub fn write_session(id: &str, url: &str) {
    let s = Session { id: id.to_string(), url: url.to_string() };
    let _ = std::fs::create_dir_all(sessions_dir());
    let tmp = dir().join(format!("{id}.json.part"));
    let written = std::fs::write(&tmp, serde_json::to_vec_pretty(&s).expect("two strings"))
        .and_then(|()| std::fs::rename(&tmp, session_file(id)));
    if let Err(e) = written {
        eprintln!("  not recorded in {}: {e}", dir().display());
    }
}

/// Remove a server's own record — the exit path, and the SWEEP for a file whose id the probe
/// contradicted or whose url refused the connection.
pub fn remove_session(id: &str) {
    let _ = std::fs::remove_file(session_file(id));
}

/// Every recorded session, unprobed. A file that does not parse is stale by construction and is
/// swept here; aliveness and identity are the PROBE's questions, answered live by each url.
pub fn sessions() -> Vec<Session> {
    let Ok(entries) = std::fs::read_dir(sessions_dir()) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let parsed = std::fs::read(entry.path())
            .ok()
            .and_then(|b| serde_json::from_slice::<Session>(&b).ok());
        match parsed {
            Some(s) => out.push(s),
            None => {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// One launchable agent: a display name and the bash command line that starts it.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Agent {
    pub name: String,
    pub command: String,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Config {
    #[serde(default)]
    agents: Vec<Agent>,
}

/// The compiled-in TEMPLATE — written once by the serve path when no config exists, so the FILE
/// stays the one owner and a test process never writes into a real home.
pub const DEFAULT_CONFIG: &str = "# goofi — the agents the app can launch, each a bash command line.\n\
    [[agents]]\nname = \"claude\"\ncommand = \"claude\"\n\n\
    [[agents]]\nname = \"codex\"\ncommand = \"codex\"\n\n\
    [[agents]]\nname = \"opencode\"\ncommand = \"opencode\"\n";

pub fn config_file() -> PathBuf {
    dir().join("config.toml")
}

/// Seed the default config, absent-only — the serve path calls this once at start.
pub fn seed_config() {
    let at = config_file();
    if !at.exists() {
        let _ = std::fs::create_dir_all(dir());
        let _ = std::fs::write(at, DEFAULT_CONFIG);
    }
}

/// The launchable agents: the config file's list. Absent reads as the default; a config that
/// does not parse degrades to the default and answers WHY — it never stops the app.
pub fn agents() -> (Vec<Agent>, Option<String>) {
    let default = || toml::from_str::<Config>(DEFAULT_CONFIG).expect("the template parses").agents;
    match std::fs::read_to_string(config_file()) {
        Err(_) => (default(), None),
        Ok(text) => match toml::from_str::<Config>(&text) {
            Ok(c) => (c.agents, None),
            Err(e) => {
                (default(), Some(format!("{} does not parse: {e}", config_file().display())))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_round_trips_and_a_malformed_file_is_swept_on_read() {
        let tmp = std::env::temp_dir().join(format!("goofi-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp); // a crashed run under a recycled pid
        // The ONE test in this crate touching GOOFI_HOME, so the process-global env is not raced.
        std::env::set_var("GOOFI_HOME", &tmp);
        write_session("abc", "http://127.0.0.1:9999");
        std::fs::write(dir().join("sessions/broken.json"), b"{").unwrap();
        assert_eq!(sessions(), vec![Session { id: "abc".into(), url: "http://127.0.0.1:9999".into() }]);
        assert!(!dir().join("sessions/broken.json").exists(), "the malformed file was swept");
        assert!(dir().join("sessions/abc.json").exists(), "a valid file SURVIVES the read");
        remove_session("abc");
        assert_eq!(sessions(), vec![]);

        // The config: reading never writes; the seed writes once, absent-only; a config that
        // does not parse degrades to the default and says why.
        let (list, warn) = agents();
        assert!(list.iter().any(|a| a.name == "claude") && warn.is_none(), "{list:?}");
        assert!(!config_file().exists(), "a READ must not write into a real home");
        seed_config();
        let seeded = std::fs::read_to_string(config_file()).unwrap();
        std::fs::write(config_file(), format!("{seeded}# mine\n")).unwrap();
        seed_config();
        assert!(std::fs::read_to_string(config_file()).unwrap().contains("# mine"),
                "the seed is absent-only — the FILE is the owner");
        std::fs::write(config_file(), "agents = [[[").unwrap();
        let (list, warn) = agents();
        assert!(list.iter().any(|a| a.name == "codex"), "malformed degrades to the default");
        assert!(warn.is_some_and(|w| w.contains("parse")), "…and says why");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
