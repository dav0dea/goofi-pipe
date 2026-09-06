//! Which cpal host a device comes from, and the name a patch stores for it.
//!
//! WASAPI is what Windows offers by default, and it publishes an interface as one stereo endpoint
//! per pair the driver chose to expose. A four-input card whose driver publishes only `Analogue
//! 1 + 2` IS only that pair to every WASAPI client, and no choice inside goofi reaches the rest of
//! it. ASIO is the driver's own multi-channel view — the same card answers as eighteen channels on
//! one device — and that is what it is here for.
//!
//! It is a build the user makes and never one that ships: the Steinberg SDK went
//! GPLv3-or-proprietary in 2025, so it cannot travel inside this binary — `roadmap/audio-engine.md`
//! carries that decision. `--features asio`, with `CPAL_ASIO_DIR` naming the unpacked SDK, is the
//! whole of the opt-in; without it every function here is the platform default it always was.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait};

/// What an ASIO device's name wears. The two hosts name one card differently but not always, and a
/// patch stores the NAME — so an unprefixed collision would open the other host's device without
/// saying it had.
pub(crate) const ASIO: &str = "ASIO: ";

/// The ASIO driver `name` asks for, or `None` for a name from the platform host.
///
/// ASIO loads ONE driver per process — the constraint is the SDK's, not cpal's, and asking for a
/// second answers `DriverAlreadyExists` rather than degrading. So a patch may name many devices
/// but only ever one ASIO driver, and [`crate::AudioEngine`] refuses the rest by this.
pub(crate) fn asio_driver(name: &str) -> Option<&str> {
    name.strip_prefix(ASIO)
}

/// Input or output, kept as a type because it picks the enumeration as well as wording a refusal.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Input,
    Output,
}

impl Kind {
    /// The word a refusal is worded with.
    fn word(self) -> &'static str {
        match self {
            Kind::Input => "input",
            Kind::Output => "output",
        }
    }

    /// A host's devices of this kind. A host that refuses to enumerate contributes none rather
    /// than failing the list: the other host's devices are still openable, and a name that is
    /// missing reports itself at open.
    fn all(self, host: &cpal::Host) -> Vec<cpal::Device> {
        match self {
            Kind::Input => host.input_devices().map(|d| d.collect()).unwrap_or_default(),
            Kind::Output => host.output_devices().map(|d| d.collect()).unwrap_or_default(),
        }
    }

    fn default_of(self, host: &cpal::Host) -> Option<cpal::Device> {
        match self {
            Kind::Input => host.default_input_device(),
            Kind::Output => host.default_output_device(),
        }
    }
}

/// The platform default: WASAPI on Windows, CoreAudio on macOS, ALSA or the sound server on Linux.
fn platform() -> cpal::Host {
    cpal::default_host()
}

/// The ASIO host, in a build that was given the SDK. `None` is a machine with no ASIO driver
/// installed, which is not an error — the platform host still answers everything.
#[cfg(feature = "asio")]
fn asio() -> Option<cpal::Host> {
    cpal::host_from_id(cpal::HostId::Asio).ok()
}

/// Without the feature there is no second host, and every name resolves as it did before.
#[cfg(not(feature = "asio"))]
fn asio() -> Option<cpal::Host> {
    None
}

/// Every ASIO device seen while its driver could still be loaded, by the name a patch stores.
///
/// Enumerating ASIO stops at the FIRST driver that is not the one already loaded — cpal returns
/// `None` there rather than spinning through the rest — so once any stream holds a driver the list
/// is empty, and even that driver's own device cannot be found again. An input opened after an
/// output would therefore fail with `no input device`, naming the very device that is playing.
///
/// A `Device` is a handle and not a session: it stays valid while its driver is loaded, and one
/// ASIO device serves input and output alike. So the handle seen before the stream opened is the
/// one to reuse, and this is where it is kept.
fn seen() -> &'static Mutex<HashMap<String, cpal::Device>> {
    static SEEN: OnceLock<Mutex<HashMap<String, cpal::Device>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashMap::new()))
}

fn name_of(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|d| d.name().to_string())
}

/// Every device goofi offers for `kind`, under the name a patch stores: the platform host's as the
/// OS gives it, ASIO's behind [`ASIO`]. Platform first, so an unprefixed name keeps meaning what it
/// meant before this module existed.
///
/// Enumerating ASIO briefly LOADS each driver to read its metadata, and a driver another stream
/// holds open cannot be loaded again under a different name — so this list is shorter while a
/// stream is running than it is while the engine is idle. That is the SDK's one-driver rule seen
/// from the other end, and nothing here can widen it.
pub(crate) fn named(kind: Kind) -> Vec<(String, cpal::Device)> {
    let mut out: Vec<(String, cpal::Device)> =
        kind.all(&platform()).into_iter().filter_map(|d| name_of(&d).map(|n| (n, d))).collect();
    if let Some(host) = asio() {
        let asio: Vec<(String, cpal::Device)> =
            kind.all(&host).into_iter().filter_map(|d| name_of(&d).map(|n| (format!("{ASIO}{n}"), d))).collect();
        if let Ok(mut seen) = seen().lock() {
            seen.extend(asio.iter().map(|(n, d)| (n.clone(), d.clone())));
        }
        out.extend(asio);
    }
    out
}

/// The device `name` names, `default` being the platform host's. The prefix is a choice of host
/// rather than decoration, so an `ASIO: ` name is resolved in the ASIO host alone.
pub(crate) fn device(kind: Kind, name: &str) -> Result<cpal::Device, String> {
    if name == crate::DEFAULT_DEVICE {
        return kind.default_of(&platform()).ok_or_else(|| format!("no default {} device", kind.word()));
    }
    if let Some((_, device)) = named(kind).into_iter().find(|(n, _)| n == name) {
        return Ok(device);
    }
    // Not in the list, and for an ASIO name that is the expected answer once a stream holds the
    // driver — see [`seen`]. A device remembered from before is the same device.
    if asio_driver(name).is_some() {
        if let Some(device) = seen().lock().ok().and_then(|s| s.get(name).cloned()) {
            return Ok(device);
        }
    }
    Err(format!("no {} device `{name}`", kind.word()))
}

#[cfg(test)]
mod tests {
    use super::{asio_driver, ASIO};

    /// The prefix is the whole of the host choice, so reading it back must be exact: a device
    /// merely CALLED something ASIO-ish is a platform device and stays one.
    #[test]
    fn a_driver_is_read_back_from_the_prefix_alone() {
        assert_eq!(asio_driver("ASIO: Focusrite USB ASIO"), Some("Focusrite USB ASIO"));
        assert_eq!(asio_driver("Analogue 1 + 2 (Focusrite Usb Audio)"), None);
        // Named for the standard, not prefixed with it: a WASAPI endpoint, and no driver claim.
        assert_eq!(asio_driver("Generic Low Latency ASIO Driver"), None);
        assert_eq!(asio_driver("default"), None);
    }

    #[test]
    fn the_prefix_round_trips_a_name() {
        let name = format!("{ASIO}Voicemeeter Virtual ASIO");
        assert_eq!(asio_driver(&name), Some("Voicemeeter Virtual ASIO"));
    }
}
