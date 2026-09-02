//! The bundle's binary, loaded once per process and never unloaded, and its factory — asked for
//! at every use, because a factory is a reference of its own and never crosses a thread.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use vst3::Steinberg::*;
use vst3::{ComPtr, Interface};

use super::host::cstr;

pub struct Module {
    library: &'static libloading::Library,
}

impl Module {
    pub fn open(binary: &Path) -> Result<Module, String> {
        static OPENED: OnceLock<Mutex<HashMap<PathBuf, &'static libloading::Library>>> = OnceLock::new();
        let mut opened = OPENED.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        if let Some(library) = opened.get(binary) {
            return Ok(Module { library });
        }
        let name = binary.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let library = load(binary).map_err(|e| format!("{name}: {e}"))?;
        opened.insert(binary.to_path_buf(), library);
        Ok(Module { library })
    }

    pub fn factory(&self) -> Result<Factory, String> {
        let get: libloading::Symbol<unsafe extern "system" fn() -> *mut IPluginFactory> =
            unsafe { self.library.get(b"GetPluginFactory\0") }.map_err(|e| format!("no `GetPluginFactory`: {e}"))?;
        unsafe { ComPtr::from_raw(get()) }.map(Factory).ok_or_else(|| "`GetPluginFactory` answered null".into())
    }
}

#[cfg(unix)]
fn load(path: &Path) -> Result<&'static libloading::Library, String> {
    use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
    // NOW, as goofi-build loads a node: the first call into a plugin must not run the resolver.
    let library = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map_err(|e| format!("could not load: {e}"))?;
    let handle = library.into_raw();
    let library: &'static libloading::Library = Box::leak(Box::new(unsafe { Library::from_raw(handle) }.into()));
    let (entry, argument) = if cfg!(target_os = "macos") { (&b"bundleEntry\0"[..], std::ptr::null_mut()) } else { (&b"ModuleEntry\0"[..], handle) };
    enter(library, entry, argument)?;
    Ok(library)
}

#[cfg(windows)]
fn load(path: &Path) -> Result<&'static libloading::Library, String> {
    let library = unsafe { libloading::Library::new(path) }.map_err(|e| format!("could not load: {e}"))?;
    let library: &'static libloading::Library = Box::leak(Box::new(library));
    if let Ok(init) = unsafe { library.get::<unsafe extern "system" fn() -> bool>(b"InitDll\0") } {
        if !unsafe { init() } {
            return Err("`InitDll` refused".into());
        }
    }
    Ok(library)
}

#[cfg(unix)]
fn enter(library: &libloading::Library, symbol: &[u8], argument: *mut c_void) -> Result<(), String> {
    if let Ok(entry) = unsafe { library.get::<unsafe extern "system" fn(*mut c_void) -> bool>(symbol) } {
        if !unsafe { entry(argument) } {
            return Err(format!("`{}` refused", String::from_utf8_lossy(&symbol[..symbol.len() - 1])));
        }
    }
    Ok(())
}

pub struct Factory(ComPtr<IPluginFactory>);

impl Factory {
    pub fn vendor(&self) -> String {
        let mut info: PFactoryInfo = unsafe { std::mem::zeroed() };
        unsafe { self.0.getFactoryInfo(&mut info) };
        cstr(&info.vendor)
    }

    /// Every "Audio Module Class": its cid and its name.
    pub fn audio_classes(&self) -> Vec<(TUID, String)> {
        (0..unsafe { self.0.countClasses() })
            .filter_map(|i| {
                let mut info: PClassInfo = unsafe { std::mem::zeroed() };
                let listed = unsafe { self.0.getClassInfo(i, &mut info) } == kResultOk;
                (listed && cstr(&info.category) == "Audio Module Class").then(|| (info.cid, cstr(&info.name)))
            })
            .collect()
    }

    pub fn create<I: Interface>(&self, cid: &TUID) -> Result<ComPtr<I>, String> {
        let mut obj: *mut c_void = std::ptr::null_mut();
        let result = unsafe { self.0.createInstance(cid.as_ptr(), I::IID.as_ptr() as FIDString, &mut obj) };
        unsafe { ComPtr::from_raw(obj as *mut I) }
            .filter(|_| result == kResultOk)
            .ok_or_else(|| format!("createInstance answered {result}"))
    }
}
