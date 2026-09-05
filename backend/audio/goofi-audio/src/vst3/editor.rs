//! A plugin's own editor: a view off the LIVE instance's controller, in a window on the window
//! thread, and every knob turned in it carried back into the document through the one param door.
//! Everything here runs on the window thread, and the table is that thread's own.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use goofi_node::Uid;
#[cfg(target_os = "linux")]
use vst3::Steinberg::Linux::{IEventHandlerTrait, ITimerHandlerTrait};
use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{Class, ComPtr, ComRef, ComWrapper};

use super::node::Derived;
use crate::control::Shared;
use crate::ui::{Host, Runloop, Window};

#[cfg(target_os = "linux")]
const PLATFORM: FIDString = kPlatformTypeX11EmbedWindowID;
#[cfg(windows)]
const PLATFORM: FIDString = kPlatformTypeHWND;
#[cfg(target_os = "macos")]
const PLATFORM: FIDString = kPlatformTypeNSView;

thread_local! {
    static PLUGINS: RefCell<HashMap<Uid, Running>> = RefCell::new(HashMap::new());
}

/// One running plugin as the window thread knows it: the controller a view is asked of, and the
/// window while one is open.
struct Running {
    controller: ComPtr<IEditController>,
    _class: Arc<Derived>,
    _handler: ComWrapper<Handler>,
    editor: Option<Editor>,
}

struct Editor {
    view: ComPtr<IPlugView>,
    window: Window,
    frame: ComWrapper<Frame>,
}

/// A plugin came up: its controller edits through here from now on, and a view can be asked of it.
pub(super) fn register(uid: Uid, controller: ComPtr<IEditController>, class: Arc<Derived>, shared: Arc<Shared>) {
    let handler = ComWrapper::new(Handler { uid, shared });
    if let Some(h) = handler.to_com_ptr::<IComponentHandler>() {
        unsafe { controller.setComponentHandler(h.as_ptr()) };
    }
    PLUGINS.with(|p| p.borrow_mut().insert(uid, Running { controller, _class: class, _handler: handler, editor: None }));
}

/// The plugin is going: its window first, then its place here — before its halves are torn down.
pub(super) fn unregister(host: &mut Host, uid: Uid) {
    if let Some(mut running) = PLUGINS.with(|p| p.borrow_mut().remove(&uid)) {
        if let Some(editor) = running.editor.take() {
            editor.close(host);
        }
        unsafe { running.controller.setComponentHandler(std::ptr::null_mut()) };
    }
}

/// Show or hide `uid`'s editor; whether that changed anything.
pub(crate) fn show(host: &mut Host, uid: Uid, title: &str, show: bool) -> Result<bool, String> {
    let found = PLUGINS.with(|p| p.borrow().get(&uid).map(|r| (r.controller.clone(), r.editor.is_some())));
    let (controller, open) = found.ok_or("the plugin is not running")?;
    if !show {
        let editor = PLUGINS.with(|p| p.borrow_mut().get_mut(&uid).and_then(|r| r.editor.take()));
        return Ok(editor.map(|e| e.close(host)).is_some());
    }
    if open {
        return Ok(false);
    }
    let editor = Editor::open(host, uid, &controller, title)?;
    PLUGINS.with(|p| {
        if let Some(r) = p.borrow_mut().get_mut(&uid) {
            r.editor = Some(editor);
        }
    });
    Ok(true)
}

/// The record's values, told to the controller so the window shows what the document holds.
pub(crate) fn sync(uid: Uid, values: Vec<(ParamID, f64)>) {
    PLUGINS.with(|p| {
        if let Some(r) = p.borrow().get(&uid) {
            for (id, v) in values {
                unsafe { r.controller.setParamNormalized(id, v) };
            }
        }
    });
}

impl Editor {
    fn open(host: &mut Host, uid: Uid, controller: &ComPtr<IEditController>, title: &str) -> Result<Editor, String> {
        let view = unsafe { ComPtr::from_raw(controller.createView(ViewType::kEditor)) }.ok_or("the plugin offers no editor")?;
        if unsafe { view.isPlatformTypeSupported(PLATFORM) } != kResultTrue {
            return Err("the plugin's editor does not draw into this platform's window".into());
        }
        let mut rect: ViewRect = unsafe { std::mem::zeroed() };
        unsafe { view.getSize(&mut rect) };
        let size = ((rect.right - rect.left).max(64) as u32, (rect.bottom - rect.top).max(64) as u32);
        let window = host.open_window(
            title,
            size,
            Box::new(move |host| {
                if let Some(editor) = PLUGINS.with(|p| p.borrow_mut().get_mut(&uid).and_then(|r| r.editor.take())) {
                    editor.close(host);
                }
            }),
        )?;
        // BEFORE the attach, which is when a plugin first asks to be resized.
        let frame = ComWrapper::new(Frame { window, runloop: host.runloop(), registered: RefCell::new(Vec::new()) });
        if let Some(f) = frame.to_com_ptr::<IPlugFrame>() {
            unsafe { view.setFrame(f.as_ptr()) };
        }
        let attached = unsafe { view.attached(window.parent, PLATFORM) };
        if attached != kResultOk {
            unsafe { view.setFrame(std::ptr::null_mut()) };
            host.close_window(window);
            return Err(format!("attaching the editor answered {attached}"));
        }
        Ok(Editor { view, window, frame })
    }

    fn close(self, host: &mut Host) {
        unsafe {
            self.view.removed();
            self.view.setFrame(std::ptr::null_mut());
        }
        host.close_window(self.window);
        drop(self.view);
        drop(self.frame);
    }
}

/// What a view edits through: the value goes to the engine's inbox, and to the document from there.
struct Handler {
    uid: Uid,
    shared: Arc<Shared>,
}

impl Class for Handler {
    type Interfaces = (IComponentHandler,);
}

impl IComponentHandlerTrait for Handler {
    unsafe fn beginEdit(&self, _id: ParamID) -> tresult {
        kResultOk
    }

    unsafe fn performEdit(&self, id: ParamID, value: ParamValue) -> tresult {
        self.shared.edits.lock().unwrap().push((self.uid, id, value));
        self.shared.waker.notify();
        kResultOk
    }

    unsafe fn endEdit(&self, _id: ParamID) -> tresult {
        kResultOk
    }

    /// A restart asks for a shape goofi fixed at scan time: accepted, so the plugin sees no error.
    unsafe fn restartComponent(&self, _flags: int32) -> tresult {
        kResultOk
    }
}

/// The host side of a view's window: the resize a plugin asks for, and on Linux the run loop it
/// registers its descriptors and timers with.
struct Frame {
    window: Window,
    runloop: Rc<RefCell<Runloop>>,
    /// What the view registered through this frame, removed with it where the view did not.
    registered: RefCell<Vec<usize>>,
}

#[cfg(target_os = "linux")]
impl Class for Frame {
    type Interfaces = (IPlugFrame, Linux::IRunLoop);
}

#[cfg(not(target_os = "linux"))]
impl Class for Frame {
    type Interfaces = (IPlugFrame,);
}

impl IPlugFrameTrait for Frame {
    unsafe fn resizeView(&self, view: *mut IPlugView, new_size: *mut ViewRect) -> tresult {
        if new_size.is_null() {
            return kInvalidArgument;
        }
        let r = &*new_size;
        self.window.request_resize(((r.right - r.left).max(1) as u32, (r.bottom - r.top).max(1) as u32));
        // Told back, because the plugin lays out for the size the frame granted.
        if let Some(v) = ComRef::from_raw(view) {
            v.onSize(new_size);
        }
        kResultOk
    }
}

#[cfg(target_os = "linux")]
impl Linux::IRunLoopTrait for Frame {
    unsafe fn registerEventHandler(&self, handler: *mut Linux::IEventHandler, fd: Linux::FileDescriptor) -> tresult {
        let Some(h) = ComRef::from_raw(handler).and_then(|h| h.cast::<Linux::IEventHandler>()) else { return kInvalidArgument };
        let key = handler as usize;
        self.registered.borrow_mut().push(key);
        self.runloop.borrow_mut().add_fd(key, fd, Box::new(move || {
            h.onFDIsSet(fd);
        }));
        kResultOk
    }

    unsafe fn unregisterEventHandler(&self, handler: *mut Linux::IEventHandler) -> tresult {
        let key = handler as usize;
        self.registered.borrow_mut().retain(|k| *k != key);
        match self.runloop.borrow_mut().remove_fd(key) {
            true => kResultOk,
            false => kResultFalse,
        }
    }

    unsafe fn registerTimer(&self, handler: *mut Linux::ITimerHandler, milliseconds: Linux::TimerInterval) -> tresult {
        let Some(h) = ComRef::from_raw(handler).and_then(|h| h.cast::<Linux::ITimerHandler>()) else { return kInvalidArgument };
        let key = handler as usize;
        self.registered.borrow_mut().push(key);
        self.runloop.borrow_mut().add_timer(key, Duration::from_millis(milliseconds.max(1)), Box::new(move || {
            h.onTimer();
        }));
        kResultOk
    }

    unsafe fn unregisterTimer(&self, handler: *mut Linux::ITimerHandler) -> tresult {
        let key = handler as usize;
        self.registered.borrow_mut().retain(|k| *k != key);
        match self.runloop.borrow_mut().remove_timer(key) {
            true => kResultOk,
            false => kResultFalse,
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let mut runloop = self.runloop.borrow_mut();
        for key in self.registered.borrow().iter() {
            runloop.remove_timer(*key);
            runloop.remove_fd(*key);
        }
    }
}
