//! A plugin's own editor, in a native window on the machine the SERVER runs on.
//!
//! Not in the browser, and it never can be: a VST3 editor is an OS window handed a parent handle,
//! so the one place it can appear is the desktop goofi itself is running on. Opening one from a
//! phone puts a window on the server's screen, and under `--headless` there is no screen to put it
//! on. That is why goofi's own parameter UI is the general answer and this is the local luxury.
//!
//! The editor owns its OWN component and controller pair, built from the same binary and seeded
//! with the running plugin's state. It is not the audio thread's instance and does not touch it:
//! the SDK already separates the two halves, and the edits a user makes travel back as parameter
//! values through the same door a patch cable uses. That is what keeps every COM call here on one
//! thread and out of `process`.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use vst3::Steinberg::Vst::*;
use vst3::Steinberg::*;
use vst3::{Class, ComPtr, ComWrapper};

use super::node::Derived;
use super::{host, module, ok};

/// What the editor thread is told to do. One thing, so far: the loop that would carry a knob's
/// value back into the patch is not built, and until it is there is nothing else to say.
enum Ask {
    Close,
}

/// How long the OPENING op will wait for the window before it returns and lets the server carry on.
/// A plugin whose `attached` blocks (BABY Audio's does) must never hold goofi's graph lock past
/// this — a slow or wedged editor thread is the editor's problem, not the whole app's.
const READY_WAIT: std::time::Duration = std::time::Duration::from_millis(1200);

/// One open editor: the thread it runs on, and the door to it. Dropping it closes the window.
pub struct Editor {
    ask: Sender<Ask>,
    /// The window's own handle, once it came up. `None` when the thread had not answered inside
    /// [`READY_WAIT`] — the window may still appear late, but goofi did not wait for it.
    wake: Option<Wake>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Editor {
    /// Show `class`'s editor. Returns as soon as the window is up OR [`READY_WAIT`] elapses,
    /// whichever is first — never blocking the caller (which holds the graph lock) on a plugin
    /// whose attach does not return promptly.
    pub fn open(class: Arc<Derived>, state: Vec<u8>, title: String) -> Result<Editor, String> {
        let (ask_tx, ask_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("vst3-editor".into())
            .spawn(move || run(class, state, title, ask_rx, ready_tx))
            .map_err(|e| format!("could not start the editor thread: {e}"))?;
        match ready_rx.recv_timeout(READY_WAIT) {
            Ok(Ok(wake)) => Ok(Editor { ask: ask_tx, wake: Some(wake), thread: Some(thread) }),
            // A plugin's own refusal ("no editor") is worth reporting; a slow one is not a failure.
            Ok(Err(e)) => Err(e),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                Ok(Editor { ask: ask_tx, wake: None, thread: Some(thread) })
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err("the editor thread stopped before it opened".into())
            }
        }
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        let _ = self.ask.send(Ask::Close);
        // Only a window that actually came up is worth waiting on. One whose thread never answered
        // is DETACHED, not joined: it may be wedged inside the plugin's attach, and joining it would
        // move that wedge onto whoever removed the node.
        match self.wake.take() {
            Some(wake) => {
                wake.post();
                if let Some(t) = self.thread.take() {
                    let _ = t.join();
                }
            }
            None => {
                self.thread.take();
            }
        }
    }
}

/// The host object a view edits through. A plugin REQUIRES one before it will draw, so this exists
/// to answer; carrying the value back into the patch is the half not built yet.
struct Handler;

impl Class for Handler {
    type Interfaces = (IComponentHandler,);
}

impl IComponentHandlerTrait for Handler {
    unsafe fn beginEdit(&self, _id: ParamID) -> tresult {
        kResultOk
    }

    unsafe fn performEdit(&self, _id: ParamID, _valueNormalized: ParamValue) -> tresult {
        kResultOk
    }

    unsafe fn endEdit(&self, _id: ParamID) -> tresult {
        kResultOk
    }

    /// A plugin asking to be restarted. goofi's manifest is fixed at scan time, so the shape it
    /// wants to change cannot change here — accepted so the plugin does not treat it as an error.
    unsafe fn restartComponent(&self, _flags: int32) -> tresult {
        kResultOk
    }
}

#[cfg(windows)]
pub use win::Wake;
#[cfg(windows)]
use win::{host_window, pump};

#[cfg(not(windows))]
pub use other::Wake;
#[cfg(not(windows))]
use other::{host_window, pump};

/// The editor thread: everything COM it touches is created, used and dropped here.
fn run(
    class: Arc<Derived>,
    state: Vec<u8>,
    title: String,
    ask: Receiver<Ask>,
    ready: Sender<Result<Wake, String>>,
) {
    match build(&class, state, &title) {
        Err(e) => {
            let _ = ready.send(Err(e));
        }
        Ok(built) => {
            let Built { view, window, controller, component, _handler, _frame } = built;
            match window.wake() {
                Err(e) => {
                    let _ = ready.send(Err(e));
                }
                Ok(wake) => {
                    let _ = ready.send(Ok(wake));
                    pump(&window, &ask);
                }
            }
            unsafe {
                view.removed();
                drop(view);
                window.close();
                controller.terminate();
                component.terminate();
            }
        }
    }
}

/// The host side of a view's window: the plugin asks HERE to be resized, and it asks during
/// `attached`. A host that never called `setFrame` leaves that pointer null, and a plugin that does
/// not guard it dereferences null inside the attach — which is how opening any editor aborted the
/// whole process.
struct Frame {
    /// The window to resize. An isize because the frame is handed across the COM boundary and an
    /// HWND is not `Send`; only the editor's own thread ever calls back into it.
    hwnd: isize,
}

impl Class for Frame {
    type Interfaces = (IPlugFrame,);
}

impl IPlugFrameTrait for Frame {
    unsafe fn resizeView(&self, view: *mut IPlugView, newSize: *mut ViewRect) -> tresult {
        if newSize.is_null() {
            return kInvalidArgument;
        }
        host_window::resize(self.hwnd, &*newSize);
        // Told back, because the plugin waits for the frame to have done it before it redraws.
        if let Some(v) = ComPtr::from_raw(view) {
            std::mem::forget(v.clone());
            v.onSize(newSize);
        }
        kResultOk
    }
}

/// What one open editor holds, in the order it must be torn down.
struct Built {
    view: ComPtr<IPlugView>,
    window: host_window::Window,
    controller: ComPtr<IEditController>,
    component: ComPtr<IComponent>,
    /// Kept alive for as long as the view can call them.
    _handler: ComWrapper<Handler>,
    _frame: ComWrapper<Frame>,
}

fn build(class: &Derived, state: Vec<u8>, title: &str) -> Result<Built, String> {
    let factory = module::factory(&class.binary)?;
    let component: ComPtr<IComponent> = factory.create(&class.cid)?;
    let context = ComWrapper::new(host::Host).to_com_ptr::<FUnknown>().expect("a host is an FUnknown");
    unsafe {
        ok(component.initialize(context.as_ptr()), "initialize")?;
        // The same introduction the scanner makes, and for the same reason: a controller that
        // cannot reach its processor has no parameters to draw.
        let controller = controller_of(&factory, &component, &context)?;
        let handler = ComWrapper::new(Handler);
        if let Some(h) = handler.to_com_ptr::<IComponentHandler>() {
            controller.setComponentHandler(h.as_ptr());
        }
        // Seeded with what the RUNNING plugin holds, so the window opens on the patch's sound
        // rather than on the plugin's factory default.
        if !state.is_empty() {
            let seed = ComWrapper::new(host::Stream::of(&state));
            if let Some(s) = seed.to_com_ptr::<IBStream>() {
                controller.setComponentState(s.as_ptr());
            }
        }
        let view = controller.createView(ViewType::kEditor);
        let view = ComPtr::from_raw(view).ok_or("the plugin offers no editor")?;
        if view.isPlatformTypeSupported(host_window::PLATFORM) != kResultOk {
            return Err(format!("the plugin's editor does not do {}", host_window::PLATFORM_NAME));
        }
        let mut rect: ViewRect = std::mem::zeroed();
        view.getSize(&mut rect);
        let window = host_window::Window::open(title, &rect)?;
        // BEFORE the attach, which is when the plugin first asks to be resized.
        let frame = ComWrapper::new(Frame { hwnd: window.raw() });
        if let Some(f) = frame.to_com_ptr::<IPlugFrame>() {
            view.setFrame(f.as_ptr());
        }
        ok(view.attached(window.handle(), host_window::PLATFORM), "attach the editor")?;
        Ok(Built { view, window, controller, component, _handler: handler, _frame: frame })
    }
}

/// The controller for this component, connected to it — one object or two, as the plugin chooses.
unsafe fn controller_of(
    factory: &module::Factory,
    component: &ComPtr<IComponent>,
    context: &ComPtr<FUnknown>,
) -> Result<ComPtr<IEditController>, String> {
    if let Some(own) = component.cast::<IEditController>() {
        return Ok(own);
    }
    let mut ccid: TUID = [0; 16];
    ok(component.getControllerClassId(&mut ccid), "ask for the controller's class")?;
    let controller: ComPtr<IEditController> = factory.create(&ccid)?;
    ok(controller.initialize(context.as_ptr()), "initialize the controller")?;
    if let (Some(cp), Some(ccp)) = (component.cast::<IConnectionPoint>(), controller.cast::<IConnectionPoint>()) {
        cp.connect(ccp.as_ptr());
        ccp.connect(cp.as_ptr());
    }
    Ok(controller)
}

#[cfg(windows)]
mod win {
    //! The window an editor is attached to, and the loop that keeps it alive. A VST3 editor wants a
    //! real HWND on a thread that pumps messages, which is why the editor gets a thread of its own
    //! rather than a corner of the server's runtime.

    use std::sync::mpsc::Receiver;

    use vst3::Steinberg::*;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging as w32;

    use super::Ask;

    pub const PLATFORM: FIDString = kPlatformTypeHWND;
    pub const PLATFORM_NAME: &str = "HWND";

    /// The message the editor thread wakes on when goofi has something to say. Above WM_USER, so
    /// it cannot collide with anything the plugin's own window posts.
    const WM_GOOFI: u32 = w32::WM_USER + 1;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A handle to post to the editor thread from any other. An HWND is not `Send` by convention
    /// because most of what you can do with one must happen on its own thread — `PostMessageW` is
    /// the documented exception, and posting is all this does.
    pub struct Wake(isize);

    unsafe impl Send for Wake {}

    impl Wake {
        pub fn post(&self) {
            unsafe { w32::PostMessageW(self.0 as HWND, WM_GOOFI, 0, 0) };
        }
    }

    pub mod host_window {
        pub use super::{resize, Window, PLATFORM, PLATFORM_NAME};
    }

    pub struct Window {
        hwnd: HWND,
    }

    impl Window {
        pub fn open(title: &str, rect: &ViewRect) -> Result<Window, String> {
            let class = wide("goofiVst3Editor");
            let name = wide(title);
            unsafe {
                let instance = GetModuleHandleW(std::ptr::null());
                // Registering twice is not an error worth reading: the second editor finds the
                // class already there, which is exactly what we want.
                let mut wc: w32::WNDCLASSW = std::mem::zeroed();
                wc.lpfnWndProc = Some(proc);
                wc.hInstance = instance;
                wc.lpszClassName = class.as_ptr();
                wc.hCursor = w32::LoadCursorW(std::ptr::null_mut(), w32::IDC_ARROW);
                w32::RegisterClassW(&wc);
                // The rect the plugin asked for is its CLIENT area, so the frame is added on top of
                // it — a window sized to the raw numbers crops the editor by its own border.
                let mut r = windows_sys::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: (rect.right - rect.left).max(320),
                    bottom: (rect.bottom - rect.top).max(240),
                };
                let style = w32::WS_OVERLAPPEDWINDOW & !w32::WS_MAXIMIZEBOX;
                w32::AdjustWindowRect(&mut r, style, 0);
                let hwnd = w32::CreateWindowExW(
                    0,
                    class.as_ptr(),
                    name.as_ptr(),
                    style,
                    w32::CW_USEDEFAULT,
                    w32::CW_USEDEFAULT,
                    r.right - r.left,
                    r.bottom - r.top,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    instance,
                    std::ptr::null(),
                );
                if hwnd.is_null() {
                    return Err("could not open a window for the editor".into());
                }
                w32::ShowWindow(hwnd, w32::SW_SHOWNORMAL);
                // Raised, or it opens behind the browser the user asked from and reads as nothing
                // having happened.
                w32::SetForegroundWindow(hwnd);
                Ok(Window { hwnd })
            }
        }

        pub fn handle(&self) -> *mut std::ffi::c_void {
            self.hwnd
        }

        pub fn raw(&self) -> isize {
            self.hwnd as isize
        }

        pub fn wake(&self) -> Result<Wake, String> {
            Ok(Wake(self.hwnd as isize))
        }

        pub fn close(self) {
            unsafe { w32::DestroyWindow(self.hwnd) };
        }
    }

    /// Resize the frame around a client area the plugin chose.
    pub fn resize(hwnd: isize, to: &ViewRect) {
        unsafe {
            let mut r = windows_sys::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: to.right - to.left,
                bottom: to.bottom - to.top,
            };
            let style = w32::WS_OVERLAPPEDWINDOW & !w32::WS_MAXIMIZEBOX;
            w32::AdjustWindowRect(&mut r, style, 0);
            w32::SetWindowPos(hwnd as HWND, std::ptr::null_mut(), 0, 0, r.right - r.left, r.bottom - r.top, w32::SWP_NOMOVE | w32::SWP_NOZORDER);
        }
    }

    unsafe extern "system" fn proc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match msg {
            // The user closing the window is the user closing the editor; goofi hears about it
            // when the thread ends, and the node keeps playing either way.
            w32::WM_CLOSE => {
                w32::PostQuitMessage(0);
                0
            }
            _ => w32::DefWindowProcW(hwnd, msg, wp, lp),
        }
    }

    /// Run until the window closes or goofi says to. `GetMessageW` blocks, so this thread costs
    /// nothing while the user is not touching the editor.
    pub fn pump(window: &Window, ask: &Receiver<Ask>) {
        unsafe {
            let mut msg: w32::MSG = std::mem::zeroed();
            while w32::GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                if msg.message == WM_GOOFI {
                    for a in ask.try_iter() {
                        match a {
                            Ask::Close => w32::PostQuitMessage(0),
                        }
                    }
                }
                w32::TranslateMessage(&msg);
                w32::DispatchMessageW(&msg);
            }
            let _ = window;
        }
    }
}

#[cfg(not(windows))]
mod other {
    //! Where goofi has no editor host yet. The refusal is the whole implementation: an editor is
    //! a platform's own window, and Linux (X11/Wayland) and macOS (NSView) each need their own.

    use std::sync::mpsc::Receiver;

    use vst3::Steinberg::*;

    use super::Ask;

    #[cfg(target_os = "macos")]
    pub const PLATFORM: FIDString = kPlatformTypeNSView;
    #[cfg(target_os = "macos")]
    pub const PLATFORM_NAME: &str = "NSView";
    #[cfg(not(target_os = "macos"))]
    pub const PLATFORM: FIDString = kPlatformTypeX11EmbedWindowID;
    #[cfg(not(target_os = "macos"))]
    pub const PLATFORM_NAME: &str = "X11EmbedWindowID";

    pub struct Wake;

    impl Wake {
        pub fn post(&self) {}
    }

    pub mod host_window {
        pub use super::{resize, Window, PLATFORM, PLATFORM_NAME};
    }

    pub struct Window;

    impl Window {
        pub fn open(_title: &str, _rect: &ViewRect) -> Result<Window, String> {
            Err(format!("goofi hosts plugin editors on Windows only; this is {}", std::env::consts::OS))
        }

        pub fn handle(&self) -> *mut std::ffi::c_void {
            std::ptr::null_mut()
        }

        pub fn raw(&self) -> isize {
            0
        }

        pub fn wake(&self) -> Result<Wake, String> {
            Ok(Wake)
        }

        pub fn close(self) {}
    }

    pub fn pump(_window: &Window, _ask: &Receiver<Ask>) {}

    pub fn resize(_hwnd: isize, _to: &ViewRect) {}
}
