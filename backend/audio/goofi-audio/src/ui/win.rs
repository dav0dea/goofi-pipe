//! Win32: a window class of goofi's own, and the thread's message queue pumped for every window
//! on it — the plugin's child windows included, which is what a plugin's timers ride on.

use std::cell::RefCell;
use std::ffi::c_void;
use std::time::Instant;

use windows_sys::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent, INFINITE};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

use super::Pumped;

pub type Id = isize;

pub struct Platform {
    class: Vec<u16>,
    event: HANDLE,
}

/// An auto-reset event the pump waits on beside the queue; setting it is thread-safe.
pub struct Waker(isize);

unsafe impl Send for Waker {}
unsafe impl Sync for Waker {}

impl Waker {
    pub fn wake(&self) {
        unsafe { SetEvent(self.0 as HANDLE) };
    }
}

thread_local! {
    static CLOSED: RefCell<Vec<Id>> = const { RefCell::new(Vec::new()) };
}

const STYLE: u32 = WS_OVERLAPPEDWINDOW & !WS_MAXIMIZEBOX & !WS_THICKFRAME;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A close is REPORTED, never performed here: the window goes once the plugin's view is off it.
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if msg == WM_CLOSE {
        CLOSED.with(|c| c.borrow_mut().push(hwnd as Id));
        return 0;
    }
    DefWindowProcW(hwnd, msg, wp, lp)
}

/// The outer size for a client area the plugin asked for.
fn framed((w, h): (u32, u32)) -> (i32, i32) {
    let mut r = RECT { left: 0, top: 0, right: w.min(i32::MAX as u32) as i32, bottom: h.min(i32::MAX as u32) as i32 };
    unsafe { AdjustWindowRect(&mut r, STYLE, 0) };
    (r.right - r.left, r.bottom - r.top)
}

impl Platform {
    pub fn open() -> Result<Platform, String> {
        let class = wide("goofiWindow");
        unsafe {
            let mut wc: WNDCLASSW = std::mem::zeroed();
            wc.lpfnWndProc = Some(wndproc);
            wc.hInstance = GetModuleHandleW(std::ptr::null());
            wc.lpszClassName = class.as_ptr();
            wc.hCursor = LoadCursorW(std::ptr::null_mut(), IDC_ARROW);
            RegisterClassW(&wc);
            let event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            if event.is_null() {
                return Err("no wake event".into());
            }
            Ok(Platform { class, event })
        }
    }

    pub fn waker(&self) -> Waker {
        Waker(self.event as isize)
    }

    pub fn create(&mut self, title: &str, size: (u32, u32)) -> Result<(Id, *mut c_void), String> {
        let name = wide(title);
        let (w, h) = framed(size);
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                self.class.as_ptr(),
                name.as_ptr(),
                STYLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                w,
                h,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                GetModuleHandleW(std::ptr::null()),
                std::ptr::null(),
            )
        };
        if hwnd.is_null() {
            return Err("could not open a window".into());
        }
        unsafe {
            ShowWindow(hwnd, SW_SHOWNORMAL);
            SetForegroundWindow(hwnd);
        }
        Ok((hwnd as Id, hwnd))
    }

    pub fn resize(&mut self, id: Id, size: (u32, u32)) {
        let (w, h) = framed(size);
        unsafe { SetWindowPos(id as HWND, std::ptr::null_mut(), 0, 0, w, h, SWP_NOMOVE | SWP_NOZORDER) };
    }

    pub fn destroy(&mut self, id: Id) {
        unsafe { DestroyWindow(id as HWND) };
    }

    pub fn pump(&mut self, until: Option<Instant>, _fds: &[i32]) -> Pumped {
        let timeout = until.map_or(INFINITE, |t| t.saturating_duration_since(Instant::now()).as_millis().min(u32::MAX as u128) as u32);
        unsafe {
            MsgWaitForMultipleObjects(1, &self.event, 0, timeout, QS_ALLINPUT);
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        Pumped { closed: CLOSED.with(|c| std::mem::take(&mut *c.borrow_mut())), ready: Vec::new(), dead: false }
    }
}
