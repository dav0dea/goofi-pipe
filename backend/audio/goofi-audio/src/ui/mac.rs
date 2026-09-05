//! AppKit, on the main thread only: the application's event queue pumped by hand, so a wait can
//! carry a deadline. A window the user closed is one that is no longer visible.

use std::ffi::c_void;
use std::time::Instant;

use objc2::rc::{Allocated, Retained};
use objc2::runtime::AnyObject;
use objc2::{class, msg_send, MainThreadMarker};
use objc2_foundation::{NSDate, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString};

use super::Pumped;

pub type Id = usize;

pub struct Platform {
    app: Retained<AnyObject>,
    windows: Vec<(Id, Retained<AnyObject>, bool)>,
}

/// The application, to post a wake-up event to from any thread — the one call AppKit allows there.
pub struct Waker(usize);

unsafe impl Send for Waker {}
unsafe impl Sync for Waker {}

const APPLICATION_DEFINED: usize = 15;
const TITLED_CLOSABLE_MINIATURIZABLE: usize = 1 | 2 | 4;
const BUFFERED: usize = 2;
const REGULAR: isize = 0;

impl Waker {
    pub fn wake(&self) {
        unsafe {
            let app = self.0 as *mut AnyObject;
            let event: *mut AnyObject = msg_send![
                class!(NSEvent),
                otherEventWithType: APPLICATION_DEFINED,
                location: NSPoint::ZERO,
                modifierFlags: 0usize,
                timestamp: 0f64,
                windowNumber: 0isize,
                context: std::ptr::null::<AnyObject>(),
                subtype: 0i16,
                data1: 0isize,
                data2: 0isize
            ];
            let _: () = msg_send![app, postEvent: event, atStart: true];
        }
    }
}

impl Platform {
    pub fn open() -> Result<Platform, String> {
        MainThreadMarker::new().ok_or("the window loop needs the main thread")?;
        unsafe {
            let app: Retained<AnyObject> = msg_send![class!(NSApplication), sharedApplication];
            let _: bool = msg_send![&*app, setActivationPolicy: REGULAR];
            let _: () = msg_send![&*app, finishLaunching];
            Ok(Platform { app, windows: Vec::new() })
        }
    }

    pub fn waker(&self) -> Waker {
        Waker(Retained::as_ptr(&self.app) as usize)
    }

    pub fn create(&mut self, title: &str, (w, h): (u32, u32)) -> Result<(Id, *mut c_void), String> {
        unsafe {
            let rect = NSRect::new(NSPoint::ZERO, NSSize::new(w as f64, h as f64));
            let window: Allocated<AnyObject> = msg_send![class!(NSWindow), alloc];
            let window: Retained<AnyObject> = msg_send![
                window,
                initWithContentRect: rect,
                styleMask: TITLED_CLOSABLE_MINIATURIZABLE,
                backing: BUFFERED,
                defer: false
            ];
            let _: () = msg_send![&*window, setReleasedWhenClosed: false];
            let _: () = msg_send![&*window, setTitle: &*NSString::from_str(title)];
            let _: () = msg_send![&*window, center];
            let _: () = msg_send![&*window, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];
            let _: () = msg_send![&*self.app, activateIgnoringOtherApps: true];
            let view: *mut AnyObject = msg_send![&*window, contentView];
            if view.is_null() {
                return Err("the window has no content view".into());
            }
            let id = Retained::as_ptr(&window) as usize;
            self.windows.push((id, window, true));
            Ok((id, view as *mut c_void))
        }
    }

    pub fn resize(&mut self, id: Id, (w, h): (u32, u32)) {
        if let Some((_, window, _)) = self.windows.iter().find(|(i, ..)| *i == id) {
            let _: () = unsafe { msg_send![&**window, setContentSize: NSSize::new(w as f64, h as f64)] };
        }
    }

    pub fn destroy(&mut self, id: Id) {
        if let Some(at) = self.windows.iter().position(|(i, ..)| *i == id) {
            let (_, window, _) = self.windows.remove(at);
            let _: () = unsafe { msg_send![&*window, close] };
        }
    }

    pub fn pump(&mut self, until: Option<Instant>, _fds: &[i32]) -> Pumped {
        let first: Retained<NSDate> = match until {
            Some(t) => NSDate::dateWithTimeIntervalSinceNow(t.saturating_duration_since(Instant::now()).as_secs_f64()),
            None => NSDate::distantFuture(),
        };
        let rest: Retained<NSDate> = NSDate::distantPast();
        let mut date = &first;
        unsafe {
            loop {
                let event: *mut AnyObject = msg_send![
                    &*self.app,
                    nextEventMatchingMask: usize::MAX,
                    untilDate: &**date,
                    inMode: NSDefaultRunLoopMode,
                    dequeue: true
                ];
                if event.is_null() {
                    break;
                }
                let _: () = msg_send![&*self.app, sendEvent: event];
                date = &rest;
            }
            let _: () = msg_send![&*self.app, updateWindows];
        }
        let closed = self
            .windows
            .iter_mut()
            .filter_map(|(id, window, shown)| {
                let visible: bool = unsafe { msg_send![&**window, isVisible] };
                let gone = *shown && !visible;
                *shown = visible;
                gone.then_some(*id)
            })
            .collect();
        Pumped { closed, ready: Vec::new(), dead: false }
    }
}
