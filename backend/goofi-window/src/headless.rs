//! No screen: the suite's window host, as its clock has no device. A window is a number, nothing
//! reaches a desktop, and a descriptor is not watched — that needs a display's poll, and the
//! suite's plugin registers a timer.

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use super::{Id, Pumped, Screen, Wake};

#[derive(Default)]
pub struct Platform {
    wake: Arc<Signal>,
    next: Id,
}

#[derive(Default)]
struct Signal {
    woke: Mutex<bool>,
    cv: Condvar,
}

impl Wake for Signal {
    fn wake(&self) {
        *self.woke.lock().unwrap() = true;
        self.cv.notify_one();
    }
}

impl Screen for Platform {
    fn waker(&self) -> Arc<dyn Wake> {
        self.wake.clone()
    }

    fn create(&mut self, _title: &str, _size: (u32, u32)) -> Result<(Id, *mut c_void), String> {
        self.next += 1;
        Ok((self.next, self.next as usize as *mut c_void))
    }

    fn resize(&mut self, _id: Id, _size: (u32, u32)) {}

    fn destroy(&mut self, _id: Id) {}

    fn pump(&mut self, until: Option<Instant>, _fds: &[i32]) -> Pumped {
        let mut woke = self.wake.woke.lock().unwrap();
        while !*woke {
            let Some(until) = until else {
                woke = self.wake.cv.wait(woke).unwrap();
                continue;
            };
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            woke = self.wake.cv.wait_timeout(woke, left).unwrap().0;
        }
        *woke = false;
        Pumped { closed: Vec::new(), ready: Vec::new(), dead: false }
    }
}
