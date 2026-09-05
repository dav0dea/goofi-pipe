//! The window thread: one loop that owns every native window goofi opens, run on the thread the
//! entry point hands it — the process main thread when goofi serves. A plugin is loaded, its
//! controller made and its editor pumped HERE, because a JUCE plugin takes the thread that loaded
//! it for its message thread and aborts on any other.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
#[path = "x11.rs"]
mod platform;
#[cfg(windows)]
#[path = "win.rs"]
mod platform;
#[cfg(target_os = "macos")]
#[path = "mac.rs"]
mod platform;

type Job = Box<dyn FnOnce(&mut Host) + Send>;
type Handler = Box<dyn FnMut()>;
type OnClose = Box<dyn FnMut(&mut Host)>;

thread_local! {
    static RESIZES: RefCell<Vec<(platform::Id, (u32, u32))>> = const { RefCell::new(Vec::new()) };
}

/// The door onto the window thread from any other thread.
#[derive(Clone)]
pub struct Ui {
    jobs: mpsc::Sender<Job>,
    waker: Arc<platform::Waker>,
    thread: ThreadId,
}

impl Ui {
    /// Run `f` on the window thread and wait for its answer. A job may borrow the caller's frame:
    /// the wait is what makes that sound, and it is why the loop never stops while a `Ui` exists.
    pub fn run<'a, T: Send + 'a>(&self, f: impl FnOnce(&mut Host) -> T + Send + 'a) -> T {
        assert_ne!(std::thread::current().id(), self.thread, "a window-thread job asked for the window thread");
        let (tx, rx) = mpsc::channel();
        let job: Box<dyn FnOnce(&mut Host) + Send + 'a> = Box::new(move |host| {
            let _ = tx.send(f(host));
        });
        // SAFETY: `recv` returns only once the job has run or was dropped unrun, so nothing it
        // borrows is touched after this frame.
        let job = unsafe { std::mem::transmute::<Box<dyn FnOnce(&mut Host) + Send + 'a>, Job>(job) };
        self.jobs.send(job).expect("the window thread is running");
        self.waker.wake();
        rx.recv().expect("the window thread answers every job")
    }

    /// Hand `f` to the window thread and carry on.
    pub fn post(&self, f: impl FnOnce(&mut Host) + Send + 'static) {
        let _ = self.jobs.send(Box::new(f));
        self.waker.wake();
    }

    /// End the loop once every job before this has run — for a loop on a thread of its own.
    pub fn stop(&self) {
        self.post(|host| host.stopped = true);
    }
}

/// The loop, owned by the thread that runs it.
pub struct Loop {
    jobs: mpsc::Receiver<Job>,
    host: Host,
}

impl Loop {
    /// Open the platform on THIS thread; refused where no display answers, and on macOS off the
    /// main thread.
    pub fn open() -> Result<(Loop, Ui), String> {
        let platform = platform::Platform::open()?;
        let waker = Arc::new(platform.waker());
        let (jobs, rx) = mpsc::channel();
        let ui = Ui { jobs, waker, thread: std::thread::current().id() };
        let host = Host {
            platform,
            runloop: Rc::new(RefCell::new(Runloop::default())),
            on_close: HashMap::new(),
            dead: false,
            stopped: false,
        };
        Ok((Loop { jobs: rx, host }, ui))
    }

    /// Pump until [`Ui::stop`]. A display that goes away ends no server: jobs are still answered,
    /// and every window call refuses.
    pub fn run(mut self) {
        loop {
            while let Ok(job) = self.jobs.try_recv() {
                let _ = catch_unwind(AssertUnwindSafe(|| job(&mut self.host)));
            }
            self.host.apply_resizes();
            if self.host.stopped {
                return;
            }
            if self.host.dead {
                match self.jobs.recv() {
                    Ok(job) => {
                        let _ = catch_unwind(AssertUnwindSafe(|| job(&mut self.host)));
                    }
                    Err(_) => return,
                }
                continue;
            }
            let (until, fds) = {
                let rl = self.host.runloop.borrow();
                (rl.timers.iter().map(|t| t.next).min(), rl.fds.iter().map(|f| f.fd).collect::<Vec<i32>>())
            };
            let pumped = self.host.platform.pump(until, &fds);
            self.host.dead = pumped.dead;
            for id in pumped.closed {
                if let Some(mut on_close) = self.host.on_close.remove(&id) {
                    let _ = catch_unwind(AssertUnwindSafe(|| on_close(&mut self.host)));
                }
            }
            for fd in pumped.ready {
                self.host.fire(|rl| rl.fds.iter_mut().find(|f| f.fd == fd).map(|f| &mut f.handler));
            }
            let now = Instant::now();
            let due: Vec<usize> =
                self.host.runloop.borrow().timers.iter().filter(|t| t.next <= now).map(|t| t.key).collect();
            for key in due {
                if let Some(t) = self.host.runloop.borrow_mut().timers.iter_mut().find(|t| t.key == key) {
                    t.next = now + t.every;
                }
                self.host.fire(|rl| rl.timers.iter_mut().find(|t| t.key == key).map(|t| &mut t.handler));
            }
            self.host.apply_resizes();
        }
    }
}

/// What the platform's pump found: windows the user closed, descriptors a plugin watches that
/// are readable, and whether the display is gone.
pub struct Pumped {
    pub closed: Vec<platform::Id>,
    pub ready: Vec<i32>,
    pub dead: bool,
}

/// The window thread's own state: the platform's windows, and what a plugin registered with it.
pub struct Host {
    platform: platform::Platform,
    runloop: Rc<RefCell<Runloop>>,
    on_close: HashMap<platform::Id, OnClose>,
    dead: bool,
    stopped: bool,
}

/// One native window, and the handle a plugin's view is attached to.
#[derive(Clone, Copy)]
pub struct Window {
    id: platform::Id,
    pub parent: *mut c_void,
}

impl Window {
    /// Asked from inside a plugin's callback, where the host is borrowed: the loop applies it
    /// right after.
    pub fn request_resize(&self, size: (u32, u32)) {
        RESIZES.with(|r| r.borrow_mut().push((self.id, size)));
    }
}

impl Host {
    /// A top-level window of `size`; `on_close` runs when the user closes it, and is what tears
    /// down whatever was inside.
    pub fn open_window(
        &mut self,
        title: &str,
        size: (u32, u32),
        on_close: OnClose,
    ) -> Result<Window, String> {
        if self.dead {
            return Err("the display is gone".into());
        }
        let (id, parent) = self.platform.create(title, size)?;
        self.on_close.insert(id, on_close);
        Ok(Window { id, parent })
    }

    fn apply_resizes(&mut self) {
        for (id, size) in RESIZES.with(|r| std::mem::take(&mut *r.borrow_mut())) {
            self.platform.resize(id, size);
        }
    }

    pub fn close_window(&mut self, window: Window) {
        self.on_close.remove(&window.id);
        self.platform.destroy(window.id);
    }

    /// The tables a plugin registers its descriptors and timers in.
    pub fn runloop(&self) -> Rc<RefCell<Runloop>> {
        self.runloop.clone()
    }

    /// Call one registered handler with the tables unborrowed, so it may register or unregister:
    /// taken out for the call, and put back where its entry still stands.
    fn fire(&mut self, find: impl Fn(&mut Runloop) -> Option<&mut Option<Handler>>) {
        let taken = find(&mut self.runloop.borrow_mut()).and_then(Option::take);
        if let Some(mut handler) = taken {
            let _ = catch_unwind(AssertUnwindSafe(&mut handler));
            if let Some(slot) = find(&mut self.runloop.borrow_mut()) {
                *slot = Some(handler);
            }
        }
    }
}

/// What a plugin registers with the host's run loop: descriptors to watch and timers to fire,
/// each keyed by the plugin's own handler pointer, which is how it unregisters them.
#[derive(Default)]
pub struct Runloop {
    timers: Vec<Timer>,
    fds: Vec<Fd>,
}

struct Timer {
    key: usize,
    every: Duration,
    next: Instant,
    handler: Option<Handler>,
}

struct Fd {
    key: usize,
    fd: i32,
    handler: Option<Handler>,
}

impl Runloop {
    pub fn add_timer(&mut self, key: usize, every: Duration, handler: Handler) {
        self.timers.push(Timer { key, every, next: Instant::now() + every, handler: Some(handler) });
    }

    pub fn remove_timer(&mut self, key: usize) -> bool {
        let n = self.timers.len();
        self.timers.retain(|t| t.key != key);
        n != self.timers.len()
    }

    pub fn add_fd(&mut self, key: usize, fd: i32, handler: Handler) {
        self.fds.push(Fd { key, fd, handler: Some(handler) });
    }

    pub fn remove_fd(&mut self, key: usize) -> bool {
        let n = self.fds.len();
        self.fds.retain(|f| f.key != key);
        n != self.fds.len()
    }
}
