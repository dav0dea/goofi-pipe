//! X11, over a connection of goofi's own: a plugin embeds into the window id it is handed and
//! talks to the server on a connection of its own, which it registers with the run loop.

use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::time::Instant;

use x11rb::connection::Connection;
use x11rb::properties::WmSizeHints;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

use super::Pumped;

pub type Id = Window;

pub struct Platform {
    conn: RustConnection,
    root: Window,
    black: u32,
    wm_protocols: Atom,
    wm_delete: Atom,
    /// A pipe: any thread writes to wake the pump out of its `poll`.
    wake: [i32; 2],
}

pub struct Waker(i32);

impl Waker {
    pub fn wake(&self) {
        unsafe { libc::write(self.0, [1u8].as_ptr() as *const c_void, 1) };
    }
}

fn err(e: impl std::fmt::Display) -> String {
    format!("X11: {e}")
}

impl Platform {
    pub fn open() -> Result<Platform, String> {
        let (conn, screen) = x11rb::connect(None).map_err(|e| format!("no X display: {e}"))?;
        let screen = &conn.setup().roots[screen];
        let (root, black) = (screen.root, screen.black_pixel);
        let atom = |name: &[u8]| conn.intern_atom(false, name).map_err(err)?.reply().map_err(err).map(|r| r.atom);
        let wm_protocols = atom(b"WM_PROTOCOLS")?;
        let wm_delete = atom(b"WM_DELETE_WINDOW")?;
        let mut wake = [0i32; 2];
        if unsafe { libc::pipe2(wake.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) } != 0 {
            return Err("no wake pipe".into());
        }
        Ok(Platform { conn, root, black, wm_protocols, wm_delete, wake })
    }

    pub fn waker(&self) -> Waker {
        Waker(self.wake[1])
    }

    pub fn create(&mut self, title: &str, (w, h): (u32, u32)) -> Result<(Id, *mut c_void), String> {
        let id = self.conn.generate_id().map_err(err)?;
        let aux = CreateWindowAux::new().background_pixel(self.black).event_mask(EventMask::STRUCTURE_NOTIFY);
        let (w, h) = (w.clamp(1, u16::MAX as u32) as u16, h.clamp(1, u16::MAX as u32) as u16);
        self.conn
            .create_window(COPY_DEPTH_FROM_PARENT, id, self.root, 0, 0, w, h, 0, WindowClass::INPUT_OUTPUT, 0, &aux)
            .map_err(err)?;
        self.conn.change_property8(PropMode::REPLACE, id, AtomEnum::WM_NAME, AtomEnum::STRING, title.as_bytes()).map_err(err)?;
        self.conn.change_property32(PropMode::REPLACE, id, self.wm_protocols, AtomEnum::ATOM, &[self.wm_delete]).map_err(err)?;
        self.fix_size(id, (w, h))?;
        self.conn.map_window(id).map_err(err)?;
        self.conn.flush().map_err(err)?;
        Ok((id, id as usize as *mut c_void))
    }

    /// The plugin draws at one size, so the window manager is told not to offer another.
    fn fix_size(&self, id: Id, (w, h): (u16, u16)) -> Result<(), String> {
        let mut hints = WmSizeHints::new();
        hints.min_size = Some((w as i32, h as i32));
        hints.max_size = Some((w as i32, h as i32));
        hints.set_normal_hints(&self.conn, id).map_err(err)?;
        Ok(())
    }

    pub fn resize(&mut self, id: Id, (w, h): (u32, u32)) {
        let (w, h) = (w.clamp(1, u16::MAX as u32) as u16, h.clamp(1, u16::MAX as u32) as u16);
        let _ = self.fix_size(id, (w, h));
        let _ = self.conn.configure_window(id, &ConfigureWindowAux::new().width(w as u32).height(h as u32));
        let _ = self.conn.flush();
    }

    pub fn destroy(&mut self, id: Id) {
        let _ = self.conn.destroy_window(id);
        let _ = self.conn.flush();
    }

    /// Park until a window event, a wake, a readable plugin descriptor or `until`.
    pub fn pump(&mut self, until: Option<Instant>, fds: &[i32]) -> Pumped {
        let dead = self.conn.flush().is_err();
        let mut polled: Vec<libc::pollfd> = [self.conn.stream().as_raw_fd(), self.wake[0]]
            .into_iter()
            .chain(fds.iter().copied())
            .map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 })
            .collect();
        let timeout = until.map_or(-1, |t| t.saturating_duration_since(Instant::now()).as_millis().min(i32::MAX as u128) as i32);
        unsafe { libc::poll(polled.as_mut_ptr(), polled.len() as libc::nfds_t, timeout) };
        let mut sink = [0u8; 64];
        while unsafe { libc::read(self.wake[0], sink.as_mut_ptr() as *mut c_void, sink.len()) } > 0 {}
        let mut closed = Vec::new();
        let mut dead = dead;
        loop {
            match self.conn.poll_for_event() {
                Ok(Some(Event::ClientMessage(m))) if m.type_ == self.wm_protocols && m.data.as_data32()[0] == self.wm_delete => {
                    closed.push(m.window)
                }
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    dead = true;
                    break;
                }
            }
        }
        let ready = polled[2..]
            .iter()
            .filter(|p| p.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0)
            .map(|p| p.fd)
            .collect();
        Pumped { closed, ready, dead }
    }
}
