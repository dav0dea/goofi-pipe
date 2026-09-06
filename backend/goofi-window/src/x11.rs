//! X11, over a connection of goofi's own: a plugin embeds into the window id it is handed and
//! talks to the server on a connection of its own, which it registers with the run loop.

use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::time::Instant;

use x11rb::connection::{Connection, RequestConnection};
use x11rb::properties::WmSizeHints;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

use super::{Id, Pumped, Screen, Wake};

pub struct Platform {
    conn: RustConnection,
    root: Window,
    black: u32,
    wm_protocols: Atom,
    wm_delete: Atom,
    /// A pipe: any thread writes to wake the pump out of its `poll`.
    wake: [i32; 2],
    depth: u8,
    /// One graphics context per presented window, and the scratch the swizzle reuses.
    gcs: std::collections::HashMap<Window, Gcontext>,
    scratch: Vec<u8>,
}

pub struct Waker(i32);

impl Wake for Waker {
    fn wake(&self) {
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
        let depth = screen.root_depth;
        Ok(Platform {
            conn,
            root,
            black,
            wm_protocols,
            wm_delete,
            wake,
            depth,
            gcs: std::collections::HashMap::new(),
            scratch: Vec::new(),
        })
    }

    /// The plugin draws at one size, so the window manager is told not to offer another.
    fn fix_size(&self, id: Window, (w, h): (u16, u16)) -> Result<(), String> {
        let mut hints = WmSizeHints::new();
        hints.min_size = Some((w as i32, h as i32));
        hints.max_size = Some((w as i32, h as i32));
        hints.set_normal_hints(&self.conn, id).map_err(err)?;
        Ok(())
    }
}

impl Screen for Platform {
    fn waker(&self) -> Arc<dyn Wake> {
        Arc::new(Waker(self.wake[1]))
    }

    fn create(&mut self, title: &str, (w, h): (u32, u32)) -> Result<(Id, *mut c_void), String> {
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
        Ok((id as Id, id as usize as *mut c_void))
    }

    fn resize(&mut self, id: Id, (w, h): (u32, u32)) {
        let id = id as Window;
        let (w, h) = (w.clamp(1, u16::MAX as u32) as u16, h.clamp(1, u16::MAX as u32) as u16);
        let _ = self.fix_size(id, (w, h));
        let _ = self.conn.configure_window(id, &ConfigureWindowAux::new().width(w as u32).height(h as u32));
        let _ = self.conn.flush();
    }

    fn destroy(&mut self, id: Id) {
        if let Some(gc) = self.gcs.remove(&(id as Window)) {
            let _ = self.conn.free_gc(gc);
        }
        let _ = self.conn.destroy_window(id as Window);
        let _ = self.conn.flush();
    }

    /// `PutImage`, in bands: one request carries at most the server's maximum, and a frame is
    /// far larger than the 256 KB a server without BIG-REQUESTS accepts.
    fn present(&mut self, id: Id, (w, h): (u32, u32), rgba: &[u8]) {
        let win = id as Window;
        let gc = match self.gcs.get(&win) {
            Some(gc) => *gc,
            None => {
                let Ok(gc) = self.conn.generate_id() else { return };
                if self.conn.create_gc(gc, win, &CreateGCAux::new()).is_err() {
                    return;
                }
                self.gcs.insert(win, gc);
                gc
            }
        };
        super::bgra_into(rgba, &mut self.scratch);
        let stride = w as usize * 4;
        let cap = RequestConnection::maximum_request_bytes(&self.conn).saturating_sub(64);
        let per = (cap / stride.max(1)).max(1);
        let mut y = 0usize;
        while y < h as usize {
            let rows = per.min(h as usize - y);
            let band = &self.scratch[y * stride..(y + rows) * stride];
            let put = self.conn.put_image(
                ImageFormat::Z_PIXMAP,
                win,
                gc,
                w as u16,
                rows as u16,
                0,
                y as i16,
                0,
                self.depth,
                band,
            );
            if put.is_err() {
                return;
            }
            y += rows;
        }
        let _ = self.conn.flush();
    }

    fn pump(&mut self, until: Option<Instant>, fds: &[i32]) -> Pumped {
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
                    closed.push(m.window as Id)
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
