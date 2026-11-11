use std::net::{SocketAddrV4, TcpListener};

/// Holds an allocated TCP port open to avoid TOCTOU races between allocation and process launch.
pub struct HeldPort {
    port: u16,
    listener: Option<TcpListener>,
}

impl HeldPort {
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Releases the held listener socket right before process launch.
    pub fn release(mut self) -> u16 {
        let port = self.port;
        drop(self.listener.take());
        // Small delay if necessary can be handled by caller
        port
    }
}

impl Drop for HeldPort {
    fn drop(&mut self) {
        drop(self.listener.take());
    }
}

/// Allocates an available TCP port starting from 8000, skipping existing ports,
/// and holds the listener open to prevent collisions.
pub fn allocate_port_held(existing_ports: &[u16]) -> Result<HeldPort, String> {
    for port in 8000..9000 {
        if existing_ports.contains(&port) {
            continue;
        }
        let addr = SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, port);
        if let Ok(listener) = TcpListener::bind(addr) {
            return Ok(HeldPort {
                port,
                listener: Some(listener),
            });
        }
    }
    Err("No free port available in range 8000-9000".to_string())
}
