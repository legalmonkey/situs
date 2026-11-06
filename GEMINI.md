# Situs Project Rules & Architectural Invariants

## Networking & Worker Lifecycle Constraints

1. **Strictly Verified Handshakes (Zero False-Success Reporting)**:
   - Any client command that joins or connects to a cluster (`/join`, `situs join`) MUST NOT report success or render "Connected" until the remote server/coordinator has explicitly sent back a verified registration acknowledgment (`WORKER_REGISTERED`).
   - Network probes and handshakes MUST NOT swallow socket, network, or HTTP exceptions with bare `except Exception: pass`. All connection failures must be captured and surfaced with explicit diagnostics.
   - If registration fails or times out, the worker daemon must be cleanly stopped and removed from active background worker registries.

2. **RFC 1918 Private Subnet Boundaries**:
   - The system MUST acknowledge that RFC 1918 addresses (`192.168.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`) cannot route across different Wi-Fi networks or residential internet connections.
   - When private IPs are dialed from differing subnets, the system must detect the mismatch and explain cross-network routing constraints (e.g. recommend same Wi-Fi, Tailscale, or public tunneling).
   - The coordinator must advertise both local LAN addresses and any detected virtual/VPN interfaces (e.g., Tailscale `100.x.y.z`).
