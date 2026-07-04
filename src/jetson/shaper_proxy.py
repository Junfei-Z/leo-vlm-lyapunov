#!/usr/bin/env python3
"""Userspace ISL link shaper for the RPC split benchmark (no root needed).
TCP proxy 127.0.0.1:50053 -> 127.0.0.1:50052 shaping BOTH directions to an
(R Mbps, D ms one-way) link. Correct pipeline semantics: chunk delivery time =
max(arrival + D, previous_delivery + len/R)  -- D is latency, R is rate; D is
NOT added per-chunk serially.
Shaping parameters are re-read from /home/htj/shape.conf (two numbers:
"RATE_MBPS DELAY_MS"; 0 0 = unlimited) every 100 ms, so the persistent RPC
connection can be re-shaped between tiers without reloading the model.
Advantage over tc/netem on lo: shapes ONLY this link, needs no sudo, and the
model load can run unshaped.
"""
import asyncio, os, time

LISTEN = 50053
TARGET = ("127.0.0.1", 50052)
CONF = "/home/htj/shape.conf"
CHUNK = 65536

_params = (0.0, 0.0)      # (bytes_per_sec 0=inf, delay_s)
_last_read = 0.0

def params():
    global _params, _last_read
    now = time.time()
    if now - _last_read > 0.1:
        _last_read = now
        try:
            r, d = open(CONF).read().split()[:2]
            rate = float(r) * 1e6 / 8.0     # Mbps -> bytes/s
            _params = (rate, float(d) / 1000.0)
        except Exception:
            _params = (0.0, 0.0)
    return _params

BURST_LOG = open("/home/htj/rpc_bursts.log", "a", buffering=1)

async def pump(reader, writer, tag=""):
    next_free = 0.0                        # link-busy horizon (rate pacing)
    try:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                break
            BURST_LOG.write(f"{time.time():.4f} {tag} {len(data)}\n")
            rate, delay = params()
            now = time.monotonic()
            if rate > 0:
                start = max(now, next_free)
                next_free = start + len(data) / rate
                deliver = max(start + len(data) / rate, now + delay)
            else:
                next_free = now
                deliver = now + delay
            wait = deliver - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

async def handle(cr, cw):
    try:
        sr, sw = await asyncio.open_connection(*TARGET)
    except OSError:
        cw.close(); return
    await asyncio.gather(pump(cr, sw, 'c2s'), pump(sr, cw, 's2c'))

async def main():
    if not os.path.exists(CONF):
        open(CONF, "w").write("0 0\n")
    server = await asyncio.start_server(handle, "127.0.0.1", LISTEN)
    print("shaper on :%d -> %s (conf: %s)" % (LISTEN, TARGET, CONF), flush=True)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
