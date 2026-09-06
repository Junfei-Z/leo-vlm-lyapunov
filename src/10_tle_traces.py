#!/usr/bin/env python3
"""P0-3: TLE-driven eclipse + ISL traces for the sim (replaces the synthetic
60/40 schedule and hash-toggled ISL with real ephemeris).

Pipeline: parse a real Starlink TLE snapshot -> 550 km shell (mean motion
15.04-15.08 rev/day; the tight band excludes orbit-raising/deorbiting sats) ->
group into strict orbital planes (RAAN within +/-1.0 deg of the plane median) ->
among planes ordered by |beta| (deepest eclipse first, the energy-worst case),
take the FIRST whose sats admit a connected 4-chain: four consecutive sats whose
neighbor ranges stay <= 4500 km over the whole study window (robust against
maneuvering sats; adjacency measured on PROPAGATED positions at the common
epoch, because TLE mean anomalies are quoted at per-sat epochs) -> propagate
with SGP4 at 1 s steps over 3 orbits -> sunlit flag via a cylindrical
Earth-shadow test against a low-precision analytic Sun vector -> ISL
connectivity via line-of-sight (chord clears Earth + 80 km) and a 5000 km laser
range cap.

FINDING baked into the selection rule: quarter-orbit phase staggering (the
synthetic sim's SAT_PHASE) and intra-plane ISL connectivity are mutually
exclusive; a physically consistent single-plane +Grid uses ADJACENT sats, so
eclipse hits all four nearly simultaneously (an energy-harder regime than the
synthetic staggered schedule).

PHYSICAL GATE (pre-registered): period ~95-96 min; sunlit fraction in
[0.55, 0.70]; every orbit-order neighbor pair ISL-available >= 0.95.
FAIL -> do not feed the sim.

Output: data/tle_traces.npz  (sunlit [T,S] bool, isl [T,S,S] bool, meta json)
"""
import os, sys, json, math
import numpy as np
from sgp4.api import Satrec

HERE = os.path.dirname(os.path.abspath(__file__))
TLE_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "starlink_20260702.tle")
OUT = os.path.join(HERE, "..", "data", "tle_traces.npz")

N_SAT = 4
HORIZON = 17100            # 3 x 5700 s, matching the sim
RE = 6378.137              # km
ATM = 80.0                 # km grazing margin
ISL_MAX_KM = 5000.0
CHAIN_MAX_KM = 4500.0      # neighbor-range bound over the window for a valid chain


def parse_tles(path):
    lines = [l.rstrip() for l in open(path) if l.strip()]
    out = []
    for i in range(0, len(lines) - 2):
        if lines[i + 1][:2] == "1 " and lines[i + 2][:2] == "2 ":
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
    return out


def sun_vector(jd_full):
    """Low-precision analytic Sun unit vector (Astronomical Almanac), ~0.01 deg."""
    n = jd_full - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam),
                     math.sin(eps) * math.sin(lam)])


def is_sunlit(r, s_hat):
    along = np.dot(r, s_hat)
    if along >= 0.0:
        return True
    return np.linalg.norm(r - along * s_hat) > RE


def los_clear(r1, r2):
    d = r2 - r1
    dd = np.dot(d, d)
    if dd == 0:
        return False
    t = min(1.0, max(0.0, -np.dot(r1, d) / dd))
    return np.linalg.norm(r1 + t * d) > (RE + ATM)


def try_plane(plane, jd0, fr0):
    """Return (picks, maxgap) for the tightest connected 4-chain, or None."""
    med_raan = sorted(x[2] for x in plane)[len(plane) // 2]
    plane = [x for x in plane if abs(x[2] - med_raan) <= 1.0]
    if len(plane) < N_SAT:
        return None
    props = []
    for entry in plane:
        e, r, _ = entry[1].sgp4(jd0, fr0)
        if e == 0:
            props.append((entry, np.array(r)))
    if len(props) < N_SAT:
        return None
    # along-track angle at the common epoch (frame built from sat0's motion)
    e2, r2, _ = props[0][0][1].sgp4(jd0, fr0 + 30 / 86400.0)
    if e2 != 0:
        return None
    n_hat = np.cross(props[0][1], np.array(r2) - props[0][1])
    n_hat /= np.linalg.norm(n_hat)
    ref_u = props[0][1] / np.linalg.norm(props[0][1])
    ref_v = np.cross(n_hat, ref_u)

    def theta(r):
        ru = r / np.linalg.norm(r)
        return math.degrees(math.atan2(np.dot(ru, ref_v), np.dot(ru, ref_u))) % 360.0

    order = sorted(range(len(props)), key=lambda j: theta(props[j][1]))
    # sampled positions over the whole window (robust against maneuvering sats)
    samples = list(range(0, HORIZON, 300))
    P = np.zeros((len(props), len(samples), 3))
    for j, (entry, _) in enumerate(props):
        for si, t in enumerate(samples):
            e, r, _ = entry[1].sgp4(jd0, fr0 + t / 86400.0)
            P[j, si] = r if e == 0 else 1e9
    best = None
    for i in range(len(order)):
        idx = [order[(i + k) % len(order)] for k in range(N_SAT)]
        maxgap = 0.0
        for k in range(N_SAT - 1):
            maxgap = max(maxgap, float(np.max(np.linalg.norm(P[idx[k]] - P[idx[k + 1]], axis=1))))
        if best is None or maxgap < best[1]:
            best = (idx, maxgap)
    idx, maxgap = best
    if maxgap > CHAIN_MAX_KM:
        return None
    picks = [props[j][0] for j in idx]
    gaps0 = [round(float(np.linalg.norm(props[idx[k]][1] - props[idx[k + 1]][1])))
             for k in range(N_SAT - 1)]
    print("picked:", [p[0] for p in picks], f"(neighbor gaps at jd0 = {gaps0} km)")
    return picks, maxgap


def main():
    tles = parse_tles(TLE_FILE)
    print(f"parsed {len(tles)} TLEs")
    shell = []
    for name, l1, l2 in tles:
        mm = float(l2[52:63])
        if 15.04 <= mm <= 15.08:
            sat = Satrec.twoline2rv(l1, l2)
            raan = float(l2[17:25]); inc = float(l2[8:16])
            shell.append((name, sat, raan, inc))
    print(f"550 km shell (mm 15.04-15.08): {len(shell)} sats")
    assert len(shell) >= 20, "shell too small -- check TLE set"

    shell.sort(key=lambda x: x[2])
    planes, cur = [], [shell[0]]
    for s in shell[1:]:
        if abs(s[2] - cur[-1][2]) < 1.5:
            cur.append(s)
        else:
            planes.append(cur)
            cur = [s]
    planes.append(cur)
    planes = [p for p in planes if len(p) >= N_SAT]
    print(f"planes with >= {N_SAT} sats: {len(planes)}")

    ref = planes[0][0][1]
    jd0, fr0 = ref.jdsatepoch, ref.jdsatepochF
    s_hat0 = sun_vector(jd0 + fr0)
    scored = []
    for p in planes:
        name, sat, raan, inc = p[0]
        i, O = math.radians(inc), math.radians(raan)
        h_hat = np.array([math.sin(O) * math.sin(i), -math.cos(O) * math.sin(i), math.cos(i)])
        beta = math.degrees(math.asin(np.clip(np.dot(h_hat, s_hat0), -1, 1)))
        scored.append((abs(beta), beta, p))
    scored.sort(key=lambda x: x[0])

    picks = beta = maxgap = None
    for absb, b, plane in scored:
        got = try_plane(plane, jd0, fr0)
        if got is not None:
            picks, maxgap = got
            beta = b
            print(f"selected plane RAAN~{plane[0][2]:.1f}, |beta|={absb:.1f} deg "
                  f"(deepest-eclipse plane admitting a connected 4-chain; "
                  f"max neighbor gap over window {maxgap:.0f} km)")
            break
    assert picks is not None, "no plane admits a connected 4-chain"

    sunlit_arr = np.zeros((HORIZON, N_SAT), dtype=bool)
    pos = np.zeros((HORIZON, N_SAT, 3))
    for t in range(HORIZON):
        fr = fr0 + t / 86400.0
        s_hat = sun_vector(jd0 + fr)
        for k, (_, sat, _, _) in enumerate(picks):
            e, r, _ = sat.sgp4(jd0, fr)
            assert e == 0, f"sgp4 error {e}"
            pos[t, k] = r
            sunlit_arr[t, k] = is_sunlit(np.array(r), s_hat)
    isl = np.zeros((HORIZON, N_SAT, N_SAT), dtype=bool)
    for t in range(HORIZON):
        for a in range(N_SAT):
            for b2 in range(a + 1, N_SAT):
                d = np.linalg.norm(pos[t, a] - pos[t, b2])
                ok = d <= ISL_MAX_KM and los_clear(pos[t, a], pos[t, b2])
                isl[t, a, b2] = isl[t, b2, a] = ok

    # ---- PHYSICAL GATE ----
    mm = float(picks[0][1].no_kozai) * 1440 / (2 * math.pi)
    period_min = 1440.0 / mm
    frac = sunlit_arr.mean(axis=0)
    adj = [isl[:, k, k + 1].mean() for k in range(N_SAT - 1)]
    print("=" * 60)
    print(f"GATE: period = {period_min:.1f} min (expect ~95-96)")
    print(f"GATE: sunlit fraction per sat = {np.round(frac, 3)} (expect 0.55-0.70)")
    print(f"GATE: neighbor-pair ISL availability = {np.round(adj, 2)} (expect >=0.95 each)")
    ok = (94.0 <= period_min <= 97.0) and all(0.55 <= f <= 0.70 for f in frac) \
         and all(a >= 0.95 for a in adj)
    print(f"GATE: {'PASS' if ok else 'FAIL -- do not feed the sim'}")

    np.savez_compressed(OUT, sunlit=sunlit_arr, isl=isl,
                        meta=json.dumps(dict(period_min=period_min, beta=beta,
                                             sats=[n for n, *_ in picks],
                                             sunlit_frac=frac.tolist(),
                                             isl_avail=[float(a) for a in adj],
                                             gate="PASS" if ok else "FAIL")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
