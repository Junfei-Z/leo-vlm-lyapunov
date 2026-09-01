#!/usr/bin/env python3
"""Shared infrastructure for TMC experiments 1-3.

Loads the split-pipeline simulator, applies the TLE primary configuration
(P_solar=12.2 W, B_max=16580 J, P_base=7 W, lambda=0.10, h=0.69 s, rho=0.85),
and exposes apply_config() so a worker can override any module attribute
(U_TGT, V, RESERVE_FRAC, P_SOLAR, B_MAX, ...) before calling run_sim().

NOTE for multiprocessing (spawn): each worker imports this module afresh, so
the simulator module M and the TLE arrays are per-process. Set attributes via
apply_config() inside the worker task, never at import time.
"""
import importlib.util, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(SPEC)
sys.modules["splitpipe"] = M
SPEC.loader.exec_module(M)

_NPZ = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = _NPZ["sunlit"]
ISL = _NPZ["isl"]
META = json.loads(str(_NPZ["meta"]))
assert META["gate"] == "PASS"
T, S = SUNLIT.shape
S2 = np.concatenate([SUNLIT, SUNLIT], axis=0)
# F[t,s] = eclipse slots remaining before the next sunrise (vectorized):
#   eclipse slot: distance to the next 0->1 transition (sunrise)
#   sunlit slot : length of the eclipse run following the current sunlit run
F = np.zeros((T, S), dtype=np.int32)
_t = np.arange(T)
for s in range(S):
    col = S2[:, s].astype(np.int8)
    trans = np.flatnonzero(np.diff(col)) + 1              # state-change positions in [1, 2T-1]
    t10 = trans[col[trans - 1] == 1]                      # 1->0 (eclipse onset)
    t01 = trans[col[trans - 1] == 0]                      # 0->1 (sunrise)
    # eclipse slots: slots until the next sunrise
    i01 = np.searchsorted(t01, _t, side="left")
    d_ecl = np.where(i01 < len(t01), t01[np.minimum(i01, len(t01) - 1)] - _t, T - _t)
    # sunlit slots: length of the next eclipse run
    i10 = np.searchsorted(t10, _t, side="left")
    i10 = np.minimum(i10, len(t10) - 1)
    end_run = t10[i10]                                    # end of the current sunlit run
    i01b = np.searchsorted(t01, end_run, side="right")    # first sunrise strictly after it
    i01b = np.minimum(i01b, len(t01) - 1)
    ecl_sunlit = t01[i01b] - end_run
    F[:, s] = np.where(col[:_t.shape[0]] == 0, d_ecl, ecl_sunlit)
BASE_ISL = ISL.copy()

# Primary configuration (mirrors src/14_tle_primary.apply_primary)
PRIMARY = dict(P_SOLAR=12.2, B_MAX=16580.0, B_INIT=0.6 * 16580.0,
               P_BASE=7.0, ARRIVAL_PROB=0.10, HANDOFF_S=0.69, RESERVE_FRAC=0.85)


def apply_primary():
    M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
    M.isl_connected = lambda t, s1, s2: bool(BASE_ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
    M.eclipse_slots_to_sunrise = lambda t, s: int(F[t % T, s])
    for k, v in PRIMARY.items():
        setattr(M, k, v)


def apply_config(cfg):
    """Reset to the primary config, then override with cfg (dict of attr->value)."""
    apply_primary()
    for k, v in (cfg or {}).items():
        setattr(M, k, v)
