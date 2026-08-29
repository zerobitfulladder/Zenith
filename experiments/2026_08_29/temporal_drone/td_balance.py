r"""Balance: zero velocity, zero angle, position free.

The validation rung under the position task. Same bound machinery as
td_bound — the command lives inside the template, partial-cue read,
top-1 winner, one graded bump per channel — on a smaller world where
the two measured killers of the position task are absent by
construction:

  * hold data exists: episodes are fixed-length, so most training
    ticks ARE the balanced hover the gate demands;
  * the blind start is aligned: the teacher wears the pupil's handicap
    (plain hover while the track windows fill), so "a second of
    uncorrected tumble, then recovery" is exactly what the memory
    stores.

Tracks: vx, vy, tilt — velocity sensed directly, its own encoding (the
speedometer law; position is not sensed at all, so velocity cannot be
somebody's slope). Success = at_goal minus its position term: speed
under 0.3 m/s and |tilt| under 0.17 rad, held 10 ticks.

The motors receive (left, right) thrust levels as always; collective
and differential exist only inside the template, for the measured
reason (a steering signal read as the difference of two independent
channels arrives with both reads' errors — SNR was exactly 1).
"""

import numpy as np

import sl_drone as W
import td_tracks as B
import td3_stack as M
import fast_oracle as F
import td_bound as T

NS = 4                        # vx, vy, tilt, gyro
VSCALE = 6.0                  # m/s at the rim of a LINEAR velocity axis
VMAX = 20.0                   # m/s at the rim of the WARPED axis
GSCALE = 4.0                  # rad/s at the rim of a LINEAR gyro axis
GMAX = 10.0                   # rad/s at the rim of the FOVEATED gyro axis
EP_LEN = 400                  # ticks per training episode (8 s)


def make_sense(ns=NS, vwarp=True, vscale=VSCALE, fovea=False):
    """fovea=True: every axis warped so the SUCCESS BAND spans cells.

    Measured need (2026-08-30): with the flat/mild encodings the whole
    success band (|v| < 0.3, |tilt| < 0.17, |gyro| small) sits inside
    ONE cell of every axis — the cue is constant across the hold basin,
    the read cannot correlate command with state there, the loop has no
    small-signal gain, and the state random-walks out (median loss of
    balance at ~1.3 s; graded top-k reads change nothing, so it is not
    winner quantisation). The fovea puts 3-5 cells inside the band.
    """
    if fovea:
        dv = float(np.arcsinh(VMAX / 0.15))
        dt = float(np.arcsinh(np.pi / 0.05))
        dg = float(np.arcsinh(GMAX / 0.1))

        def sense(s):
            out = [np.clip(np.arcsinh(s[2] / 0.15) / dv, -1, 1),
                   np.clip(np.arcsinh(s[3] / 0.15) / dv, -1, 1),
                   np.clip(np.arcsinh(W.wrap(s[4]) / 0.05) / dt, -1, 1)]
            if ns >= 4:
                out.append(np.clip(np.arcsinh(s[5] / 0.1) / dg, -1, 1))
            return np.array(out, np.float32)
        return sense
    """The sensed channels. Two measured lessons live here (probe of
    2026-08-29, balance/results):

      * ns >= 4 adds the gyro as its own track — the spin rate as the
        tilt window's slope is ~1 cell of bump offset per tap, so
        opposite spins at the same tilt read ~90% alike and the pupil
        corrects angle without killing rotation (the sawtooth tumble
        in eval_traces.png). The speedometer law, one derivative up.
      * vwarp replaces the hard +-6 m/s clip with an arcsinh fovea —
        17/40 tumble episodes reached engagement already saturated
        (mean speed there 6.0 m/s), state-blind from the first read.
    """
    den = float(np.arcsinh(VMAX / 1.5))

    def enc_v(v):
        if vwarp:
            return float(np.arcsinh(v / 1.5)) / den
        return v / vscale

    def sense(s):
        out = [np.clip(enc_v(s[2]), -1, 1), np.clip(enc_v(s[3]), -1, 1),
               np.clip(W.wrap(s[4]) / np.pi, -1, 1)]
        if ns >= 4:
            out.append(np.clip(s[5] / GSCALE, -1, 1))
        return np.array(out, np.float32)
    return sense


def teacher(s):
    """fast_oracle's cascade with the commanded velocity pinned to 0."""
    ph = W.wrap(s[4])
    phi_des = np.clip(F.KVX * s[2], -F.PHIMAX, F.PHIMAX)
    coll = (W.HOVER - F.KVY * s[3]) / max(np.cos(ph), 0.5)
    coll = np.clip(coll, 0.3, W.TMAX)
    diff = np.clip(F.KA * (phi_des - ph) - F.KDA * s[5], -F.DMAX, F.DMAX)
    t1 = np.clip(coll - diff, 0, W.TMAX)
    t2 = np.clip(coll + diff, 0, W.TMAX)
    return (int(np.argmin(np.abs(W.LEVELS - t1))),
            int(np.argmin(np.abs(W.LEVELS - t2))))


def balanced(s):
    return (np.hypot(s[2], s[3]) < 0.3
            and abs(W.wrap(s[4])) < 0.17)


def air_init(rng):
    """Position pinned at the origin (it does not matter and is not
    sensed); attitude and velocities as the drone world hands them out."""
    return np.array([0.0, 0.0, rng.uniform(-1.5, 1.5),
                     rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0),
                     rng.uniform(-1.5, 1.5)])


def load_policy(npz, cfg):
    """Hook for the shared viewer. The target is ignored — this policy
    only stops and levels. Reset detection watches the sensed
    quantities (a teleported tilt or velocity), not position."""
    st = T.load_stack(npz, cfg)
    sense = make_sense(ns=int(cfg.get("ns", 3)),
                       vwarp=bool(cfg.get("vwarp", False)),
                       vscale=float(cfg.get("vscale", VSCALE)),
                       fovea=bool(cfg.get("fovea", False)))
    prev = {"p": None}

    def act(s, tgt, lv_prev):
        # A teleported position or tilt is a viewer reset — clear the
        # tracks. A velocity step alone is the viewer's GUST: a real
        # in-world disturbance, so keep the history and react to it.
        p = (float(s[0]), float(s[1]), float(W.wrap(s[4])))
        if (prev["p"] is None
                or np.hypot(p[0] - prev["p"][0], p[1] - prev["p"][1]) > 2.0
                or abs(p[2] - prev["p"][2]) > 0.5):
            st.reset()
        prev["p"] = p
        st.push(sense(s))
        lv, _, _ = st.act()
        return lv
    return act
