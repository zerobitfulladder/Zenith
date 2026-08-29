r"""The bound form on sensor tracks: the command lives inside the template.

A layer-3 template is one vector —

    [ dx L2 code | dy L2 code | tilt L2 code | collective | differential ]
       128 cells    128          128            48 cells     48 cells

Training is the ordinary rule and nothing else: the full 480-cell
vector (sensor codes AND the teacher's command bumps) competes, the
winner rotates toward it. Inference is the canonical partial-cue read,
exactly the curve rig's: write the 384 sensor cells, leave the command
cells empty, masked norm-corrected scores (contrast floor 0.25), top-1
winner; the winner's own command cells are the answer, each channel one
graded bump sharpened once on its foveated axis. No identity chain.

Changes against td_tracks, each carrying a measured reason (2026-08-29):
  * The present command has dedicated cells instead of riding at the
    last tap of a motor track's code — recovering it through
    argmax(l2 id) -> argmax(l1 id) capped steering at corr 0.43
    (graded: 0.62) even when layer 3 pointed at the right code.
  * The motor-history tracks are gone (user's call): the thrusters
    have no dynamics and the teacher is a pure function of state, so
    past commands carry nothing the sensor windows don't. Their
    measured help on collective (0.63 -> 0.82) was a smoothness prior
    that self-confirms in closed loop.
  * Train/read metric mismatch fixed: td_tracks selected learning
    winners on five tracks but cued on three.
  * Command encoding is td3's measured fix: collective + differential
    on foveated axes, never a difference of two rotor reads.

Template counts are FIXED (user's call): 64 / 128 per track, 8192 top.

Staged training (the freeze law): track L1s learn first, then L2s on
frozen L1s, then L3 alone on frozen tracks — a memory of codes whose
meaning keeps moving underneath it rots silently.
"""

import numpy as np

import sl_drone as W
import td_tracks as B
import td3_stack as M

SENSORS = ("dx", "dy", "tilt")
NS = len(SENSORS)
NCMD = M.NB_M                       # 48 cells per command channel
SHARE_CMD = 0.25                    # command's energy share, split over 2

HOVER_LV = (W.NLEV // 2, W.NLEV // 2)
EMPTY = (np.zeros(0, np.int64), np.zeros(0, np.float32))


class Stack:
    """Three sensor tracks and one bound top layer.

    stage 0: track L1s learn.  stage 1: track L2s learn (L1 frozen).
    stage 2: L3 learns alone (tracks frozen).  frozen=True stops all.
    """

    def __init__(self, k1=64, k2=128, k3=8192, topm=8, eta=0.05, seed=0,
                 cmd_share=SHARE_CMD, ns=NS, pres_share=0.0):
        self.ns = ns
        self.tr = [B.Track(k1, k2, topm, seed=seed + 7 * c)
                   for c in range(ns)]
        self.k2, self.topm = k2, topm
        self.cmd_share = float(cmd_share)
        # [present ; lagged keys], sensor edition: the present sense
        # values get their own bump cells in the metric. Without them
        # the current tick is ~3% of the cue's energy (one tap of five,
        # one code of seven) and the read follows the HISTORY — a
        # feedback delay measured as drift/oscillation (2026-08-30).
        self.pres = float(pres_share)
        self.dim_pres = ns * B.NB1 if self.pres > 0 else 0
        self.dim_sensor = self.dim_pres + ns * k2
        self.dim = self.dim_sensor + 2 * NCMD
        self.l3 = B.Hyper(k3, self.dim, eta=eta, seed=seed + 99)
        self.stage = 2
        self.frozen = False
        self.every = 5
        self.hist = np.zeros((B.HIST, ns), np.float32)
        self.rings = [[EMPTY] * B.RING for _ in range(ns)]
        self.n = 0
        self._codes = None

    def reset(self):
        self.hist[:] = 0.0
        self.rings = [[EMPTY] * B.RING for _ in range(self.ns)]
        self.n = 0
        self._codes = None

    def push(self, sv):
        """Advance every track one tick and cache the three L2 codes."""
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = sv
        self.n += 1
        if self.n < B.HIST:
            self._codes = {}          # present cells can still cue
            return
        teach = (not self.frozen) and (self.n % self.every == 0)
        idx = B.HIST - 1 - (B.T1 - 1 - np.arange(B.T1)) * B.ST1
        codes = {}
        for c in range(self.ns):
            i, v = self.tr[c].enc1(self.hist[idx, c])
            if teach and self.stage == 0:
                self.tr[c].l1.learn(i, v)
            self.rings[c] = self.rings[c][1:] + [
                B.graded(self.tr[c].l1, i, v, self.topm)]
            taps = [self.rings[c][-1 - (B.T2 - 1 - t) * B.ST2]
                    for t in range(B.T2)]
            i2, v2 = self.tr[c].enc2(taps)
            if len(i2) == 0:
                codes[c] = EMPTY
                continue
            if teach and self.stage == 1:
                self.tr[c].l2.learn(i2, v2)
            codes[c] = B.graded(self.tr[c].l2, i2, v2, self.topm)
        self._codes = codes

    def _sensor_cells(self):
        cells, vals = [], []
        if self.pres > 0:
            share = self.pres / self.ns
            for c in range(self.ns):
                k, w = B.bump(self.hist[-1, c])
                nrm = float(np.linalg.norm(w))
                if nrm < 1e-9:
                    continue
                cells.append(c * B.NB1 + k)
                vals.append(w * (np.sqrt(share) / nrm))
        share = (1.0 - self.cmd_share - self.pres) / self.ns
        for c in range(self.ns):
            ci, cv = self._codes.get(c, EMPTY)
            if len(ci) == 0:
                continue
            nrm = float(np.linalg.norm(cv))
            if nrm < 1e-9:
                continue
            cells.append(self.dim_pres + c * self.k2 + ci)
            vals.append(cv * (np.sqrt(share) / nrm))
        if not cells:
            return EMPTY
        return (np.concatenate(cells).astype(np.int64),
                np.concatenate(vals).astype(np.float32))

    def act(self):
        """Partial-cue read: sensor cells in, command cells out.

        Returns (levels, (collective, differential) in newtons, winner
        index or -1 when no read happened).
        """
        if self._codes is None or self.l3.n == 0:
            return HOVER_LV, (W.HOVER, 0.0), -1
        i, v = self._sensor_cells()
        if len(i) == 0:
            return HOVER_LV, (W.HOVER, 0.0), -1
        s = self.l3.scores(i, v)
        j = int(np.argmax(s))
        row = self.l3.W[j]
        pc = np.clip(row[self.dim_sensor:self.dim_sensor + NCMD], 0, None)
        pd = np.clip(row[self.dim_sensor + NCMD:], 0, None)
        c = B.sharpen(pc, M.COLL_C) if pc.max() > 0 else W.HOVER
        d = B.sharpen(pd, M.DIFF_C) if pd.max() > 0 else 0.0
        return M.cd_to_levels(c, d), (c, d), j

    def learn(self, lv):
        """Ordinary full-vector learning: [sensor codes ; command bumps]."""
        if (self._codes is None or self.stage != 2 or self.frozen
                or self.n % self.every):
            return
        i, v = self._sensor_cells()
        if len(i) == 0:
            return
        c, d = M.levels_to_cd(lv)
        kc, wc = M.bump_idx(M.COLL_C, c)
        kd, wd = M.bump_idx(M.DIFF_C, d)
        half = np.sqrt(self.cmd_share / 2.0)
        cells = np.concatenate([
            i, self.dim_sensor + kc, self.dim_sensor + NCMD + kd])
        vals = np.concatenate([
            v, wc * (half / (np.linalg.norm(wc) + 1e-9)),
            wd * (half / (np.linalg.norm(wd) + 1e-9))]).astype(np.float32)
        self.l3.learn(cells, vals)


def load_stack(npz, cfg):
    """Rebuild a frozen Stack from a checkpoint."""
    st = Stack(k1=int(cfg["k1"]), k2=int(cfg["k2"]), k3=int(cfg["k3"]),
               topm=int(cfg["topm"]), seed=int(cfg.get("seed", 0)),
               cmd_share=float(cfg.get("cmd_share", SHARE_CMD)),
               ns=int(cfg.get("ns", NS)),
               pres_share=float(cfg.get("pres", 0.0)))
    for c in range(st.ns):
        st.tr[c].l1.W = npz[f"t{c}l1"]
        st.tr[c].l1.n = int(npz[f"t{c}n1"])
        st.tr[c].l2.W = npz[f"t{c}l2"]
        st.tr[c].l2.n = int(npz[f"t{c}n2"])
    st.l3.W = npz["l3"]
    st.l3.n = int(npz["l3n"])
    st.frozen = True
    return st


def load_policy(npz, cfg):
    """Hook for the shared viewer at the repo root.

    The viewer teleports the drone (Reset) and swaps targets (click)
    without telling the policy, so the policy watches for its own
    discontinuities: on the first call, a target change, or a position
    jump one tick of physics cannot produce, the track history is
    cleared. The pupil is then blind — plain hover — for the ~1 s the
    windows take to refill; that blindness is real, the memory reads
    240-720 ms windows and can say nothing without them.
    """
    st = load_stack(npz, cfg)
    prev = {"s": None, "tgt": None}

    def act(s, tgt, lv_prev):
        if (prev["s"] is None
                or abs(tgt[0] - prev["tgt"][0]) > 1e-9
                or abs(tgt[1] - prev["tgt"][1]) > 1e-9
                or np.hypot(s[0] - prev["s"][0], s[1] - prev["s"][1]) > 1.0):
            st.reset()
        prev["s"] = (float(s[0]), float(s[1]))
        prev["tgt"] = (float(tgt[0]), float(tgt[1]))
        st.push(B.sense(s, tgt))
        lv, _, _ = st.act()
        return lv
    return act
