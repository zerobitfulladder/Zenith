"""The cascade chaser: small, fast, two loops, motor vocabularies.

    OUTER (10 Hz): position error -> velocity command.
        key   = present bumps of (dx, dy), 48 dims. K = 256 rows.
        vocab = 32 prototypes over the (vcx, vcy) command code.
    INNER (50 Hz): track the command, stay level.
        key   = [now | -4 | -16] slices of (vx-vcx, vy-vcy, tilt,
                gyro), 288 dims — the pole's delay-line recipe; the
                velocity ERROR makes the loop command-invariant.
                K = 1024 rows.
        vocab = 32 prototypes over the (coll, diff) motor code.

    Both stages: sensory-only rows (the split law), a cetele beside
    each row accumulating the LABEL'S BANK CODE (pole-style mean
    tally), hard read = argmax vocabulary unit -> decode prototype.
    Mode over coarse action classes, not over fine levels — the pole's
    granularity lesson.

    Teacher decomposes exactly: outer label = the brake-limited
    velocity profile; inner label = the attitude cascade GIVEN the
    commanded velocity — so during pupil episodes the inner labels are
    on-policy (relative to the pupil's own command) for free.

Run:  .venv/bin/python experiments/2026_08_29/cascade/gpu_cascade.py
Env:  GD_B GD_TICKS GD_BANKS GD_FREEZE GD_PHASE2 GD_REPORT GD_TAG
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chase"))
import gpu_chase as G                                   # noqa: E402
from gpu_chase import (xp, asnumpy, physics, at_goal,   # noqa: E402
                       wrap, LEVELS, bump, Layer, init_worlds,
                       NB, CEN, HOVER, TMAX, NLEV, EP_CAP,
                       POS_P, AMAX, KVX, KVY, KA, KDA, PHIMAX, DMAX)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / os.environ.get('GD_TAG', '').lstrip("_")
B = int(os.environ.get("GD_B", "128"))
TICKS = int(os.environ.get("GD_TICKS", "1600000"))
BANKS = int(os.environ.get("GD_BANKS", "30000"))       # banks learn to here
FREEZE = int(os.environ.get("GD_FREEZE", "100000"))    # tops freeze here
PHASE2 = int(os.environ.get("GD_PHASE2", "250000"))    # pupil episodes from
REPORT = int(os.environ.get("GD_REPORT", "160000"))
K_OUT, K_IN, K_VC, K_MB = 256, 1024, 32, 32
LAGS = [4, 16]
MAXLAG = max(LAGS)
D_OUT = 2 * NB
D_IN = 4 * (1 + len(LAGS)) * NB
OUTER_EVERY = 5
# GD_ORACLE=pid: the tuned viewer PID, decomposed into cascade form —
# outer v_des = clip(1.178*err), inner ax = KD*(v_des - v) etc.
ORACLE = os.environ.get("GD_ORACLE", "fast")
KPD, KDD, KT_P = 1.178, 4.084, 0.071
KAP_P, KAD_P, PHI_P, VMAXC = 6.426, 0.943, 0.696, 8.0
GRAV = 9.8


def fov(v, scale, rim):
    den = float(np.arcsinh(rim / scale))
    return xp.clip(xp.arcsinh(v / scale) / den, -1, 1).astype(xp.float32)


def vcmd(e):
    if ORACLE == "pid":
        return xp.clip(-KPD * e, -VMAXC, VMAXC)
    a = xp.abs(e)
    d = AMAX / (POS_P * POS_P)
    v = xp.where(a <= d, POS_P * a,
                 xp.sqrt(xp.maximum(2 * AMAX * (a - 0.5 * d), 0)))
    return -xp.sign(e) * v


def inner_label(s, vc):
    """The teacher's attitude cascade, GIVEN a velocity command."""
    ph = wrap(s[:, 4])
    if ORACLE == "pid":
        axd = KDD * (vc[:, 0] - s[:, 2])
        ayd = KDD * (vc[:, 1] - s[:, 3])
        phd = xp.clip(-KT_P * axd, -PHI_P, PHI_P)
        coll = xp.clip(0.5 * (GRAV + ayd)
                       / xp.maximum(xp.cos(ph), 0.5), 0.3, TMAX)
        diff = xp.clip(KAP_P * (phd - ph) - KAD_P * s[:, 5],
                       -2.5, 2.5)
        return coll, diff
    phi_des = xp.clip(KVX * (s[:, 2] - vc[:, 0]), -PHIMAX, PHIMAX)
    coll = (HOVER + KVY * (vc[:, 1] - s[:, 3])) / xp.maximum(
        xp.cos(ph), 0.5)
    coll = xp.clip(coll, 0.3, TMAX)
    diff = xp.clip(KA * (phi_des - ph) - KDA * s[:, 5], -DMAX, DMAX)
    return coll, diff


def snap_levels(coll, diff):
    t1 = xp.clip(coll - diff, 0, TMAX)
    t2 = xp.clip(coll + diff, 0, TMAX)
    return (xp.argmin(xp.abs(LEVELS[None] - t1[:, None]), axis=1)
            .astype(xp.int32),
            xp.argmin(xp.abs(LEVELS[None] - t2[:, None]), axis=1)
            .astype(xp.int32))


def code_vc(vc):
    return bump(fov(vc, 0.2, 8.0)).reshape(vc.shape[0], D_OUT)


def code_motor(coll, diff):
    v = xp.stack([fov(coll - HOVER, 0.15, 3.2),
                  fov(diff, 0.05, 2.5)], axis=1)
    return bump(v).reshape(coll.shape[0], D_OUT)


def decode_bank(row, scale_a, rim_a, scale_b, rim_b):
    """Prototype row -> the two values it encodes (centroid, unwarp)."""
    r = np.clip(asnumpy(row), 0, None).reshape(2, NB)
    cen = asnumpy(CEN)
    out = []
    for h, (sc, rim) in zip(r, ((scale_a, rim_a), (scale_b, rim_b))):
        u = float((h * cen).sum() / h.sum()) if h.sum() > 0 else 0.0
        out.append(sc * np.sinh(u * np.arcsinh(rim / sc)))
    return out


class Stage:
    """Sensory rows + a cetele of bank codes + a vocabulary bank."""

    def __init__(self, k, dim, kb, seed):
        self.top = Layer(k, dim, seed=seed)
        self.bank = Layer(kb, D_OUT, seed=seed + 1)
        self.C = xp.zeros((k, kb), xp.float32)
        self.cnt = xp.zeros(k, xp.float32)

    def winners(self, key):
        sc = self.top.scores(key)
        if sc.shape[1] == 0:
            return xp.full(key.shape[0], -1, xp.int64)
        return xp.argmax(sc, axis=1)

    def bank_code(self, label_code):
        sc = self.bank.scores(label_code)
        if sc.shape[1] == 0:
            return xp.zeros((label_code.shape[0], self.bank.k),
                            xp.float32)
        g = xp.maximum(sc, 0)
        out = xp.zeros((label_code.shape[0], self.bank.k), xp.float32)
        out[:, :g.shape[1]] = g
        return out

    def notch(self, w, label_code, mask):
        """Hard assignment: count the NEAREST prototype. The soft
        version (accumulating relu-cosine profiles) smears the tally
        and the duty-mean read then over-smooths — the balance cetele
        that worked counted exact choices."""
        sc = self.bank.scores(label_code)
        if sc.shape[1] == 0:
            return
        u = xp.argmax(sc, axis=1)
        v = mask & (w >= 0)
        if not bool(v.any()):
            return
        idx = xp.clip(w, 0, self.C.shape[0] - 1)
        flat = idx * self.C.shape[1] + u
        if xp is np:
            np.add.at(self.C.reshape(-1), asnumpy(flat[v]), 1.0)
        else:
            import cupyx
            cupyx.scatter_add(self.C.reshape(-1), flat[v], 1.0)

    def read_mean(self, w, unit_vals):
        """Duty-weighted mean of the decoded prototypes — mean for
        regulation (the bounded law); the coarse vocabulary supplies
        the decisiveness, the duty mix the fine control."""
        rows = self.C[xp.clip(w, 0, self.C.shape[0] - 1)]
        tot = rows.sum(axis=1)
        ok = (tot > 0) & (w >= 0)
        wts = rows / xp.maximum(tot, 1)[:, None]
        return wts @ xp.asarray(unit_vals), ok


def outer_key(s, tgt):
    v = xp.stack([fov(s[:, 0] - tgt[:, 0], 0.25, 30.0),
                  fov(s[:, 1] - tgt[:, 1], 0.25, 30.0)], axis=1)
    return bump(v).reshape(s.shape[0], D_OUT)


class Casc:
    def __init__(self, seed=0):
        self.outer = Stage(K_OUT, D_OUT, K_VC, seed)
        self.inner = Stage(K_IN, D_IN, K_MB, seed + 50)
        self.buf = xp.zeros((B, MAXLAG + 1, 4), xp.float32)
        self.vc = xp.zeros((B, 2), xp.float32)
        self.vc_units = None    # decoded (K_VC, 2) after banks freeze
        self.mb_vals = None     # decoded (K_MB, 2)

    def freeze_decode(self):
        self.vc_units = np.array(
            [decode_bank(self.outer.bank.W[u], 0.2, 8.0, 0.2, 8.0)
             for u in range(K_VC)], np.float32)
        self.mb_vals = np.array(
            [decode_bank(self.inner.bank.W[u], 0.15, 3.2, 0.05, 2.5)
             for u in range(K_MB)], np.float32)
        self.mb_vals[:, 0] += HOVER

    def inner_key(self, s):
        # buffer RAW state; errors for ALL slices are measured against
        # the CURRENT command (the reference bug, fourth appearance:
        # mixed references inside one window — the lag slices used to
        # carry errors against commands that were since replaced)
        self.buf = xp.roll(self.buf, -1, axis=1)
        self.buf[:, -1] = xp.stack([s[:, 2], s[:, 3],
                                    wrap(s[:, 4]), s[:, 5]], axis=1)
        # the present carries HALF the key's energy (route-1 law);
        # the two lags share the rest as context
        blocks = []
        for sl, share in ((self.buf[:, -1], 0.5),
                          (self.buf[:, -1 - LAGS[0]], 0.25),
                          (self.buf[:, -1 - LAGS[1]], 0.25)):
            ch = xp.stack([fov(sl[:, 0] - self.vc[:, 0], 0.15, 20.0),
                           fov(sl[:, 1] - self.vc[:, 1], 0.15, 20.0),
                           fov(sl[:, 2], 0.05, np.pi),
                           fov(sl[:, 3], 0.1, 10.0)], axis=1)
            b = bump(ch).reshape(B, 4 * NB)
            n = xp.linalg.norm(b, axis=1, keepdims=True) + 1e-9
            blocks.append(b / n * np.sqrt(share))
        return xp.concatenate(blocks, axis=1)

    def act_outer(self, s, tgt):
        w = self.outer.winners(outer_key(s, tgt))
        vcs, ok = self.outer.read_mean(w, self.vc_units)
        return xp.where(ok[:, None], vcs, 0), w

    def act_inner(self, key):
        w = self.inner.winners(key)
        mv, ok = self.inner.read_mean(w, self.mb_vals)
        coll = xp.where(ok, mv[:, 0], HOVER)
        diff = xp.where(ok, mv[:, 1], 0.0)
        return snap_levels(coll, diff), w


def evaluate(casc, rng, po=True, pi=True):
    sb, sv = casc.buf.copy(), casc.vc.copy()
    s, tgt = init_worlds(rng, B)
    casc.buf[:] = 0
    casc.vc[:] = 0
    hold = xp.zeros(B, xp.int32)
    done = xp.zeros(B, bool)
    best = xp.full(B, 1e9, xp.float32)
    for t in range(EP_CAP):
        if t % OUTER_EVERY == 0:
            if po:
                casc.vc, _ = casc.act_outer(s, tgt)
            else:
                casc.vc = xp.stack([vcmd(s[:, 0] - tgt[:, 0]),
                                    vcmd(s[:, 1] - tgt[:, 1])], axis=1)
        key = casc.inner_key(s)
        if pi:
            (li, ri), _ = casc.act_inner(key)
        else:
            li, ri = snap_levels(*inner_label(s, casc.vc))
        s = physics(s, li, ri)
        best = xp.minimum(best, xp.hypot(s[:, 0] - tgt[:, 0],
                                         s[:, 1] - tgt[:, 1]))
        g = at_goal(s, tgt)
        hold = xp.where(g, hold + 1, 0)
        done = done | (hold >= 10)
    casc.buf, casc.vc = sb, sv
    return float(done.mean()), float(best.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    casc = Casc()
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(99)
    s, tgt = init_worlds(rng, B)
    ep_tick = np.zeros(B, np.int64)
    pupil = np.zeros(B, bool)
    episodes, agree, stats = 0, [], []
    best = {"score": -1.0, "closest": 99.0, "tick": 0}
    t0 = time.time()

    for step in range(TICKS // B):
        te = step * B
        lm = (np.arange(B) + step) % 5 == 0
        # ---- outer, at 10 Hz -----------------------------------------
        if step % OUTER_EVERY == 0:
            okey = outer_key(s, tgt)
            lab_vc = xp.stack([vcmd(s[:, 0] - tgt[:, 0]),
                               vcmd(s[:, 1] - tgt[:, 1])], axis=1)
            if te < BANKS:
                casc.outer.bank.learn_rows(code_vc(lab_vc)[xp.asarray(lm)])
            if te < FREEZE:
                casc.outer.top.learn_rows(okey[xp.asarray(lm)])
            wo = casc.outer.winners(okey)
            if te >= FREEZE:      # notch only on frozen rows
                casc.outer.notch(wo, code_vc(lab_vc),
                                 xp.ones(B, bool))
            # the outer is already at 1.00 — during training the
            # teacher always commands, so pupil episodes train the
            # INNER loop alone on its own closed-loop states
            casc.vc = lab_vc
        # ---- inner, every tick ---------------------------------------
        key = casc.inner_key(s)
        lc, ld = inner_label(s, casc.vc)
        if te < BANKS:
            casc.inner.bank.learn_rows(code_motor(lc, ld)[xp.asarray(lm)])
        if te < FREEZE:
            casc.inner.top.learn_rows(key[xp.asarray(lm)])
        if te == BANKS or (casc.vc_units is None and te > BANKS):
            casc.freeze_decode()
        wi = casc.inner.winners(key)
        if te >= FREEZE:          # notch only on frozen rows
            casc.inner.notch(wi, code_motor(lc, ld), xp.ones(B, bool))
        tl, tr = snap_levels(lc, ld)
        if te >= PHASE2:
            (pl, pr), _ = casc.act_inner(key)
            pm = xp.asarray(pupil)
            fl = xp.where(pm, pl, tl)
            fr = xp.where(pm, pr, tr)
            agree.append(float(((pl == tl) & (pr == tr)).mean()))
        else:
            fl, fr = tl, tr
        s = physics(s, fl, fr)
        ep_tick += 1
        spd = asnumpy(xp.hypot(s[:, 2], s[:, 3]))
        dist = asnumpy(xp.hypot(s[:, 0] - tgt[:, 0],
                                s[:, 1] - tgt[:, 1]))
        blown = pupil & ((spd > 8) | (np.abs(asnumpy(s[:, 5])) > 6)
                         | (dist > 15))
        over = (ep_tick >= EP_CAP) | blown
        if over.any():
            episodes += int(over.sum())
            ns, nt = init_worlds(rng, int(over.sum()))
            om = xp.asarray(over)
            s[om] = ns
            tgt[om] = nt
            casc.buf[om] = 0
            casc.vc[om] = 0
            ep_tick[over] = 0
            pupil[over] = (te >= PHASE2) & (np.random.default_rng(
                episodes).random(int(over.sum())) < 0.5)

        if (step + 1) % max(1, REPORT // B) == 0:
            if casc.vc_units is None:
                casc.freeze_decode()
            ev, near = evaluate(casc, erng)
            ev_o, near_o = evaluate(casc, erng, po=True, pi=False)
            ev_i, near_i = evaluate(casc, erng, po=False, pi=True)
            row = {"exp_ticks": te + B, "EVAL": round(ev, 3),
                   "closest": round(near, 2),
                   "outer_only": [round(ev_o, 2), round(near_o, 2)],
                   "inner_only": [round(ev_i, 2), round(near_i, 2)],
                   "agree": round(float(np.mean(agree)), 3)
                   if agree else None,
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree = []
            np.savez(OUT / "checkpoint.npz",
                     Wo=asnumpy(casc.outer.top.W),
                     Wi=asnumpy(casc.inner.top.W),
                     Bo=asnumpy(casc.outer.bank.W),
                     Bi=asnumpy(casc.inner.bank.W),
                     Co=asnumpy(casc.outer.C), Ci=asnumpy(casc.inner.C))
            if (ev, -near) > (best["score"], -best["closest"]):
                best = {"score": ev, "closest": near, "tick": te + B}
                import shutil
                shutil.copy(OUT / "checkpoint.npz",
                            OUT / "weights_best.npz")
                print(f"  new best {ev} (closest {near:.2f})", flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "reports": stats}, f, indent=2)

    print(f"done — best {best['score']} closest {best['closest']:.2f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
