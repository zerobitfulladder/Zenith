"""The vision pupil for the viewer: load_policy(npz, cfg) -> fn(state, target, lv_prev) -> (l, r).
It renders the egocentric frame itself, so the viewer only hands it the state."""
import numpy as np
import cupy as cp
import box_world as B
import vrig as V


def load_policy(npz, cfg):
    if cfg.get("read") == "l1":
        return _l1_policy(npz, cfg)
    if cfg.get("read") == "l1joint":
        return _l1joint_policy(npz, cfg)
    if cfg.get("read") == "rl":
        return _rl_policy(npz, cfg)
    rig = V.VRig(48, int(cfg["ps"]), int(cfg["frames"]), int(cfg["grid"]), int(cfg["k1"]))
    W1, W2 = cp.asarray(npz["W1"]), cp.asarray(npz["W2"])
    mu, sd = cp.asarray(npz["mu"]), cp.asarray(npz["sd"])
    TJ, TL, TR = cp.asarray(npz["TJ"]), cp.asarray(npz["TL"]), cp.asarray(npz["TR"])
    NL = cp.asarray(npz["NL"]) if "NL" in npz else None
    NR = cp.asarray(npz["NR"]) if "NR" in npz else None
    mode = cfg.get("read", "marginal")
    if mode == "mean" and NL is None:
        mode = "marginal"
    mem = {"prev": None, "hist": []}
    track = bool(cfg.get("track", False))
    if track:
        tmu, tsd, tw = cp.asarray(npz["tmu"]), cp.asarray(npz["tsd"]), float(cfg["track_w"])

    def act(s, tgt, lv_prev):
        fr = B.render(s, tgt).astype(np.float32)
        if rig.C == 2:
            prev = fr if mem["prev"] is None else mem["prev"]
            d = (fr - prev) if cfg.get("diff") == "signed" else 0.5 + 0.5 * (fr - prev)
            x = np.stack([fr, d], 0); mem["prev"] = fr
        else:
            x = fr[None]
        C = V.encode_contrast(rig, W1, cp.asarray(x.reshape(1, -1)))
        if track:
            mem["hist"].append(np.asarray(s[2:6], np.float32))
            h = mem["hist"]
            feats = np.concatenate([h[max(0, len(h) - 1 - lag)] for lag in V.TRACK_LAGS])[None]
            K = cp.asarray(V.encode_track(feats))
            C = cp.concatenate([(C - mu) / sd, tw * (K - tmu) / tsd], 1)
            C /= cp.linalg.norm(C, axis=1, keepdims=True) + V.EPS
        else:
            C = V.standardize(C, mu, sd)
        if mode == "colldiff":
            w0 = int((C @ W2.T).argmax(1)[0]); NC = NL.shape[1]
            lv = cp.arange(NC, dtype=cp.float32)
            mc = float(((NL[w0] + 1) * lv).sum() / (NL[w0] + 1).sum())
            md = float(((NR[w0] + 1) * lv).sum() / (NR[w0] + 1).sum()) - (V.NLEV - 1)
            return int(np.clip(round((mc + md) / 2), 0, V.NLEV - 1)), int(np.clip(round((mc - md) / 2), 0, V.NLEV - 1))
        l, r = V.read(W2, TJ, TL, TR, C, mode=mode, NL=NL, NR=NR)
        return int(l[0]), int(r[0])
    act.mem = mem
    return act


def reset(act):
    act.mem["prev"] = None; act.mem["hist"] = []


def _l1_policy(npz, cfg):
    """No object layer: stroke winners per cell + track bins -> two tallies, mean read."""
    rig = V.VRig(48, int(cfg["ps"]), 1, int(cfg["grid"]), int(cfg["k1"]))
    W1 = cp.asarray(npz["W1"])
    T = {k: cp.asarray(npz[k]) for k in ("TiL", "TtL", "TiR", "TtR")}
    scale = float(cfg["trackw"]) * float(npz["img_votes"]) / float(npz["trk_votes"])
    NCH, NB, NLEV = 4 * len(V.TRACK_LAGS), V.TRACK_NB, V.NLEV
    mem = {"prev": None, "hist": []}

    def act(s, tgt, lv_prev):
        fr = B.render(s, tgt).astype(np.float32)
        Q, keep = rig.patches(cp.asarray(fr.reshape(1, -1)))
        idx = ((Q.reshape(-1, rig.dim) @ W1.T) ** 2).argmax(1); k = keep.reshape(-1)
        mem["hist"].append(np.asarray(s[2:6], np.float32)); h = mem["hist"]
        feats = np.concatenate([h[max(0, len(h) - 1 - lag)] for lag in V.TRACK_LAGS])[None]
        Kt = cp.asarray(V.encode_track(feats)).reshape(NCH, NB)
        out = []
        for Ti, Tt in ((T["TiL"], T["TtL"]), (T["TiR"], T["TtR"])):
            ev = (Ti[idx, rig.cells] * k[:, None]).sum(0) + scale * cp.einsum('cb,bcl->l', Kt, Tt)
            z = ev - ev.max(); p = cp.exp(z); p /= p.sum()
            out.append(int(cp.rint((p * cp.arange(NLEV)).sum())))
        return out[0], out[1]
    act.mem = mem
    return act


def _l1joint_policy(npz, cfg):
    """One tally over the command pair; read = pair mean per motor."""
    rig = V.VRig(48, int(cfg["ps"]), 1, int(cfg["grid"]), int(cfg["k1"]))
    W1, Ti, Tt = cp.asarray(npz["W1"]), cp.asarray(npz["Ti"]), cp.asarray(npz["Tt"])
    scale = float(cfg["trackw"]) * float(npz["img_votes"]) / float(npz["trk_votes"])
    NCH, NB, NLEV = 4 * len(V.TRACK_LAGS), V.TRACK_NB, V.NLEV
    mem = {"prev": None, "hist": []}

    def act(s, tgt, lv_prev):
        fr = B.render(s, tgt).astype(np.float32)
        Q, keep = rig.patches(cp.asarray(fr.reshape(1, -1)))
        idx = ((Q.reshape(-1, rig.dim) @ W1.T) ** 2).argmax(1); k = keep.reshape(-1)
        mem["hist"].append(np.asarray(s[2:6], np.float32)); h = mem["hist"]
        feats = np.concatenate([h[max(0, len(h) - 1 - lag)] for lag in V.TRACK_LAGS])[None]
        Kt = cp.asarray(V.encode_track(feats)).reshape(NCH, NB)
        ev = (Ti[idx, rig.cells] * k[:, None]).sum(0) + scale * cp.einsum('cb,bcl->l', Kt, Tt)
        p = cp.exp(ev - ev.max()); p /= p.sum(); P3 = p.reshape(NLEV, NLEV); lv = cp.arange(NLEV, dtype=cp.float32)
        return int(cp.rint((P3.sum(1) * lv).sum())), int(cp.rint((P3.sum(0) * lv).sum()))
    act.mem = mem
    return act


def _rl_policy(npz, cfg):
    """The reward-trained single layer over sensory tracks: nearest template, greedy on Q."""
    import novision as NV, rl_tracks as RL
    W, live, Q = npz["W"], npz["live"], npz["Q"]
    Wl = W[live]; Ql = Q[live]
    mem = {"prev": None, "hist": []}

    def act(s, tgt, lv_prev):
        mem["hist"].append(NV.raw(s, tgt))
        x = RL.features(mem["hist"]); x = x / (np.linalg.norm(x) + 1e-9)
        t = int(np.argmax((Wl @ x) ** 2))
        return RL.levels(int(np.argmax(Ql[t])))
    act.mem = mem
    return act
