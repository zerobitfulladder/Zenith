"""The tally sets the step. No floor, no rank, no schedule.

`../plasticity_rf` froze templates by importance RANK, from batch 1, and lost:
the freeze fired on templates that were still noise, and the frozen templates
kept winning everything while learning nothing. This replaces the rank with the
absolute number the tally already holds. A template's step is the uncertainty
of its own class row:

    P(y|t)  = (N[t,y] + 1) / (sum_y N[t,y] + NL)          smoothed, as in the table
    eta_t   = (1 - max_y P(y|t)) / (1 - 1/NL)              empty row -> 1, pure row -> 0

read from the row as it stands BEFORE the batch is counted, so a fresh template
takes its first batch at full step (the champion's cnt/n does the same: m/(n+m)).

For a pure row with n wins that is 9/(n+10): the champion's 1/n annealing,
except it is earned by COMMITMENT, not by age. A template that has won a
thousand mixed images stays fully plastic; one that has won three hundred of a
single class is nearly still. Nothing is clamped.

Prediction from the sink argument: at beta 0 the committed templates still win
the intruders, refuse to move for them, but COUNT them, so their purity erodes
and they unlock. So the competition is also asked to consult the same row, with
the label in place of the belief:

    none     win = argmin(err)
    belief   win = argmin(err - beta * q . T[t,c,:])       the champion's routing
    label    win = argmin(err - beta * T[t,c,y])           q = one-hot(label)

2 step rules x 3 competitions, 28x28 split 0-4 -> 5-9 -> 0-9, 3 seeds. Through
the 5-9 phase the templates committed at the end of 0-4 are followed: their row
purity, how far their shapes move, and what share of the wins they take.

Usage:  uv run python purity_split.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import rf_sweep as R

OUT = HERE / "results"
SMOKE = "--smoke" in sys.argv
PS = 28
R.BATCH = 512
EPOCHS = 2 if SMOKE else 20
SEEDS = [7] if SMOKE else [7, 8, 9]
BETA = 1.5
BETA_L, BETA_B = BETA, BETA          # label and belief strengths; route "both" uses both
CHASE, CHASE_ETA = False, 0.5        # dead templates chase the patch they are nearest to
CHASE_MODE = "all"                   # "all": every patch; "err": weighted by the live winner's error
HIRE = False                         # misread image with a committed winner -> cheapest template takes it
SOFT, TAU = None, 0.1                # "hebb": losers move toward the input by softmax activity; "anti": away
LOO, LOO_RIDGE = False, 1e-3         # winner by PARTIAL correlation: the input's correlation with template t
                                     # after everything the other templates explain is removed (Gram inverse)


def loo_err(V, W):
    G = W @ W.T + LOO_RIDGE * cp.eye(W.shape[0], dtype=W.dtype)
    Ginv = cp.linalg.inv(G)
    beta = (V @ W.T) @ Ginv
    pc = beta / cp.sqrt(cp.maximum(cp.diag(Ginv), 1e-9))[None, :]
    return 1.0 - pc ** 2


def code_any(rig, W, X, chunk=256):
    """rig.code, or the partial-correlation winner when LOO is on (train and read must agree)."""
    if not LOO:
        return rig.code(W, X)
    idx = cp.zeros((len(X), rig.npos), cp.int32); keepm = cp.zeros((len(X), rig.npos), bool)
    for a in range(0, len(X), chunk):
        Q, keep = rig.patches(X[a:a + chunk]); m = len(Q)
        idx[a:a + m] = loo_err(Q.reshape(-1, rig.dim), W).argmin(1).reshape(m, -1)
        keepm[a:a + m] = keep
    return idx, keepm
PROBE_EVERY = 10
COMMIT = 0.9                      # row purity at the end of 0-4 that counts as committed
K, NL = R.K, R.NL
RULES = ["cntn", "purity"]
ROUTES = ["none", "belief", "label"]


def rows(N):
    """Class row per template, summed over cells (one cell at 28x28)."""
    return N.sum(1)


def purity(N):
    """Per-(template, cell) row purity, averaged over cells weighted by the
    template's count at each cell. One cell at 28x28, so identical to the
    plain row there; on patches a template that is pure at every position but
    mixed overall counts as committed, because its per-cell table entries are
    informative and moving it would invalidate them."""
    tot = N.sum(2)                                            # (K, gg)
    pc = (N + R.ALPHA) / (tot[..., None] + R.ALPHA * NL)
    unc_c = 1.0 - pc.max(2)                                   # (K, gg)
    ws = tot.sum(1)
    unc = cp.where(ws > 0, (tot * unc_c).sum(1) / cp.maximum(ws, 1e-9), 1.0 - 1.0 / NL)
    return 1.0 - unc


def eta_purity(N):
    return ((1.0 - purity(N)) / (1.0 - 1.0 / NL)).astype(cp.float32)


def split_acc(rig, T, codes, y):
    win, keep = codes
    C = cp.tile(rig.cells, len(win)).reshape(len(win), rig.npos)
    pred = (T[win, C] * keep[..., None]).sum(1).argmax(1)
    old = y < 5
    return (float((pred[old] == y[old]).mean()),
            float((pred[~old] == y[~old]).mean()),
            float((pred == y).mean()))


def run(rig, rule, route, phases, Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((K, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True)
    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
    N = cp.zeros((K, rig.gg, NL)); n = cp.zeros(K)
    curve, bounds, snaps, step = [], [], {}, 0
    committed, W_A, class_A = None, None, None

    def probe(eta, cnt, pi, win, lab, k):
        itr, ktr = code_any(rig, W, Xtr)
        T_re = rig.table_from(rig.build_table(itr, ytr, ktr))
        ct = code_any(rig, W, Xte)
        rec = {"step": step, "phase": pi,
               "recount": split_acc(rig, T_re, ct, yte),
               "online": split_acc(rig, rig.table_from(N), ct, yte),
               "n_still": int((eta < R.ETA_MIN).sum()),
               "n_dead": int((N.sum((1, 2)) == 0).sum()),
               "purity_live": float(purity(N)[N.sum((1, 2)) > 0].mean())}
        if committed is not None and int(committed.sum()):
            cos = cp.abs((W[committed] * W_A[committed]).sum(1))
            conv = rows(N).argmax(1)[committed] != class_A[committed]
            kept = committed.copy(); kept[committed] = ~conv
            cos_kept = cp.abs((W[kept] * W_A[kept]).sum(1))
            rec.update({"n_committed": int(committed.sum()),
                        "converted": float(conv.mean()),
                        "drift_kept": float((1.0 - cos_kept).mean()) if int(kept.sum()) else 0.0,
                        "purity_committed": float(purity(N)[committed].mean()),
                        "drift_committed": float((1.0 - cos).mean()),
                        "win_share_committed":
                            float(cnt[committed].sum() / cp.maximum(cnt.sum(), 1))})
        return rec

    for pi, (X, y, _) in enumerate(phases):
        bounds.append(step)
        if pi == 1:
            committed = (purity(N) > COMMIT) & (N.sum((1, 2)) > 0)
            W_A = W.copy(); class_A = rows(N).argmax(1)
            snaps["A"] = (cp.asnumpy(W), cp.asnumpy(committed), cp.asnumpy(purity(N)))
        for ep in range(EPOCHS):
            order = np.arange(len(X)); rng.shuffle(order)
            for s in range(0, len(order), R.BATCH):
                ids = cp.asarray(order[s:s + R.BATCH]); m = len(ids)
                Q, keep = rig.patches(X[ids])
                V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
                cells = cp.tile(rig.cells, m)
                lab = cp.repeat(y[ids], rig.npos)
                T = rig.table_from(N)
                err = loo_err(V, W) if LOO else 1.0 - (V @ W.T) ** 2
                score = err
                if route in ("belief", "both"):
                    q = rig.belief(T, err.argmin(1), k, cells, m)
                    bias = cp.einsum('ik,ihk->ih', cp.repeat(q, rig.npos, 0),
                                     cp.ascontiguousarray(T.transpose(1, 0, 2))[cells])
                    score = score - BETA_B * bias
                if route in ("label", "both"):
                    score = score - BETA_L * T[:, cells, lab].T
                win = score.argmin(1)
                if HIRE and rig.npos == 1:
                    # the tally's own prediction for each image, from the online table
                    qh = rig.belief(T, err.argmin(1), k, cells, m)
                    pred = qh.argmax(1)
                    pur = purity(N)
                    wrong = (pred != y[ids]) & (pur[win] >= 0.5) & k
                    if int(wrong.sum()):
                        tot = float(N.sum())
                        imp = cp.zeros(K) if tot <= 0 else ((N / tot) * T).sum(axis=(1, 2))
                        imp[win] = cp.inf                       # this batch's winners are not for hire
                        for yy in cp.unique(y[ids][wrong]).tolist():
                            grp = wrong & (y[ids] == yy)
                            t = int(imp.argmin())
                            imp[t] = cp.inf
                            win[grp] = t
                            N[t] = 0; n[t] = 0                  # its old role is forgotten on purpose
                if HIRE and rig.npos > 1:
                    # parts: the image is misread by the SUM; the offending patches are those
                    # whose winner is committed at that cell to evidence against the label.
                    # Each goes to the nearest template with no opinion at that cell yet.
                    ev = (T[win, cells] * k[:, None]).reshape(m, rig.npos, NL).sum(1)
                    wrong_p = cp.repeat(ev.argmax(1) != y[ids], rig.npos) & k
                    tot_c = N.sum(2)
                    pur_c = ((N + R.ALPHA) / (tot_c[..., None] + R.ALPHA * NL)).max(2)   # (K, gg)
                    off = wrong_p & (pur_c[win, cells] >= 0.5) & (T[win, cells, lab] < 0)
                    if int(off.sum()):
                        oi = cp.where(off)[0]
                        Sh = (V[oi] @ W.T) ** 2
                        Sh = cp.where((pur_c < 0.5).T[cells[oi]], Sh, -1.0)
                        new = Sh.argmax(1)
                        ok = Sh[cp.arange(len(oi)), new] > -1.0
                        win[oi[ok]] = new[ok]
                eta_pre = eta_purity(N)          # the row BEFORE this batch is counted
                flat = (win[k] * rig.gg + cells[k]) * NL + lab[k]
                N += cp.bincount(flat, minlength=K * rig.gg * NL).reshape(N.shape)

                cnt = cp.bincount(win[k], minlength=K)
                live = cnt >= rig.min_s
                if rule == "cntn":
                    n[live] += cnt[live]
                    eta = cp.clip(cnt / cp.maximum(n, 1.0), R.ETA_MIN, 1.0).astype(cp.float32)
                else:
                    eta = eta_pre
                if int(live.sum()):
                    sums = cp.zeros((K, rig.dim), cp.float32)
                    cupyx.scatter_add(sums, win[k], V[k])
                    W[live] += eta[live][:, None] * (sums[live] / cnt[live, None] - W[live])
                    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
                if SOFT is not None:
                    # SoftHebb-style soft plasticity for the non-winners: activity = softmax of the
                    # match over templates; each loser moves toward (hebb) or away from (anti) the
                    # activity-weighted mean of the patches, sign-aligned, gated by its own step.
                    Vk = V[k]; Sk = Vk @ W.T
                    Y = cp.exp((Sk ** 2 - (Sk ** 2).max(1, keepdims=True)) / TAU)
                    Y /= Y.sum(1, keepdims=True)
                    Y[cp.arange(len(Vk)), win[k]] = 0.0                  # winners already moved
                    act = Y.sum(0)                                        # (K,)
                    Msum = (Y * cp.sign(Sk)).T @ Vk                       # (K, dim)
                    hit = act > 1e-6
                    Mbar = Msum[hit] / act[hit, None]
                    g = cp.minimum(1.0, act[hit])
                    sgn = -1.0 if SOFT == "anti" else 1.0
                    W[hit] += sgn * (eta[hit] * g)[:, None] * (Mbar - W[hit])
                    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS

                if CHASE and CHASE_MODE == "uncommitted":
                    # every non-winner is a candidate, weighted by its row uncertainty:
                    # dead -> 1, committed -> ~0. Nearest eligible template per patch moves.
                    u = eta_purity(N)                                   # (K,)
                    Vk = V[k]
                    Sc = (Vk @ W.T) ** 2 * u[None, :]
                    Sc[cp.arange(len(Vk)), win[k]] = -1.0
                    near = Sc.argmax(1)
                    sgn = cp.sign((Vk * W[near]).sum(1)); sgn[sgn == 0] = 1
                    e1 = 1.0 - (Vk * W[win[k]]).sum(1) ** 2
                    sums = cp.zeros((K, rig.dim), cp.float32)
                    cupyx.scatter_add(sums, near, Vk * (sgn * e1)[:, None])
                    cd = cp.bincount(near, weights=e1, minlength=K)
                    cn = cp.bincount(near, minlength=K)
                    hit = cd > 0
                    rate = CHASE_ETA * u[hit] * (cd[hit] / cn[hit])
                    W[hit] += rate[:, None] * (sums[hit] / cd[hit, None] - W[hit])
                    W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
                elif CHASE:
                    dead = N.sum((1, 2)) == 0
                    nd = int(dead.sum())
                    if nd:
                        Vk = V[k]
                        Sd = Vk @ W[dead].T
                        near = (Sd ** 2).argmax(1)
                        sgn = cp.sign(Sd[cp.arange(len(Vk)), near]); sgn[sgn == 0] = 1
                        if CHASE_MODE == "err":
                            e1 = 1.0 - (Vk * W[win[k]]).sum(1) ** 2        # how poorly the live winner fits
                        else:
                            e1 = cp.ones(len(Vk), cp.float32)
                        sums = cp.zeros((nd, rig.dim), cp.float32)
                        cupyx.scatter_add(sums, near, Vk * (sgn * e1)[:, None])
                        cd = cp.bincount(near, weights=e1, minlength=nd)
                        cn = cp.bincount(near, minlength=nd)
                        hit = cd > 0
                        Wd = W[dead]
                        rate = CHASE_ETA * (cd[hit] / cn[hit])                  # mean error of its patches
                        Wd[hit] += rate[:, None] * (sums[hit] / cd[hit, None] - Wd[hit])
                        W[dead] = Wd
                        W /= cp.linalg.norm(W, axis=1, keepdims=True) + R.EPS
                if step % PROBE_EVERY == 0:
                    curve.append(probe(eta, cnt, pi, win, lab, k))
                step += 1
        curve.append({**probe(eta, cnt, pi, win, lab, k), "step": step})
        if pi == 1:
            snaps["B"] = (cp.asnumpy(W), cp.asnumpy(purity(N)))
    snaps["final"] = (cp.asnumpy(W), cp.asnumpy(N.sum((1, 2))), cp.asnumpy(purity(N)))
    return curve, bounds, snaps


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    phases = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    print(f"{PS}x{PS}, {K} templates, {EPOCHS} epochs/phase, batch {R.BATCH}, "
          f"beta {BETA}, seeds {SEEDS}, committed = purity > {COMMIT}")
    print(f"phases {int(A.sum())} / {int((~A).sum())} / {len(ytr)}, test {len(yte)}\n")

    hdr = (f"{'':<16}{'after 0-4':>10}{'after 5-9: old/new':>20}{'final old/new/all':>22}"
           f"{'online':>8}{'  #com':>6}{'pur B':>7}{'drift':>7}{'wsB':>6}{'still':>6}")
    print(hdr)
    res, store = {}, {}
    for rule in RULES:
        for route in ROUTES:
            tag = f"{rule}_{route}"
            t0 = time.time()
            runs = []
            for seed in SEEDS:
                c, b, sn = run(rig, rule, route, phases, Xtr, ytr_g, Xte, yte_g, seed)
                runs.append({"seed": seed, "curve": c, "bounds": b})
                if seed == SEEDS[0]:
                    store[tag] = sn
            res[tag] = {"rule": rule, "route": route, "runs": runs}
            e = [end_points(r["curve"]) for r in runs]
            mean = lambda key: np.mean([x[key] for x in e])
            print(f"  {tag:<14}{mean('A_old'):>10.4f}"
                  f"{mean('B_old'):>10.4f}/{mean('B_new'):.4f}"
                  f"{mean('C_old'):>8.4f}/{mean('C_new'):.4f}/{mean('C_all'):.4f}"
                  f"{mean('C_online'):>8.4f}{mean('n_com'):>6.0f}{mean('pur_B'):>7.3f}"
                  f"{mean('drift_B'):>7.3f}{mean('ws_B'):>6.2f}{mean('still_C'):>6.0f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "purity_split.json").write_text(json.dumps(res, indent=1))
    draw(res, store)


def end_points(c):
    ends = {}
    for p in c:
        ends[p["phase"]] = p          # last record of each phase wins
    A, B, C = ends[0], ends[1], ends[2]
    return {"A_old": A["recount"][0],
            "B_old": B["recount"][0], "B_new": B["recount"][1],
            "C_old": C["recount"][0], "C_new": C["recount"][1], "C_all": C["recount"][2],
            "C_online": C["online"][2],
            "n_com": B.get("n_committed", 0), "pur_B": B.get("purity_committed", np.nan),
            "drift_B": B.get("drift_committed", np.nan),
            "ws_B": B.get("win_share_committed", np.nan), "still_C": C["n_still"]}


COL = {"none": "#333333", "belief": "#1b6ca8", "label": "#c1121f"}
LS = {"cntn": ":", "purity": "-"}


def mean_curve(runs, get):
    steps = [p["step"] for p in runs[0]["curve"]]
    vals = np.array([[get(p) for p in r["curve"]] for r in runs], float)
    return steps, np.nanmean(vals, 0)


def draw(res, store):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    panels = [("old classes 0-4, recount", lambda p: p["recount"][0]),
              ("new classes 5-9, recount", lambda p: p["recount"][1]),
              ("all ten, online table", lambda p: p["online"][2]),
              ("purity of templates committed after 0-4", lambda p: p.get("purity_committed", np.nan)),
              ("drift of committed templates  (1 - |cos| from their 0-4 shape)",
               lambda p: p.get("drift_committed", np.nan)),
              ("share of batch wins taken by committed templates",
               lambda p: p.get("win_share_committed", np.nan))]
    bounds = res[next(iter(res))]["runs"][0]["bounds"]
    for ax, (ttl, get) in zip(axes.flat, panels):
        for tag, r in res.items():
            x, yv = mean_curve(r["runs"], get)
            ax.plot(x, yv, LS[r["rule"]], color=COL[r["route"]], lw=1.8, label=tag)
        for bb, nm in zip(bounds, ("0-4", "5-9", "0-9")):
            ax.axvline(bb, color="k", lw=0.8, ls=":")
            ax.text(bb + 3, ax.get_ylim()[0], nm, fontsize=8, va="bottom")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("training batches"); axes[1, 1].set_xlabel("training batches")
    axes[1, 2].set_xlabel("training batches")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"28x28 split, {len(SEEDS)} seed(s).  Step: dotted = champion cnt/n, "
                 f"solid = row purity (no floor).  Competition: black none, blue belief, red label",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "purity_split.png", dpi=140); plt.close(fig)

    tags = list(store)
    fig, axes = plt.subplots(2 * len(tags), 12, figsize=(12 * 0.72, 2 * len(tags) * 0.8))
    for i, tag in enumerate(tags):
        WA, com, pA = store[tag]["A"]; WB, pB = store[tag]["B"]
        idx = np.where(com)[0]
        pick = idx[np.argsort(-pA[idx])[:12]] if len(idx) else np.arange(12)
        for c, t in enumerate(pick):
            for r, (Wm, pp) in enumerate(((WA, pA), (WB, pB))):
                ax = axes[2 * i + r, c]
                w = Wm[t]; v = np.abs(w).max() + 1e-9
                ax.imshow(w.reshape(PS, PS), cmap="RdBu_r", vmin=-v, vmax=v)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{pp[t]:.2f}", fontsize=6, pad=1)
        for c in range(len(pick), 12):
            axes[2 * i, c].axis("off"); axes[2 * i + 1, c].axis("off")
        axes[2 * i, 0].set_ylabel(f"{tag}\nafter 0-4", fontsize=6, rotation=0, ha="right", va="center")
        axes[2 * i + 1, 0].set_ylabel("after 5-9", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("the 12 most committed templates after 0-4, and the same templates after 5-9 "
                 "(number = row purity)", fontsize=9)
    fig.subplots_adjust(wspace=0.06, hspace=0.45, top=0.94, left=0.12)
    fig.savefig(OUT / "committed_templates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT / 'purity_split.png'}, {OUT / 'committed_templates.png'}")


if __name__ == "__main__":
    main()
