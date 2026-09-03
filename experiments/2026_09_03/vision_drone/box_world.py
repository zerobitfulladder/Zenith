"""A box world the drone SEES. Egocentric vision, PID teacher.

World: the drone physics of sl_drone (2-D quadrotor: x, y, vx, vy, tilt,
tilt rate; two thrust levels), inside a square box of half-size BOX. The
drone starts at the centre at rest and level; the target is drawn uniformly
inside the box with a MARGIN so the drone never needs to leave it.

Vision: an egocentric image of SIZE x SIZE pixels covering FOV world units
either side of the drone. The drone is always at the centre, drawn as a bar
rotated by its tilt with a rotor blob at each end; the target is a bright blob. No walls are drawn: only the
drone and the target. Nonnegative, 0..1. No velocity in the image:
what the drone sees is where things are, not how they move.

Teacher: the PID from viewer.py (position PID -> desired tilt + collective
-> attitude PD -> the two thrust levels), tuned 2026-08-30.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))
import sl_drone as W

BOX, MARGIN = 5.0, 1.0
SIZE, FOV = 48, 6.0
TARGET_R, DRONE_L, DRONE_W, ROTOR_R = 0.45, 2.0, 0.22, 0.32


def make_pid():
    G = {"KP": 4.811, "KI": 0.011, "KD": 4.084, "ILIM": 1.987,
         "KT": 0.071, "PHIMAX": 0.696, "KAp": 6.426, "KAd": 0.943}
    st = {"I": np.zeros(2), "prev": None}

    def pid(s, tgt):
        key = (float(tgt[0]), float(tgt[1]), float(s[0]), float(s[1]))
        if (st["prev"] is None or key[:2] != st["prev"][:2]
                or np.hypot(key[2] - st["prev"][2], key[3] - st["prev"][3]) > 2.0):
            st["I"][:] = 0.0
        st["prev"] = key
        ex, ey = tgt[0] - s[0], tgt[1] - s[1]
        st["I"][0] = np.clip(st["I"][0] + ex * W.DT, -G["ILIM"], G["ILIM"])
        st["I"][1] = np.clip(st["I"][1] + ey * W.DT, -G["ILIM"], G["ILIM"])
        ax = G["KP"] * ex + G["KI"] * st["I"][0] - G["KD"] * s[2]
        ay = G["KP"] * ey + G["KI"] * st["I"][1] - G["KD"] * s[3]
        ph = W.wrap(s[4])
        ph_des = np.clip(-G["KT"] * ax, -G["PHIMAX"], G["PHIMAX"])
        coll = np.clip(0.5 * (9.8 + ay) / max(np.cos(ph), 0.5), 0.3, W.TMAX)
        diff = np.clip(G["KAp"] * (ph_des - ph) - G["KAd"] * s[5], -2.5, 2.5)
        t1 = np.clip(coll - diff, 0, W.TMAX); t2 = np.clip(coll + diff, 0, W.TMAX)
        return (int(np.argmin(np.abs(W.LEVELS - t1))), int(np.argmin(np.abs(W.LEVELS - t2))))
    return pid


def init(rng):
    s = np.zeros(6)
    tgt = rng.uniform(-(BOX - MARGIN), BOX - MARGIN, size=2)
    return s, tgt


def render(s, tgt, size=SIZE, fov=FOV):
    """Egocentric view: world coordinates relative to the drone, up = +y."""
    px = np.linspace(-fov, fov, size)
    X, Y = np.meshgrid(px, -px)                        # row 0 is the top
    img = np.zeros((size, size), np.float32)
    # target blob
    dx, dy = tgt[0] - s[0], tgt[1] - s[1]
    img = np.maximum(img, np.exp(-((X - dx) ** 2 + (Y - dy) ** 2) / (2 * TARGET_R ** 2)))
    # the drone itself: a bar at the centre rotated by its tilt, with a rotor blob at
    # each end, so the angle is unmistakable. No walls: only the drone and the target.
    ph = W.wrap(s[4])
    u, v = np.cos(ph), np.sin(ph)
    along = X * u + Y * v; across = -X * v + Y * u
    bar = np.exp(-(across ** 2) / (2 * DRONE_W ** 2)) * (np.abs(along) < DRONE_L / 2)
    img = np.maximum(img, 0.8 * bar)
    for e in (-1.0, 1.0):
        ex, ey = e * u * DRONE_L / 2, e * v * DRONE_L / 2
        img = np.maximum(img, np.exp(-((X - ex) ** 2 + (Y - ey) ** 2) / (2 * ROTOR_R ** 2)))
    return np.clip(img, 0.0, 1.0)


def episode(rng, pid=None, cap=W.EP_CAP):
    """One PID episode from the centre. Returns states, targets, commands, frames."""
    pid = pid or make_pid()
    s, tgt = init(rng)
    S, L, F = [], [], []
    hold = 0
    for t in range(cap):
        lv = pid(s, tgt)
        S.append(s.copy()); L.append(lv); F.append(render(s, tgt))
        s = W.physics(s, lv)
        s[0] = np.clip(s[0], -BOX, BOX); s[1] = np.clip(s[1], -BOX, BOX)
        hold = hold + 1 if W.at_goal(s, tgt) else 0
        if hold >= 10:
            break
    return np.array(S), tgt, np.array(L), np.array(F)


def episode_no_frames(rng, pid=None, cap=W.EP_CAP):
    pid = pid or make_pid()
    s, tgt = init(rng)
    S, L = [], []
    hold = 0
    for t in range(cap):
        lv = pid(s, tgt)
        S.append(s.copy()); L.append(lv)
        s = W.physics(s, lv)
        s[0] = np.clip(s[0], -BOX, BOX); s[1] = np.clip(s[1], -BOX, BOX)
        hold = hold + 1 if W.at_goal(s, tgt) else 0
        if hold >= 10:
            break
    return np.array(S), tgt, np.array(L), None


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    S, tgt, L, F = episode(rng)
    picks = [0, len(S) // 4, len(S) // 2, len(S) - 1]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.6))
    for c, t in enumerate(picks):
        ax = axes[0, c]
        ax.plot([-BOX, BOX, BOX, -BOX, -BOX], [-BOX, -BOX, BOX, BOX, -BOX], "k-", lw=2)
        ax.plot([-(BOX - MARGIN), BOX - MARGIN, BOX - MARGIN, -(BOX - MARGIN), -(BOX - MARGIN)],
                [-(BOX - MARGIN), -(BOX - MARGIN), BOX - MARGIN, BOX - MARGIN, -(BOX - MARGIN)],
                "k:", lw=0.8)
        ax.plot(S[:t + 1, 0], S[:t + 1, 1], "-", color="#1b6ca8", lw=1.2, alpha=0.7)
        ax.plot(tgt[0], tgt[1], "o", color="#c1121f", ms=10)
        ph = W.wrap(S[t, 4]); u, v = np.cos(ph) * DRONE_L / 2, np.sin(ph) * DRONE_L / 2
        ax.plot([S[t, 0] - u, S[t, 0] + u], [S[t, 1] - v, S[t, 1] + v], "-", color="k", lw=3)
        sq = FOV
        ax.add_patch(plt.Rectangle((S[t, 0] - sq, S[t, 1] - sq), 2 * sq, 2 * sq,
                                   fill=False, ec="#1b6ca8", ls="--", lw=0.8))
        ax.set_xlim(-BOX - 1.5, BOX + 1.5); ax.set_ylim(-BOX - 1.5, BOX + 1.5); ax.set_aspect("equal")
        ax.set_title(f"tick {t}: the box  (levels {L[t][0]},{L[t][1]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax2 = axes[1, c]
        ax2.imshow(F[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title(f"what the drone sees  ({SIZE}x{SIZE}, ±{FOV:g} units)", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle(f"Box world, egocentric vision. Drone starts at the centre, target inside the dotted margin. "
                 f"PID episode, {len(S)} ticks.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(HERE / "results" / "sample.png", dpi=130)
    print(f"episode {len(S)} ticks, target {tgt.round(2)}, wrote results/sample.png")
