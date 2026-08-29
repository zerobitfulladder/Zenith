"""A faster autopilot for the drone world.

The old oracle sets tilt straight from position error:

    phi_des = 0.15*ex + 0.3*vx

Solve that for its equilibrium and you find the velocity it is really
asking for: vx = -0.5*ex. A fixed gain of 0.5, so the commanded speed
shrinks in proportion to the distance left — an exponential approach
that gets slower exactly as it closes in. That is the creep, and it is
a gain choice, not a thrust limit (the drone has 6.2 m/s^2 of upward
authority, 9.8 down and ~5.5 sideways at the tilt limit).

This one makes the velocity command explicit and doubles its gain, then
divides collective by cos(tilt) so vertical thrust survives the tilt.

Measured over 400 random episodes (success stays 1.00 throughout):

    PD (old)                    160 ticks to the target, 212 total
    velocity-P, gain 1.0         84                      184
      + cos(tilt) compensation   86                      173
      + brake-limited profile    85                      172

So the win is the gain, plus a little from the tilt compensation. The
brake-limited profile is worth ~1 tick at these distances; it is kept
because it bounds the commanded speed by what can still be stopped in
the distance remaining, which matters for far targets.

Total time improves only 18% because `at_goal` requires the drone to be
inside 0.25 of the target, under 0.3 m/s, tilted under 0.17 rad, for 10
consecutive ticks. Braking hard needs tilt, so a fast arrival fights the
criterion: pushed harder, approach drops to 39 ticks but success falls
to 0.02. The far field is where the time is; the last 0.25 is pinned by
the success definition.
"""

import numpy as np

import sl_drone as W

POS_P = 1.0          # velocity commanded per unit of position error
AMAX = 3.0           # deceleration the profile is allowed to plan for
KVX = 0.30           # tilt per unit of horizontal velocity error
KVY = 1.80           # collective per unit of vertical velocity error
KA, KDA = 2.0, 0.6   # attitude loop (unchanged from the old oracle)
PHIMAX, DMAX = 0.6, 2.5


def velocity_command(e, pos_p=POS_P, amax=AMAX):
    """How fast to travel, given how far is left.

    Inside d = amax/pos_p^2 it is proportional, so the drone is actually
    stopped when it arrives instead of still drifting. Outside, it is
    sqrt(2*amax*(|e| - d/2)) — the fastest speed from which the distance
    remaining is still enough to stop in. The two meet smoothly.
    """
    d = amax / (pos_p * pos_p)
    a = abs(e)
    v = pos_p * a if a <= d else np.sqrt(2.0 * amax * (a - 0.5 * d))
    return -np.sign(e) * v


def teacher(s, tgt):
    """Cascade: position -> velocity -> tilt & collective -> thrusts."""
    ph = W.wrap(s[4])
    vdx = velocity_command(s[0] - tgt[0])
    vdy = velocity_command(s[1] - tgt[1])
    # tilt is how the drone accelerates sideways; chase the velocity error
    phi_des = np.clip(KVX * (s[2] - vdx), -PHIMAX, PHIMAX)
    # collective, then undo the cosine loss from being tilted
    coll = (W.HOVER + KVY * (vdy - s[3])) / max(np.cos(ph), 0.5)
    coll = np.clip(coll, 0.3, W.TMAX)
    diff = np.clip(KA * (phi_des - ph) - KDA * s[5], -DMAX, DMAX)
    t1 = np.clip(coll - diff, 0, W.TMAX)
    t2 = np.clip(coll + diff, 0, W.TMAX)
    return (int(np.argmin(np.abs(W.LEVELS - t1))),
            int(np.argmin(np.abs(W.LEVELS - t2))))


def bench(fn, n=400, seed=0):
    """(success, median ticks to reach the target, median ticks total)."""
    rng = np.random.default_rng(seed)
    ok, done, near = [], [], []
    for _ in range(n):
        s, tgt = W.any_init(rng)
        hold, fin, tn = 0, None, None
        for t in range(W.EP_CAP):
            s = W.physics(s, fn(s, tgt))
            if tn is None and np.hypot(s[0] - tgt[0], s[1] - tgt[1]) < 0.5:
                tn = t
            if W.at_goal(s, tgt):
                hold += 1
                if hold >= 10:
                    fin = t
                    break
            else:
                hold = 0
            if abs(s[0]) > 30 or abs(s[1]) > 30:
                break
        ok.append(fin is not None)
        if fin is not None:
            done.append(fin)
            near.append(tn if tn is not None else fin)
    return (float(np.mean(ok)), float(np.median(near)), float(np.median(done)))


if __name__ == "__main__":
    for name, fn in (("PD (old)", W.teacher), ("fast", teacher)):
        o, n_, t = bench(fn)
        print(f"{name:<10} success {o:.2f}  to target {n_:.0f}  "
              f"total {t:.0f} ticks ({t * W.DT:.2f} s)")
