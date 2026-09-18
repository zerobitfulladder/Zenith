"""Does credit survive depth?   python depth.py

Lavender's originating claim: there are no layers here, and what plays the part of depth is the
trajectory of the search -- so prediction error should propagate back along that trajectory as a
rotation on the angles.  `phase.py` tested the rotation rule at a chain of one bind and one
bundle.  This tests whether it still works at depth.

XVII.2 argues depth is free: binding is addition, so the derivative of the final match with
respect to EVERY angle in the chain is the same scalar, +-sin(disagreement).  Nothing multiplies
along the chain, so there is no vanishing or exploding term.  A pure chain of bindings therefore
cannot fail, and testing one would prove nothing.

What can fail is what sits between the links:

    bind (exact, lossless)  ->  bundle with distractors (adds crosstalk)  ->  cleanup (a snap)

Bundling degrades the signal by ~1/sqrt(N) each time.  Cleanup is non-differentiable: the chain
is cut at every snap unless the error is passed through it (straight-through, as VQ-VAE does).
So the experiment is a chain of d such stages, and the questions are:

  1. does a leaf code still learn when the error is observed d stages away?
  2. does cleanup between stages rescue the signal (IV.3's re-sparsification, which the design
     says is structurally necessary) or does it cut the credit path?
  3. how far down the chain does the correction actually reach?

Ground truth is synthetic so that depth is the only thing varying: a hierarchy of cards, each
card K children at K distinct offsets, names derived by walking (XVI product names) so a card's
name is a function of its parts and the gradient has somewhere to go.
"""
import argparse, time
import numpy as np


TAU = 2 * np.pi


def harmonics(m):
    k = np.arange(1, m + 1, dtype=float)
    w = 1 - k / (m + 1)
    return k, w / w.sum()


def scatter(arr, idx, upd):
    cnt = np.bincount(idx, minlength=len(arr))
    acc = np.zeros_like(arr); np.add.at(acc, idx, upd)
    nz = cnt > 0
    arr[nz] += acc[nz] / cnt[nz, None]


class World:
    """A hierarchy of cards over a leaf vocabulary, fixed before any learning happens."""

    def __init__(self, rng, V, ND, K, depth, n_cards):
        self.V, self.ND, self.K, self.depth = V, ND, K, depth
        self.levels = []
        pool = list(range(V))                       # level 0 = the leaves
        for d in range(depth):
            cards = []
            for _ in range(n_cards):
                kids = rng.choice(len(pool), K, replace=False)
                offs = rng.choice(ND, K, replace=False)
                cards.append((d, [pool[i] for i in kids], list(offs)))
            self.levels.append(cards)
            pool = list(range(len(cards)))
        self.top = self.levels[-1]

    def leaves(self, card, off=0):
        """Expand a card to (leaf id, accumulated offset count) pairs.  Offsets accumulate by
        addition because binding is addition -- the accumulated pose of a leaf is the sum of the
        poses on the path down to it."""
        d, kids, offs = card
        out = []
        for kid, o in zip(kids, offs):
            if d == 0:
                out.append((kid, [o] + ([off] if off else [])))
            else:
                for lf, path in self.leaves(self.levels[d - 1][kid], o):
                    out.append((lf, path + [o] + ([off] if off else [])))
        return out


def run(args):
    rng = np.random.default_rng(args.seed)
    V, ND, B = args.vocab, args.offsets, args.blocks
    w_k, w_w = harmonics(args.harm)
    world = World(rng, V, ND, args.K, args.depth, args.cards)

    theta = rng.uniform(0, TAU, (V, B))
    rho = rng.uniform(0, TAU, (ND, B))
    theta0 = theta.copy()

    def episode(learn, lr):
        """Observe every leaf of a top card but one; predict the missing leaf.

        Each observed leaf votes through the accumulated pose of its path, which is a chain of
        `depth` bindings.  Between levels the partial bundle is optionally cleaned up, which is
        where re-sparsification would act.
        """
        card = world.top[rng.integers(len(world.top))]
        lv = world.leaves(card)
        if len(lv) < 2:
            return 0, None
        j = rng.integers(len(lv))
        true = lv[j][0]
        ctx = [x for i, x in enumerate(lv) if i != j]
        tgt_path = lv[j][1]

        # the chain: each contributor's angle is its code walked by every pose on its path,
        # then walked BACK by the target's path -- a chain of len(path)+len(tgt) bindings.
        src, offs, angs = [], [], []
        for lf, path in ctx:
            a = theta[lf].copy()
            for o in path:
                a = a + rho[o]
            for o in tgt_path:
                a = a - rho[o]
            if args.clean:                      # re-sparsification: snap back to a clean code
                sc = (np.exp(1j * a)[None] * np.exp(-1j * theta)).real.sum(1)
                a = a + (theta[int(sc.argmax())] - a) * args.clean
            src.append(lf); offs.append(path + tgt_path); angs.append(a)
        ang = np.stack(angs)
        s = np.array(src)

        dl = ang[:, None, :] - theta[None, :, :]
        kd = w_k[:, None, None, None] * dl[None]
        score = np.einsum("m,mjvb->v", w_w, np.cos(kd)) / len(ang)
        pred = int(score.argmax())

        if learn:
            G = np.einsum("m,mjvb->jvb", w_w * w_k, np.sin(kd))
            p = np.exp(args.beta * (score - score.max())); p /= p.sum()
            d = p.copy(); d[true] -= 1.0
            theta[:] -= lr * d[:, None] * G.sum(0) / len(ang)
            gj = G[:, true, :] - np.einsum("v,jvb->jb", p, G)
            scatter(theta, s, -lr * gj)
            # every pose on the path takes the SAME step -- binding is addition, so the
            # derivative is identical at every link.  Averaged per pose across the contributors
            # that used it, exactly as the leaf codes are, so depth carries no built-in handicap.
            fo = np.concatenate([np.asarray(p, int) for p in offs])
            fg = np.concatenate([np.repeat(gj[i][None], len(p), 0) for i, p in enumerate(offs)])
            scatter(rho, fo, -lr * fg)
        return int(pred == true), true

    def test(n=400):
        hit = 0
        for _ in range(n):
            h, t = episode(False, 0)
            hit += h
        return hit / n

    t0 = time.time()
    base = test()
    for ep in range(args.train):
        episode(True, args.lr * (1 - .9 * ep / args.train))
    acc = test()
    moved = float(np.abs(np.angle(np.exp(1j * (theta - theta0)))).mean())
    return base, acc, moved, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=40); ap.add_argument("--offsets", type=int, default=24)
    ap.add_argument("--blocks", type=int, default=32); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--cards", type=int, default=12); ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--harm", type=int, default=1); ap.add_argument("--clean", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if not args.sweep:
        b, a, mv, t = run(args)
        print(f"depth {args.depth}: {b:.3f} -> {a:.3f}   codes moved {mv:.3f} rad   [{t:.0f}s]")
        return

    print(f"{args.vocab} leaves, {args.K} children per card, {args.blocks} blocks, "
          f"{args.train} episodes, chance = 1/{args.vocab} = {1/args.vocab:.3f}\n")
    print("            leaves |   before    after   moved | with cleanup between levels")
    for d in (1, 2, 3, 4):
        args.depth = d
        args.clean = 0.0
        b, a, mv, t = run(args)
        args.clean = 1.0
        b2, a2, mv2, _ = run(args)
        print(f"  depth {d}  {args.K ** d:>5}     |  {b:.3f}    {a:.3f}    {mv:.2f}  |"
              f"   {b2:.3f} -> {a2:.3f}   moved {mv2:.2f}   [{t:.0f}s]")


if __name__ == "__main__":
    main()
