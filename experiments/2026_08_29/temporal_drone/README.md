# temporal_drone — the shared drone rig

Not an experiment of its own: the modules that the drone experiments of
2026-08-29 → 08-31 import, kept in one place so the viewer at the repo root
(`viewer.py`) can find them with one path. The account of the whole line of
work — the world, the champion, the laws, the exact computations — is
[`ARCHITECTURE.md`](ARCHITECTURE.md).

| module | what it is | used by |
|---|---|---|
| `fast_oracle.py` | the retuned autopilot the pupils imitate | all the CPU rigs, `viewer.py` |
| `td_stack.py` | two layers over time, the reservoir, the policy | [`../temporal_stack/`](../temporal_stack/README.md) |
| `td3_stack.py` | the three-layer version | `../temporal_stack/`, and as a base by `td_bound`, `td_balance` |
| `td_tracks.py` | per-signal tracks | [`../tracks/`](../tracks/README.md) and every later CPU rig |
| `td_bound.py` | the bound form on sensor tracks | [`../bound/`](../bound/README.md), `../balance/`, `../critic/`, `../tally/`, `../split/` |
| `td_balance.py` | the balance rung | [`../balance/`](../balance/README.md), `../critic/`, `../tally/`, `../split/` |
| `chase_view.py` | viewer hook for `../chase/` checkpoints | `viewer.py`, `attend_view.py` |
| `cascade_view.py` | viewer hook for the cascade champion (and its reward tilt) | `viewer.py` (`../cascade/`, `../reward_rl/`) |
| `attend_view.py` | viewer hook for the attention rig | `viewer.py` (`../attend/`) |
| `grid_view.py` | viewer hook for the grid-partition policy | `viewer.py` (`../grid_ablation/`) |
| `purerl_view.py` | viewer hook for the from-scratch pure-RL policy | `viewer.py` (`../reward_rl/`) |

Every checkpoint's sidecar json names its module and
`"dir": "experiments/2026_08_29/temporal_drone"`; the viewer adds that folder to
its path and calls the module's `load_policy`. The modules import `sl_drone`
from `experiments/2026_08_28/single_layer_drone/`, which the calling script
(or the viewer) puts on the path.

The GPU scripts (`gpu_chase`, `gpu_cascade`, `gpu_deep_outer`, `gpu_attend`) stay
in their own experiment folders and import each other through sibling-folder
paths.
