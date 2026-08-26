# `arm/` — two-joint reaching arm, parked at the dimensionality wall

Scripts: [`run_arm.py`](run_arm.py) (delta-command form), [`run_arm_nonext.py`](run_arm_nonext.py) (no-next-slot form), [`arm_viewer.py`](arm_viewer.py) (click-to-command viewer on the delta weights). Results: [`results/delta/`](results/delta/) and [`results/nonext/`](results/nonext/) — `report.md`, `weights.npz` each.

## Two-axis reaching arm (run_arm.py, run_arm_nonext.py, arm_viewer.py)
## — CHARACTERIZED, parked at the dimensionality wall

User spec: kinematic 2-joint arm, target zone as 12x12 2D semantic
bumps, joints as circular bumps + present/trail channels, click-to-
command viewer. Expert (greedy 3-deg descent) 249/250.

Findings, in order: (1) emission-as-absolute-pose = teleport-and-
freeze, measured (10-26deg jump then 0.000deg forever) — EMISSION
MUST BE A COMMAND, NOT A STATE (every working rig already obeyed
this; the arm briefly didn't). (2) Three emission forms at K=512 —
absolute pose (2/0/3), delta command (3/8/6), and the user's
no-next-slot lag-advance design (4/2/2, run_arm_nonext.py: query with
the present half empty; the stored present, sitting AHEAD of the
matched trails, IS the prediction — time from the integration lag
itself) — fail in the SAME band: the failure is upstream of emission,
in the memory partition. (3) Capacity test K=512 -> 2048 (delta
form): 3/8/6 -> 10/11/17, still climbing — CURSE OF DIMENSIONALITY
measured: 4x memory ~ 2-3x success on the first genuinely 4D task
(target x,y times two joints; the pole was ~2D and cheap).

LAW: flat memory-based control pays exponentially for state
dimensions. The known escape is the composition thread's own result —
FACTOR THE POLICY (per-joint banks, marginal associations, local
reads: arm-A/Gq machinery applied to control). Queued as the natural
next-session opener, alongside K=8192 and more DAgger rounds as brute
comparisons. The user's lag-advance design deserves a retest in a
low-D task where the partition is not the binding constraint.
arm_viewer.py works with the K=2048 weights (click to place the
zone; expect ~1-in-5 clean reaches, drifty orbits otherwise).
