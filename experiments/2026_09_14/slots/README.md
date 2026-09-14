# slots — L2 as a counted belief network with named parent slots

The 09-14 design, built as agreed. No labels, no tally, no probe.

**What is fixed from before.** L1 is `2026_09_13/neurons/share.py`, unchanged and
frozen after two passes: 144 whole-image cells, nonnegative pictures, settling read, share-then-scale
learning. The only change on its input: pixels are quantised to four levels (0, 1/3, 2/3, 1). L1
still gives strokes, five on per image.

**What is new.** Every node has four levels, level 0 = silent. There is no pixel table. L2 has 100
nodes and they are the *parents* of the L1 nodes:

- each L1 child has four parent **slots**; a slot's **name** is a histogram over the L2 nodes of who
  filled it, and the slot is read as the histogram's argmax (a child's four names distinct);
- each child keeps **one full table of counts** over the levels of its four slots (256 rows) by its
  own four levels; when a slot is renamed, that axis of the table is collapsed to its mean;
- each L2 node keeps a **root prior** as counts over its four levels; this is the price of being on;
- **inference** is settling: twelve damped mean-field steps over beliefs about the L2 levels, each L2
  node scored by its prior plus the expected log-probability of every child that names it, given the
  beliefs of the child's other slots;
- **learning** is counting the settled state: prior counts, the child's table row, and, for children
  that were on, which on-L2 nodes filled which slot (existing names first, then free slots by
  affinity, strongest belief first);
- **pictures and generations** are L1 settled under the expected levels the tables give, then
  painted; never a weighted sum.

Two things were needed to make the loop close at all, both stated here as choices: the starting
prior is loose (P(on) 0.33) and a node's prior P(on) has a floor of 0.03. With the sparse prior the
theory asks for, nothing at L2 ever turned on from a random start, and a learned prior then only
gets sparser.

## Result (held out, 2 passes over 8k, 127 s)

```
L2 on per image            0.73     dead 28%     images with no L2 node on 42%
members per L2 node        mean 1.0, median 1;  <=1 member: 78%,  >=3 members: 5%
slot names                 purity 0.40, 26% of slots above 0.5, 91% unchanged after pass 1
log-prob of the L1 code    tables under settled L2 -21.6, prior -5.6, total -27.2
per held-out image         all L2 off -27.4      children independent, no L2 at all: -25.9
```

The wrapper wall, now in table form. Seventy-eight percent of the L2 nodes raise at most one child.
The five nodes with three or more members are the only real groups, and they are the ones that
draw as pieces of a digit on the board. Generations are one or two strokes. The whole model,
prior included, explains the held-out L1 code *worse* than treating the 144 children as independent.

Why, as the tables show it. Two failures, one for the nodes that fire and one for the nodes that
never do, plus the gap between them.

- **The nodes that fire are each carried by one child.** Of the fifteen busiest nodes, most have
  exactly one child whose row alone gives 6 to 8 nats when the prior charges 3.5, so the node fires
  when that child fires; the ten-odd other children naming it give about 1 nat each. The floor in
  the tables makes this possible: a child that is never at level 3 with all parents off sits at the
  floor there, so any parent-on row, even at its starting strength, clears the prior. The copy is
  the maximum-likelihood table for one child, and one child is enough to pay the price. Across all
  slots one child clears the prior for only 18% (median 1.1 nats), but those are the slots that
  drive the firing.
- **The nodes that never fire are named by nobody.** Naming is Hebbian-positive, so a node that
  fires collects names from every child that was on with it (the busiest node is named by 50 of the
  144 children) and a node that has not fired collects none (the 28 dead nodes are named by 0.4
  children on average). Rich-get-richer in the names, structural death for the rest.
- **In between, the rows that would make groups are never visited.** 82% of slots hold weak rows
  because the rows where the parent is on are only counted when the node fires: a group needs its
  rows visited to sharpen and needs sharp rows to be visited.

41% of held-out firings come from nodes with two or more members; the rest are single-child nodes.
The diagnostic `single_child_clears_prior` in the json measures the first point; `named_by` and
`members` in the npz measure the second.

Board: `results/board_slots_j100_p2.png`. Log: `results/log_slots_j100_p2.txt`.

## The one lever this points at (not run)

Hold the evidence one child can deliver *below* the prior's cost of turning on: with a table floor
of 0.03 a single row can give at most 3.5 nats, exactly the prior's price, so only two or more
children on together can switch an L2 node on. That is the "threshold above what any single line
can deliver" rule of the discussion, in table form. Watch members >= 3 and the log-prob against
independence. It does not touch the second failure; a node no child names stays dead whatever the
floor, and that needs its own rule.
