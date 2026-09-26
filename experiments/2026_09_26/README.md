# 2026-09-26 — reasoning, attention as a window, and the backward path

Three days of conversation (the 24th to the 26th), recorded in [`conversation.md`](conversation.md), and nine experiments.

Reasoning was reframed as composition in time over maps that stay live, with attention as a window that moves a fixed budget and the feedback path as a decoder learned from local mismatch. Such a decoder renders and imagines from a coarse grid of features, but nothing today got it to invent the codes of parts it had not seen: sampling averages, a render-read-correct loop drifts toward what the stack likes, and code arithmetic brings the bundle. What worked was a painter that takes parts from memory and keeps the ones that make the whole read as the goal: sharp and consistent faces, a faint mustache on a woman who stays a woman, and, with only lines and ovals, a drawing in the stack's own style.

| experiment | result |
|---|---|
| [`feedback_render/`](feedback_render/) | a backward path trained by local mismatch reconstructs from switches but cannot imagine without them; the forward kernels flipped can |
| [`what_where/`](what_where/) | with a learned unpool, content alone renders from a 4x4 grid; a bag of features plus one pointer renders nothing |
| [`attend_parts/`](attend_parts/) | sweeping windows rebuilds sharper than one look, to a quarter of the pixel error at 16 windows of 8 |
| [`both_ways/`](both_ways/) | no labels: "copy your twin" collapses to silence, "rebuild the level below" reads out at 94% |
| [`faces/`](faces/) | CelebA at full crop: attributes render from their mean codes, an attended mouth sharpens, a mustache by arithmetic brings the man |
| [`sample_windows/`](sample_windows/) | sampling the top code gives new faces along the covariance; sampling a window's detail gives an average; smaller windows and feathering fix seams, not codes |
| [`loop/`](loop/) | render-read-correct helps for one cycle, then satisfies the stack instead of the face; a noise-blind denoiser makes it worse |
| [`painter/`](painter/) | parts from memory chosen by the look-back: sharp and consistent, real and imagined; a faint mustache on a woman; lines and ovals in the stack's style |
