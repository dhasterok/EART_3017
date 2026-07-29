Re-audited against the current state of all 10 weeks (timelines + activity sheets). Most of the structural fixes from the last pass landed; several are still open, and the pass surfaced a good number of new issues, including one shared root cause behind three different weeks' labeling bugs.

## 0. Root cause behind the recurring label bugs

`\timeline` (in `latex/aucourse.sty`) only special-cases two type keywords: `slot` (prints no label, does **not** step the `wshopitem` counter) and `demo` (prints "Demo," does step it). Every other keyword — including invented ones like `warmup` — falls through to the generic branch, which prints "Activity" and *does* step the counter. Meanwhile `\begin{activity}[Warm-up]{...}` on the sheet side has its own independent mechanism for swapping the printed word to "Warm-up." These two mechanisms don't talk to each other, which produces two distinct symptoms depending on which type a timeline entry uses:

- Tagged `slot` when it shouldn't be → letter never advances, everything after it is one behind (Weeks 5, 7, 8 below).
- Tagged `warmup`/`item` for an actual warm-up sheet → letter is fine, but the timeline prints "Activity N.A" while the sheet prints "Warm-up N.A" (Weeks 7, 9 below).

**Recommended fix:** add a real `warmup` case to `\timeline` in `aucourse.sty` that both steps `wshopitem` and prints "Warm-up" (mirroring what `\begin{activity}[Warm-up]{...}` already does for sheets). That one change fixes the wording mismatch in Weeks 7 and 9 outright, and once it exists, swapping the wrongly-tagged `slot` entries (Weeks 5, 7, 8) over to the correct type is a one-line fix per week.

## 1. Timeline vs. activity sheet consistency

**Clean: Weeks 1, 4.** Activities present, correctly labeled, correctly ordered, times continuous.

**Week 2.** Both previously-flagged typos are fixed. Still open: the 45–60 "Simulation Walkthrough" and Demo 2.C *Path Simulation* (125–155) still read as the same demonstration scheduled twice — no distinguishing text has been added to either blurb. New: timeline titles 2.A as "Rock Fabric and Mixing Laws," sheet header uses "Rock Fabric & Mixing Laws" — trivial, but pick one form.

**Week 3.** The 3.B/3.C timing overlap is fixed — the session is now contiguous. Still open: timeline calls 3.C "Buried Sphere — Background" while the sheet is titled *Inversion of a Buried Sphere*. New: the Investigation blurb states the $a^3\Delta\rho$ trade-off and its resolution before students derive the non-uniqueness argument themselves — see §4. New: "local vs. global minimum" is repeated almost verbatim across all three activity sheets — see §3. The suggested 3.C grid-search-vs-inversion-path figure has already been added (two flowcharts, matching caption) — removed from the open figure list in §7.

**Week 5 — partially fixed.** 5.A Part 1/Part 2 and 5.B are now labeled correctly. Still open: the Numerical Demonstration is still unlabeled because its timeline entry uses `\timeline[slot]`; change it to `[item]` (it should become 5.C). Session length is now 170 min, inside the 160–170 target — the earlier 180-min overrun is resolved. The suggested 5.B slab-cancellation figure has not been added — still open in §7.

**Week 6.** The 6.B naming variant ("Magnetic Anomalies vs. Gravity" → "Magnetic Structures") is fixed. New: 6.D's timeline blurb truncates the sheet title ("Curie Depth and Magnetic Crust" vs. the sheet's "...Magnetic Crustal Thickness"). New: 6.C's own Key Ideas text mislabels a self-reference as "Activity 6.C" where it should say "Activity 6.B." New: a numeric inconsistency — 6.D's Key Ideas give the Curie temperature of magnetite as ~580°C, but the sheet's own question uses 570°C. New: the 6.B timeline blurb's example structures don't match what the sheet actually covers — worth a line-by-line check against the sheet's real structure list.

**Week 7.** Labeling bug unchanged: Fold Belt Survey is still unlabeled (`\timeline[slot]` at line 47 should be `[item]`), pushing Nevada and Tower Hill one letter early. The chapter-level Key Ideas block after the preview is confirmed to be a deliberate one-off (checked against all 10 timelines) — no action needed, just noting the question is resolved. New: the Nevada blurb still references an "effect of aliasing on inversion results" step that corresponds to a section now commented out in the activity sheet. New: the warm-up blurb still promises "isostatic filtering," which no longer happens anywhere in Week 7 now that the isostatic-correction step has been removed from `nevada_fft.py`. New: "RTP is a phase filter, not an amplitude filter" is stated three times verbatim (Purpose, Key Ideas, and timeline blurb) — see §3. New: Fold Belt's blurb gives away the aliasing punchline before the activity's own reveal, and Tower Hill's blurb gives away the asymmetric-anomaly result before Q1/Q2 ask students to predict it — see §4. New: four orphaned draft files are still sitting in `activities/` — see §8.

**Week 8 — improved but not fixed.** The previously-missing Structures activity (`activity_structural.tex`, flux refraction at conductivity contrasts) is now scheduled. However, the Warm-up's timeline entry is still tagged `[slot]`, so it still doesn't step the counter — every subsequent label is now off by **one** instead of two. Since the sheet itself just prints "Activity 8.A" (it doesn't use the `[Warm-up]` sheet variant), the fix here is purely mechanical: change the Warm-up's `\timeline[slot]` to `[item]` and every label downstream falls back into sync. The duplicated first paragraph of 8.A's Key Ideas is still present, unchanged (see §3). New: the Structures timeline entry has no blurb paragraph at all, unlike every other entry in the booklet. New: `activity_geotherm.tex` has a broken cross-reference, `\ref{fig:layres}`, that should be `fig:layers`. New: `activity_velocity_heat_production.tex` has a malformed unit, `\si{mW/m^{-2}}`, that should be `\si{mW.m^{-2}}`. Session is 175 min (5 over target); `week8_workshop.pdf` in the repo predates these source changes and should be recompiled before it's handed out.

**Week 9.** The Warm-up/Activity wording mismatch is still present — this is the `warmup`-vs-generic-type issue described in §0, not yet fixed. Session is still 175 min (5 over target); the earlier suggestion to shift 10 min from 9.C to 9.D hasn't been applied. New: timeline titles the plate-cooling entry "Half-space and Plate Cooling," but the sheet header reads "Oceanic Lithosphere: Half-Space and Plate Cooling." New, and worth a real decision: `activity_collisions.tex` is a fully written, ready-to-run activity that the warm-up's closing line explicitly promises ("looked at in greater detail in scenarios examined later") but it is never scheduled anywhere in the timeline — unlike the week's other cut activities, there's no commented-out timeline stanza for it, which suggests this is an oversight rather than a deliberate cut. New: two cross-references to "Activity 9.A" should say "Activity 9.C" — one in `activity_oceanic_cooling.tex`, and one in `half_space_figure.py`'s own module docstring (a file touched this session). New: the timeline's content outline for the plate-cooling entry describes Fourier-mode decay content that lives in the orphaned `activity_plate_cooling.tex`, not in the sheet that's actually scheduled (`activity_oceanic_cooling.tex`). New: `activity_thermal_timescales.tex`'s figure caption says "three times $t_1<t_2<t_3$" but the figure now plots six curves, $t_0$–$t_5$, including the $t=0$ discontinuity added this session. Minor: that same file uses "Key Idea" (singular) where every other sheet uses "Key Ideas."

**Week 10.** 10.E and 10.F are confirmed fully orphaned — not just unscheduled, but not referenced anywhere in the tree, so they're effectively out of the booklet rather than mid-revision. Two of the three name variants from the last pass have converged (Human Finite Difference Solver; Numerical Stability and Information Transfer); "Boundary Conditions as Geological Assumptions" (timeline) vs. "…and Geological Meaning" (sheet, confirmed at `activity_geologic_boundaries.tex:2`) is still mismatched. The old note about 10.C/10.D sheets having no questions yet is obsolete — both now have real, numbered questions. New, and the most concrete bug in this pass: `activity_stability.tex:153` `\input`s `figures/explicit_table` a second time instead of `figures/implicit_table` — this duplicates a table on the page, leaves `\ref{tab:implicit}` undefined, and contradicts the surrounding text, which is specifically discussing the implicit scheme's unconditional stability. New: the stability threshold $\Delta t < \Delta x^2/2\kappa$ is stated twice before the discovery question that's supposed to have students find it themselves (once in the mini-lecture, once in 10.D's own intro) — see §4. Session length is down to 180 min (from 220) but still 10–20 over target.

## 2. Time budgets

Current session totals: 165, 175, 175, 170, 170, 180, 180, 175, 175, 180 (Weeks 1–10). Target is 160–170.

- **Week 6 is unchanged and is still the one that needs surgery** — 240 min against a 160–170 target is 70–80 min over, and the session still has only one scheduled break across four hours. 6.D (50 min, 10 questions) and 6.C's three-station rotation remain the natural places to trim, per the last pass.
- **Week 10 improved substantially** (220 → 180) but is still 10–20 min over. Trimming here is far more tractable than Week 6 — the orphaned 10.E/10.F content isn't consuming any of that 180, so the overrun is coming entirely from 10.A–10.D plus the "when analytic solutions break down" discussion.
- **Week 5 is now fully in range** (180 → 170) — no further action needed once 5.C's label is fixed.
- Weeks 2, 3, 8, and 9 are all sitting at 175 (5 over target) — each is close enough that trimming any single 5–10 min block (e.g., the debrief/wrap-up slots) would bring them into range.
- Week 8's total moved from 170 to 175 because the previously-unscheduled Structures activity is now counted — that's expected, not a regression.
- Weeks 1, 4 remain within target, no change needed.
- Still worth revisiting from the last pass: 1.B (15 min/6 questions), 7.C Nevada (50 min, the densest single block in the booklet — protect it by trimming the Fold Belt debrief instead), and 9.D (35 min/7 questions, conceptually the hardest material in Week 9 while 9.C is comparatively generous at 45 min/4 questions).

## 3. Redundant or duplicated Key Ideas

- **8.A** (`activity_refraction.tex`): first paragraph of the Key Ideas is duplicated nearly verbatim — still present, not yet cleaned up.
- **1.A**: Key Ideas contains a duplicated sentence, the same pattern as 8.A above — new finding, not previously flagged.
- **3.A/3.B/3.C**: "local vs. global minimum" is repeated almost word-for-word across all three sheets' Key Ideas. Worth keeping the concept in each but varying the phrasing, or stating it fully once (3.B, where it's first load-bearing) and referring back afterward.
- **4.C**: internal duplication between its own first and third Key Ideas bullets, plus overlap with 4.B — both restate the width/depth relationship almost identically. Decide which activity owns that idea and have the other reference it.
- **Week 7**: "RTP is a phase filter, not an amplitude filter" appears verbatim in the Purpose, the Key Ideas, and the timeline blurb. Once is a key idea; three times is either deliberate hammering-home or copy-paste — worth a deliberate decision either way.
- **Week 9**: no new duplication issues — the near-duplicate risk with `activity_half_space_cooling.tex` was avoided because that sheet is currently cut from the schedule.
- **Week 10**: mild overlap between 10.B and 10.D on "information propagates at a finite rate," but this reads as intentional scaffolding (10.B introduces it, 10.D formalizes it) rather than an editing leftover — flagging for awareness, not necessarily something to fix. If 10.E is ever reinstated, check its Key Ideas against 10.C's before scheduling it — they cover adjacent ground.

## 4. Timeline blurbs that give away the answer

- **Week 3**: the Investigation blurb states the $a^3\Delta\rho$ trade-off and its resolution before students are asked to derive the non-uniqueness argument themselves.
- **Week 4**: the Buried Sphere blurb states the conclusion to Q2 (the depth/width relationship) and near-verbatim reprints Q3's own sub-questions ahead of the sheet.
- **Week 7**: Fold Belt's blurb states the aliasing conclusion before the activity's own reveal; Tower Hill's blurb states the asymmetric-anomaly result before Q1/Q2 ask students to predict it first.
- **Week 10**: the stability threshold ($\Delta t < \Delta x^2/2\kappa$) is given twice before 10.D's discovery question is meant to have students find it themselves; milder versions of the same issue show up in 10.A's (Laplacian-as-curvature) and 10.B's (smoothing) blurbs.

## 5. Other content/reference bugs found along the way

- `week_10_finite_differences/workshop/activities/activity_stability.tex:153` — wrong `\input`, duplicate table, undefined `\ref{tab:implicit}` (see §1, Week 10).
- `week_8_steady_state_heat/workshop/activities/activity_geotherm.tex` — `\ref{fig:layres}` should be `fig:layers`.
- `week_8_steady_state_heat/workshop/activities/activity_velocity_heat_production.tex` — `\si{mW/m^{-2}}` should be `\si{mW.m^{-2}}`.
- `week_9_transient_heat/workshop/activities/activity_oceanic_cooling.tex` and `src/utils/half_space_figure.py` (module docstring) — both say "Activity 9.A" where they mean "Activity 9.C."
- `week_9_transient_heat/workshop/activities/activity_thermal_timescales.tex` — figure caption undercounts the curves (says three, shows six) and uses "Key Idea" instead of "Key Ideas."
- Week 9's plate-cooling timeline entry describes content that lives in the orphaned `activity_plate_cooling.tex` rather than the sheet actually scheduled.
- Week 8's Structures timeline entry has no blurb paragraph at all.

## 6. Orphaned or superseded files

Worth a deliberate archive/delete pass rather than leaving them in `activities/` where they can be edited by mistake or picked up by a future grep:

- **Week 2**: two orphaned demo files superseded by the current pair of sheets.
- **Week 7**: `activity_nevada_fft.tex` (non-v2 — still contains a live Airy isostatic-correction section that contradicts the current HP-only direction) and `activity_tower_hill_rtp.tex` (non-v2), plus `activity_fft_demo1.tex`/`activity_fft_demo2.tex`.
- **Week 9**: `activity_plate_cooling.tex` and `activity_half_space_cooling.tex` (both superseded by `activity_oceanic_cooling.tex`/`activity_thermal_timescales.tex`), and `activity_collisions.tex` — though note this last one is a live candidate for scheduling rather than deletion, see §1.
- **Week 10**: 10.E/10.F sources (confirmed fully unreferenced, see §1).

## 7. Summary figures for Key Ideas — still open

Of the seven originally suggested, one has been added since the last pass:

- ~~3.C grid search vs. inversion path~~ — **done**, two flowcharts with matching caption are already in the sheet.

Still open, in the same rough priority order as before:

1. **8.A Thermal Refraction — kinked geotherm.** Two T(z) profiles (exposed basement vs. sediment-over-basement) with the same q arrows and visibly different slopes.
2. **9.D — heat flow and subsidence vs. age.** The canonical two-panel figure for the topic; still text-only.
3. **8.C — k∥/k⊥ layered block**, deliberately echoing the 2.A cartoon style.
4. **6.D — Curie depth vs. Moho columns**, craton and rift side by side.
5. **5.B — slab cancellation** (topographic slab over root slab, summing to zero) — confirmed still text-only.
6. **9.C — z ~ √(κt) scaling**, penetration depth vs. time — confirmed still text-only.

## 8. Typos

All four previously flagged ("Download mateials," "propoerties," the stray tilde in "2̃0 min," "sesimic") are fixed. The only new one is the `\si{mW/m^{-2}}` unit issue already listed in §5.
