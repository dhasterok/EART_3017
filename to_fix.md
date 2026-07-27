Good timing for a consistency check — I found several genuine discrepancies, including one activity that's missing from its timeline entirely. Here's the full audit.

## 1. Timeline vs. activity sheet consistency

**Clean weeks: 1, 2, 4.** Activities present, correctly labeled, correctly ordered, times continuous.

**Week 3 — timing conflict.** The timeline reads: 3.B *Linear or Non-linear?* at 60–90 min, but 3.C *Buried Sphere — Background* starts at **70–85**, followed by Jacobian Derivatives at 85–105. So 3.B overlaps both following segments. It looks like the 3.C segments were inserted without updating 3.B's end time. A repair that preserves the 170-min total: trim the mini-lecture to 10–35, then 3.A 35–55, 3.B 55–85, Background 85–95, Jacobian 95–110, Investigation 110–150 — or simply shorten 3.B to 60–70 if the equation-handout discussion can run that fast (25–30 min feels more realistic for classify + propose linearization + notes). Also, the timeline calls 3.C "Buried Sphere — Background" but the sheet is titled *Inversion of a Buried Sphere*; worth matching since students navigate by sheet titles.

**Week 5 — labels off by one.** The sheets are: 5.A *Oceanic vs. Continental Crust* (containing both Part 1 and Part 2 internally), 5.B *Isostatic Gravity Correction*, 5.C *Numerical Demonstration: Isostatic Gravity Along a Cross Section*. The timeline instead labels Part 2 as "Activity 5.B," the gravity correction as "Activity 5.C," and leaves the numerical demonstration unlabeled. Fix: 20–50 → Activity 5.A Part 1; 60–85 → Activity 5.A Part 2; 85–125 → Activity 5.B; 125–155 → Activity 5.C.

**Week 6 — name variant.** Timeline: "Activity 6.B: Magnetic Anomalies vs. Gravity"; sheet: *Magnetic Structures*. Minor, but pick one.

**Week 7 — labels off by one from Fold Belt onward.** The timeline has "Fold Belt Survey" unlabeled at 25–60, then calls Nevada "7.B" and Tower Hill "7.C." The sheets are 7.B *Fold Belt Survey*, 7.C *Nevada Gravity*, 7.D *Tower Hill Magnetics*. Same fix pattern as Week 5. Separately, Week 7 is the only week with a chapter-level Key Ideas block after the preview in the session flow — intentional, or a structural one-off worth propagating or removing?

**Week 8 — an activity is missing from the timeline.** This is the most substantive finding. The sheets run 8.A through 8.E, but the timeline only schedules four items:

| Timeline entry | Actual sheet |
|---|---|
| 0–15 Warm-up — Thermal Refraction | Activity 8.A *Thermal Refraction* (unlabeled) |
| — (absent) | Activity 8.B *Structures and Steady-State Fields* |
| 30–65 "Activity 8.A" Layered/Anisotropic Conductivity | Activity 8.C |
| 70–110 "Activity 8.B" Heat Production from Seismic Velocity? | Activity 8.D |
| 110–150 "Activity 8.C" Building a Geotherm | Activity 8.E |

So 8.B (flux refraction at conductivity contrasts — the field-line bending activity, 3 questions) never appears in the schedule, and every label is shifted two positions. At 170 min the session has no room for it: you'd need to find ~20–25 min, either by extending toward 180 (matching Weeks 5 and 7) and trimming the mini-lecture, or by folding 8.B's core figure into the 15–30 mini-lecture and dropping it as a standalone activity. Given its Key Idea is the analogy with current and magnetic field-line refraction (a nice Week 6 callback), it'd be a shame to lose it silently.

**Week 9 — type label.** Timeline says "Activity 9.A: Wedge vs. Orogen"; the sheet header is *Warm-up 9.A*. Since you have a convention that Demos share the Activity counter, decide whether Warm-ups do too and label consistently.

**Week 10 — 10.E and 10.F unscheduled** (acknowledging you're mid-revision here). The timeline covers 10.A–10.D plus an unlabeled "When Analytic Solutions Break Down" discussion at 120–145, but 10.E *Verification Against Analytic Solutions* and 10.F *Building Geological Intuition with Numerical Experiments* don't appear. Note that the 120–145 discussion is *not* 10.E — 10.E is about verification/convergence, whereas the discussion is about motivation for numerics (which arguably duplicates the 0–10 warm-up "What breaks down?"). There are also three name variants: timeline "Human Finite Difference Solver" vs. sheet *Solving the Heat Equation as a Human Computer*; "Boundary Conditions as Geological Assumptions" vs. *…and Geological Meaning*; "Numerical Stability and Information Transfer" vs. *Stability, Time Stepping, and Information Transfer*.

**One possible content duplication, Week 2.** The 45–60 "Simulation Walkthrough" (pre-built Python visualization of transit times through three media, histogram shapes vs. mixing laws) describes essentially the same content as Demo 2.C *Path Simulation* at 125–155. If the walkthrough is a deliberate preview and the Demo goes deeper, fine — but as written they read as the same demonstration scheduled twice. Worth a sentence in each distinguishing their scope.

## 2. Time budgets

Session totals are 165, 175, 170, 170, 180, **240**, 180, 170, 175, **220** min. If your timetabled slot is 3 hours, Weeks 6 and 10 overrun by 60 and 40 min respectively — and Week 6 has only one 10-min break in 4 hours, which is rough on everyone. If those weeks genuinely have longer slots, ignore this; otherwise Week 6 is the one needing surgery (6.D at 50 min with 10 questions is the natural place to split or trim, or 6.C's three stations could rotate faster).

Within sessions, most allocations look sane at roughly 8–12 min per question for group work. The ones running hot, at ~4–5 min per question:

- **1.B Sheet Model, 15 min for 6 questions.** Even as instructor-led predict-observe, that's brisk. Week 1 ends at 165, so you have 15 min of slack against a 180 slot if you want to stretch it to 25.
- **5.C Numerical Demonstration, 30 min for 7 questions.** Workable if students answer while you drive the GUI, but the sheet questions need to be short-answer.
- **7.C Nevada, 50 min** for spectra → wavelength ID → filter design → Parker–Oldenburg → aliasing effects. This is the densest single block in the booklet. It's fine if the code is fully scaffolded and students mainly interpret outputs, but any live debugging blows this up. I'd protect it by trimming the Fold Belt debrief rather than the other way around.
- **8.E Building a Geotherm (timeline "8.C"), 40 min for 7 questions** involving actual computation — borderline, same caveat about scaffolding.
- **9.D Half-space/Plate Cooling, 35 min for 7 questions** covering Fourier modes, decay times, and two cooling models. This one I'd genuinely lengthen; the material is conceptually the hardest of Week 9, and 9.C at 45 min for 4 questions looks comparatively generous — shifting 10 min from 9.C to 9.D would balance them.

Conversely, 10.C (20 min) and 10.D (40 min) have no questions on their sheets yet, so those can't be assessed until Week 10 revisions land.

## 3. Summary figures for Key Ideas

You already have effective Key Ideas figures in 1.A (estimation flowchart), 2.A (three-geometry cartoon), 3.B (linearity decision flowchart), 4.B (narrow/broad depth cartoon), 6.A (dipole), 10.A (stencil), 10.C (Dirichlet/Neumann), and 10.D (stable/unstable). The strongest candidates for additions, in rough priority order:

1. **8.A Thermal Refraction — kinked geotherm.** Two T(z) profiles side by side (exposed basement vs. sediment-over-basement), same q arrows, visibly different slopes in the sediment. The Key Idea "conductivity controls gradients" is inherently a picture, and this figure would get reused mentally for the rest of Weeks 8–10. (Also: the first paragraph of 8.A's Key Ideas is duplicated nearly verbatim — an editing leftover to delete.)

2. **9.D — heat flow and subsidence vs. age.** Two small panels: q vs. t and subsidence vs. √t, with half-space and plate curves diverging at t ~ τ. This is *the* canonical figure of the topic, and it directly illustrates all three Key Idea bullets (two observables, no internal length scale, early-time agreement).

3. **8.C (timeline 8.A) — k∥/k⊥ layered block.** A single block with layering, two flux arrows (along vs. across), labeled arithmetic and harmonic. Deliberately echo the 2.A cartoon style — the visual callback reinforces that the mixing-law framework transfers from Week 2 to heat flow, which is exactly the kind of transferable principle your Key Ideas convention is built for.

4. **6.D — Curie depth vs. Moho columns.** Craton and rift columns side by side with the Curie isotherm crossing at different depths relative to the Moho. Makes "thermal boundary, not compositional" instantly legible and supports the Antarctica case study.

5. **5.B — slab cancellation.** Topographic slab (+2πGρch) over root slab (−2πGρch) with the anomalies summing to zero. Small, cheap in TikZ, and it's the single equation-as-picture of the week.

6. **3.C — grid search vs. inversion path.** One misfit-surface contour panel with a grid of dots (exhaustive sampling) and one with an iterative descent path. Directly contrasts the two paragraphs of the Key Ideas.

7. **9.C — z ~ √(κt) scaling.** Penetration depth vs. time with markers for a daily wave, glacial cycle, and orogenic timescale. Optional, but scaling laws stick better as log-log lines than sentences.

I'd stop there. 2.C, 5.A, 8.D, and 8.E have Key Ideas that are genuinely argumentative rather than geometric (diagnosis, epistemology of proxies, boundary-condition dependence), and forcing figures onto those would restate specifics rather than encode transferable structure — against your own convention.

One last small thing while you're in the timelines: "Download mateials" (Week 1), "propoerties" (Week 2 warm-up), "2̃0 min" (Week 2, a stray tilde), and "sesimic" (Week 8, 8.D description) are all in session-flow text.

Want me to draft the corrected Session Flow blocks for Weeks 3, 5, 7, and 8 with the relabeling and repaired times, so you can paste them into the LaTeX?