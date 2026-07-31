Re-verified against the current state of every file (compiled each week standalone and checked the rendered PDF, not just source text). Most of the previous list is now fixed — removed below. What's left is mostly the time overruns, plus a handful of new things the verification pass turned up.

## 0. Root cause behind the recurring label bugs — fixed, two call-sites still need syncing

`\timeline`'s `warmup` type (in `aucourse.sty`) now correctly steps the letter counter and prints "Warm-up," matching `demo`. Weeks 5 and 7's stray `[slot]`-tagged lettered entries are fixed (5.C Numerical Demonstration and 7.B Fold Belt Survey both letter and print correctly now). Two wording mismatches remain from this same family of bug:

- **Week 7 (`week7_timeline.tex:37`):** the warm-up is tagged `[item]` with "Warm-up ---" baked into the title text, so it renders as "Activity 7.A: Warm-up — Two Bumps, One Line" while the sheet says "Warm-up 7.A." Change the tag to `[warmup]` and drop the redundant "Warm-up ---" prefix from the title text.
- **Week 8 (`week8_timeline.tex:30`):** now tagged `[warmup]`, so the timeline prints "Warm-up 8.A," but `activity_refraction.tex` uses a plain `\begin{activity}{Thermal Refraction}` and prints "Activity 8.A" — the opposite mismatch from before. Add `[Warm-up]` to the sheet's `\begin{activity}` call to match Weeks 7 and 9's convention.

## 1. Timeline vs. activity sheet consistency

**Clean: Weeks 1, 2, 4, 5, 6, 8, 9.** All labels/letters now sync correctly between timeline and sheets (verified against compiled PDFs). Week 2's title mismatch ("and" vs "&") and the Simulation-Walkthrough/Demo-2.C duplication are both resolved — the walkthrough was replaced with the in-person "Crowd Conductivity" demo, which is a genuinely different activity. Week 6's naming, 6.D title, 6.C self-reference, and the Curie-temperature inconsistency are all fixed.

**Week 3 — new letter-desync.** The old title mismatch is fixed (timeline now says "Inversion of a Buried Sphere," matching the sheet). But `week3_timeline.tex:74`, `{Buried Sphere --- Background}`, is tagged `[item]` and shouldn't be — it steps the counter for content that's part of the same sheet as the main Buried Sphere activity (like the `[slot]`-tagged "Jacobian Derivatives" entry right after it correctly does). Right now the timeline shows the final entry as "3.D: Inversion of a Buried Sphere," but the actual sheet page says "Activity 3.C." Fix: change line 74 to `[slot]`.

**Week 7.** Letters now sync (Fold Belt Survey is properly lettered 7.B). The Nevada blurb's stale aliasing reference and the warm-up's dangling "isostatic filtering" promise are both still open — see below. RTP's "phase filter, not amplitude filter" repetition is much reduced (no longer in the timeline blurb for Nevada; still appears in the Tower Hill sheet as intro/learning-outcome/question, which is reasonable) but Tower Hill's *timeline* blurb still adds its own echo: "Key emphasis: RTP rotates phase; it does not filter noise" — a fourth restatement of the same idea, worth cutting.

**Week 9.** Warm-up wording is fixed (`[warmup]` tag matches the sheet's `[Warm-up]`). The plate-cooling title mismatch and its stale Fourier-mode content outline are both fixed (updated this session). Both "Activity 9.A" cross-references (in the sheet and in `half_space_figure.py`'s docstring) are fixed. `activity_collisions.tex` has been moved to `activities/unused/` — the warm-up's closing promise ("the orogen case is looked at in greater detail in scenarios examined later") is still honored, though: `activity_transient_scenarios.tex` has its own orogenic-thickening scenario, so this isn't a broken promise after all. Still open: `activity_thermal_timescales.tex`'s figure caption still says "three times, $t_1<t_2<t_3$" while the figure now plots six curves ($t_0$–$t_5$); and it still uses "Key Idea" (singular) where every other sheet says "Key Ideas."

**Week 10.** `activity_stability.tex`'s wrong `\input` is fixed — `explicit_table`/`implicit_table` are each `\input` exactly once now, in the right places. 10.D's title now matches between timeline and sheet. Still mismatched: timeline says "Boundary Conditions as Geological Assumptions," the sheet (`activity_geologic_boundaries.tex:2`) still says "...and Geological Meaning." 10.E/10.F remain fully orphaned (unreferenced, not just unscheduled). The stability-threshold spoiler is still open — see below.

## 2. Time budgets

Current session totals: **165, 170, 175, 170, 170, 180, 180, 175, 175, 180** (Weeks 1–10). Target is 160–170. This is a large improvement — Week 6 dropped from 240 to 180 and Week 10 from 220 to 180 — but overruns are still the main thing left:

- **Weeks 6, 7, 10 sit at 180** (10 over target) — the biggest remaining gaps, though Week 6 in particular is no longer the outlier crisis it was.
- **Weeks 3, 8, 9 sit at 175** (5 over).
- **Weeks 1, 2, 4, 5 are within target** (165–170), no action needed.

## 3. Redundant or duplicated Key Ideas

- **3.A/3.B/3.C**: "local vs. global minimum" is still repeated almost word-for-word across all three sheets' Key Ideas.
- **4.C** (`activity_buried_sphere.tex`, Purpose section, not Key Ideas): new finding — two consecutive sentences both start "Such a relationship is useful for..." (estimating depth to structures / back-of-envelope estimates in the field). Reads like an editing leftover; 4.C's own Key Ideas are otherwise clean now, and the old 4.B/4.C Key-Ideas overlap is gone.
- **Week 7**: see the Tower Hill "Key emphasis" line noted in §1 — the only remaining redundant echo of the RTP phase-filter idea.
- **1.A, 8.A**: fixed — no more duplicated Key Ideas paragraphs.

## 4. Timeline blurbs that give away the answer

- **Week 3**: the Investigation blurb (`week3_timeline.tex:97`) still states the $a^3\Delta\rho$ trade-off and resolution-ellipse content as bullet points ahead of the activity.
- **Week 4**: the Buried Sphere blurb still states the answer outright — "Explain why increasing depth increases width but decreasing $\Delta\rho$ does not" is the conclusion, not a lead-in question.
- **Week 10**: the stability threshold ($\Delta t < \Delta x^2/2\kappa$) is still given twice in the timeline (mini-lecture and 10.D's own intro) before the sheet's Q2 has students find it themselves.
- **Week 7 Fold Belt**: fixed — the blurb now just names the topic ("the impact of aliasing") without previewing the conclusion. **Tower Hill**: fixed similarly, though see the RTP echo above.

## 5. Other content/reference bugs

- `week_8_steady_state_heat/workshop/activities/activity_velocity_heat_production.tex:110,150` — `\si{mW.m^2}` is missing the negative sign on the exponent (should be `\si{mW.m^{-2}}`, as it correctly is at lines 100, 134, and 136 in the same file). New finding — the original `mW/m^{-2}` bug is fixed, but this is a different instance of the same unit slipping through elsewhere in the file.
- Everything else previously listed here — `fig:layres`, the "Activity 9.A" cross-references, the thermal-timescales stale outline, the missing Structures blurb — is fixed.

## 6. Orphaned or superseded files — resolved

All of the previously-flagged orphaned files (Weeks 2, 7, 9, 10) have been moved into `activities/unused/` subfolders rather than left loose where they could be edited by mistake. Week 8 also archived three additional unused drafts. Nothing further to do here.

## 7. Summary figures for Key Ideas — mostly done

Four of the seven originally suggested have been added since the last pass:

- ~~3.C grid search vs. inversion path~~ — done.
- ~~5.B slab cancellation~~ — done (topographic slab over root slab, "cancels").
- ~~6.D Curie depth vs. Moho columns~~ — done, via the real Antarctica cross-section figure (craton/rift framing in the caption).
- ~~8.C k∥/k⊥ layered block~~ — done (the parallel/perpendicular flow cartoons with SLOW/GO markers).

Still open:

1. **8.A Thermal Refraction — kinked geotherm.** Two T(z) profiles (exposed basement vs. sediment-over-basement).
2. **9.D — heat flow and subsidence vs. age.** Still text-only.
3. **9.C — z ~ √(κt) scaling.** Still text-only.

## 8. Typos

All previously-flagged typos are fixed. The only remaining unit issue is the `mW.m^2` (missing negative exponent) noted in §5.
