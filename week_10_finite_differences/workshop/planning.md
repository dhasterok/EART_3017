Having looked through the progression of the course and the detail level of the earlier workshops, my overall impression is that **Week 10 is conceptually very strong**, but some activities are currently more like guided discussions than the richer investigative exercises used in Weeks 3–6. Week 10 is doing an important job: it is the point where students move from:

* gradients and curvature (Week 1),
* physical properties (Week 2),
* inversion and modelling (Week 3),
* analytic potential-field solutions (Weeks 4–7),
* analytic geothermics (Weeks 8–9),

to the idea that **modern geodynamics and geophysics are fundamentally numerical disciplines**. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

That bridge is worth making much more explicit.

## First: does the progression make sense?

Yes, mostly.

I see the underlying narrative as:

1. **Week 1:** fields, gradients, Laplacians.
2. **Weeks 4–9:** analytic solutions to increasingly complex problems.
3. **Week 10:** what happens when the assumptions needed for analytic solutions fail?

That is a very logical endpoint. In fact, I would emphasize this more strongly in the introduction:

> Every analytic solution in the course was obtained by making assumptions. Finite differences are not a new kind of physics; they are what we use when we can no longer make those assumptions.

At present, that theme appears in the warm-up and briefly in the "analytic solutions break down" section, but it could become the organizing principle of the whole workshop. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

***

# Activity 10.A

## Current strength

The derivation from conservation of energy is important and should stay. It connects numerical methods to physics rather than mathematics. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

## Current weakness

Compared with activities such as:

* Activity 4.A (porosity compaction),
* Activity 5.A (isostatic reasoning),
* Activity 3.B (linearity classification),

students are largely following a derivation rather than making decisions.

## Suggested enhancement

Add a section:

### "Which Finite Difference Equation is Wrong?"

Present three update equations:

$$
\frac{\partial T_i}{\partial t}
=
\kappa
\frac{T_{i+1}-2T_i+T_{i-1}}
{\Delta z^2}
$$

$$
\frac{\partial T_i}{\partial t}
=
\kappa
\frac{T_{i+1}+T_i+T_{i-1}}
{\Delta z^2}
$$

$$
\frac{\partial T_i}{\partial t}
=
\kappa
\frac{T_{i+1}-T_{i-1}}
{\Delta z}
$$

and ask:

* Which conserves energy?
* Which smooths temperature?
* Which represents a gradient rather than diffusion?
* Which would lead to runaway heating?

This ties directly back to Week 1 gradients vs Laplacians.

***

# Activity 10.B

This is probably the strongest activity in the week.

It resembles the successful physical demonstrations in Weeks 1, 2 and 6 and gives students an embodied understanding of diffusion. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

I would keep it largely unchanged.

### Small addition

After several iterations ask:

> "Which Week 1 quantity are you implicitly calculating each timestep?"

Desired answer:

> The node is responding to local curvature (Laplacian).

This closes a beautiful loop back to Week 1.

***

# Activity 10.C

This is currently the thinnest activity.

The idea is excellent, but the current questions could be answered in a few minutes. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

## Suggested redesign

Instead of identifying boundary conditions, give groups four geological situations:

### Scenario 1

Mid-ocean ridge lithosphere.

### Scenario 2

Stable continental craton.

### Scenario 3

Cooling granite pluton.

### Scenario 4

Antarctic ice sheet geothermal model.

For each group:

1. Choose top boundary condition.
2. Choose bottom boundary condition.
3. Decide whether each is Dirichlet or Neumann.
4. Defend choices.
5. Explain what assumptions are being made.

This mirrors Activity 2.B, where students must justify modelling choices rather than identify definitions.

### Why this works

It forces them to think like modellers rather than equation users.

***

# "When Analytic Solutions Break Down"

Honestly, I think this deserves becoming a proper activity rather than a discussion.

The idea is central to the week.

## Suggested activity

Create a table.

| Problem                     | Analytic? | Why / Why not |
| --------------------------- | --------- | ------------- |
| Uniform half-space cooling  | Yes       |               |
| Layered crust               | Maybe     |               |
| Variable conductivity       |           |               |
| Magma intrusion             |           |               |
| Iberia basin model          |           |               |
| Antarctica geothermal model |           |               |

Students determine:

* which assumptions fail,
* which require numerical methods,
* whether finite differences are actually needed.

This would connect directly back to almost every previous week.

***

# Activity 10.D

Conceptually good but slightly abstract.

## Suggested enhancement

I would make stability a prediction activity.

Give groups three hypothetical models:

### Model A

$$
\Delta t = 0.1 \frac{\Delta x^2}{\kappa}
$$

### Model B

$$
\Delta t = 0.9 \frac{\Delta x^2}{2\kappa}
$$

### Model C

$$
\Delta t = 10 \frac{\Delta x^2}{\kappa}
$$

Ask:

* Which is most accurate?
* Which is fastest?
* Which blows up?
* Which would you choose for continental lithosphere?

This feels more like the investigative style seen in Weeks 3–5.

***

# Activity 10.E

I actually think this is a very valuable addition. It introduces something many undergraduate courses never discuss:

> verification versus validation.

That idea links naturally to inversion (Week 3) and model testing. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

### Suggested addition

Use actual solutions from the course.

Ask groups:

* Which Week 8 solution would you use as a benchmark?
* Which Week 9 solution?
* What would indicate the code is broken?

This provides a satisfying callback to material they already know.

***

# Activity 10.F

Conceptually excellent, but it currently reads more like a worksheet than a culminating activity. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)

I would turn this into a capstone exercise.

## Suggested replacement

### "Design a Geodynamic Model"

Give groups a geological problem:

* Antarctic heat flow.
* Basin maturation.
* Cooling pluton.
* Subduction wedge.

For each, they must specify:

1. Domain geometry.
2. Physical properties.
3. Boundary conditions.
4. Initial conditions.
5. Analytic or numerical?
6. Why?

This would directly integrate:

* Week 2 properties,
* Week 5 lithosphere structure,
* Weeks 8–9 heat flow,
* Week 10 numerical methods.

It feels like a genuine synthesis activity.

***

# One thing I think is missing entirely

Throughout the course you repeatedly emphasize that:

> geophysics measures fields but interprets physical properties.

Week 10 should probably contain:

## "The Model Builder's Workflow"

Something like:

```text
Geology
   ↓
Physical Properties
   ↓
Forward Model
   ↓
Field Prediction
   ↓
Compare with Observations
```

Students have seen every part of this chain separately:

* Week 2 → physical properties.
* Week 3 → inversion.
* Weeks 4–9 → forward models.
* Week 10 → numerical implementation.

But nowhere are they explicitly connected into the workflow used in modern geodynamic modelling.

A short activity where students identify every component needed to build, for example, an Antarctic geothermal model would make Week 10 feel like the culmination of the course rather than simply "the numerical methods week".

## Overall recommendation

If I were revising Week 10, I would:

1. Keep 10.A and 10.B largely intact.
2. Expand 10.C into scenario-based model design.
3. Convert "When Analytic Solutions Break Down" into a full activity.
4. Make 10.D more predictive/comparative.
5. Keep 10.E.
6. Turn 10.F into a capstone geophysical/geodynamic model design exercise.

That would bring Week 10 much closer to the richness of Weeks 3–6 while making the workshop feel like the intellectual culmination of the entire course. [\[workshop_booklet \| PDF\]](https://uao365-my.sharepoint.com/personal/a1638626_adelaide_edu_au/Documents/Microsoft%20Copilot%20Chat%20Files/workshop_booklet.pdf)
