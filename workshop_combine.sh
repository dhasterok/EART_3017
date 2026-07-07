#!/bin/bash

# ==============================================
# Combines workshop documents into a single PDF
# ==============================================

gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=workshop_booklet.pdf \
workshop_coversheet.pdf \
workshop_rubric.pdf \
week_1_fields/workshop/week1_workshop.pdf \
week_2_physical_properties/workshop/week2_workshop.pdf \
week_3_inversion/workshop/week3_workshop.pdf \
workshop_rubric.pdf \
week_4_gravity/workshop/week4_workshop.pdf \
week_5_isostasy/workshop/week5_workshop.pdf \
week_6_magnetics/workshop/week6_workshop.pdf \
week_7_fourier_transforms/workshop/week7_workshop.pdf \
workshop_rubric.pdf \
week_8_steady_state_heat/workshop/week8_workshop.pdf \
week_9_transient_heat/workshop/week9_workshop.pdf \
week_10_finite_differences/workshop/week10_workshop.pdf