# Makefile for LaTeX Geophysics Notes
#
# Targets
#   all        build geophysics_notes.pdf and remove all temporary files
#   clean      remove temporary files only (keep PDF)
#   distclean  remove temporary files AND the PDF

.PHONY: all clean distclean

DOC = geophysics_notes

SOURCES = geophysics_notes.tex \
	preface/preface.tex \
	introduction/introduction.tex \
	potential/potential.tex \
	gravity/gravity.tex \
	gravity/isostasy.tex \
	magnetics/magnetics.tex \
	magnetics/paleomagnetics.tex \
	geothermics/geothermics_1.tex \
	geothermics/geothermics_2.tex \
	electromagnetics/electromagnetics.tex \
	seismology/seismology.tex \
	seismology/seismics.tex \
	numerical/numerical.tex \
	numerical/differences.tex \
	numerical/fourier_series.tex \
	numerical/fourier_transforms.tex \
	numerical/inversion.tex \
	numerical/inversion2.tex \
	physprop/properties.tex \
	math_primer/math_primer.tex \
	math_primer/algebra.tex \
	math_primer/coordinates.tex \
	math_primer/calculus.tex \
	math_primer/tod.tex \
	math_primer/toi.tex \
	math_primer/vectors.tex \
	math_primer/linear_algebra.tex \
	math_primer/complex_numbers.tex \
	math_primer/vector_calculus.tex \
	diffusion/diffusion.tex \
	waves/waves.tex

BIBFILE = ~/Library/texmf/bibtex/bib/ref.bib

# bibtex writes each .bbl into the same directory as the .aux file,
# so the targets must include the subdirectory path.
CHAPTER_BBLS = \
	gravity/gravity.bbl \
	gravity/isostasy.bbl \
	magnetics/magnetics.bbl \
	geothermics/geothermics_1.bbl \
	geothermics/geothermics_2.bbl \
	electromagnetics/electromagnetics.bbl \
	seismology/seismology.bbl \
	seismology/seismics.bbl \
	numerical/inversion.bbl \
	physprop/properties.bbl \
	math_primer/algebra.bbl \
	math_primer/calculus.bbl \
	math_primer/vector_calculus.bbl

TMPFILES = \
	*.aux */*.aux \
	*.bbl */*.bbl \
	*.blg */*.blg \
	*.toc \
	*.log */*.log \
	*.lof *.lot \
	*.out */*.out \
	*.synctex.gz \
	*.fls *.fdb_latexmk

# Build PDF then remove all intermediate files.
all: $(DOC).pdf
	rm -f $(TMPFILES)

# Remove intermediate files only — keep the PDF.
clean:
	rm -f $(TMPFILES)

# Remove everything including the PDF.
distclean: clean
	rm -f $(DOC).pdf

# ── Pass 1 ────────────────────────────────────────────────────────────────────
# First pdflatex run: writes the main .aux and all chapter .aux files.
$(DOC).aux: $(SOURCES)
	pdflatex $(DOC)

# ── Chapter bibliographies ────────────────────────────────────────────────────
# Each chapter .bbl depends on its own .aux (created in pass 1 above).

gravity/gravity.bbl: $(BIBFILE) $(DOC).aux
	bibtex gravity/gravity

gravity/isostasy.bbl: $(BIBFILE) $(DOC).aux
	bibtex gravity/isostasy

magnetics/magnetics.bbl: $(BIBFILE) $(DOC).aux
	bibtex magnetics/magnetics

geothermics/geothermics_1.bbl: $(BIBFILE) $(DOC).aux
	bibtex geothermics/geothermics_1

geothermics/geothermics_2.bbl: $(BIBFILE) $(DOC).aux
	bibtex geothermics/geothermics_2

electromagnetics/electromagnetics.bbl: $(BIBFILE) $(DOC).aux
	bibtex electromagnetics/electromagnetics

seismology/seismology.bbl: $(BIBFILE) $(DOC).aux
	bibtex seismology/seismology

seismology/seismics.bbl: $(BIBFILE) $(DOC).aux
	bibtex seismology/seismics

numerical/inversion.bbl: $(BIBFILE) $(DOC).aux
	bibtex numerical/inversion

physprop/properties.bbl: $(BIBFILE) $(DOC).aux
	bibtex physprop/properties

math_primer/algebra.bbl: $(BIBFILE) $(DOC).aux
	bibtex math_primer/algebra

math_primer/calculus.bbl: $(BIBFILE) $(DOC).aux
	bibtex math_primer/calculus

math_primer/vector_calculus.bbl: $(BIBFILE) $(DOC).aux
	bibtex math_primer/vector_calculus

# ── Passes 2 & 3 ─────────────────────────────────────────────────────────────
# Two more pdflatex runs to resolve citations then cross-references.
$(DOC).pdf: $(DOC).aux $(CHAPTER_BBLS)
	pdflatex $(DOC)
	pdflatex $(DOC)
