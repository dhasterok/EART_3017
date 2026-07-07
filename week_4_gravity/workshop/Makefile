# List of TeX files
TEXFILES = week4_density_worksheet.tex \
		   week4_sandstone_activity.tex \
           week4_formula_sheet.tex \
           week4_match3_handout.tex \
		   week4_buried_sphere_activity.tex \
           week4_superposition_components.tex \
		   week4_workshop.tex

# Generate corresponding PDF filenames
PDFS = $(TEXFILES:.tex=.pdf)

# Default target: build all PDFs
all: $(PDFS)

# Rule to compile each tex file to pdf
%.pdf: %.tex
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.nav *.snm *.vrb

# Optional: clean PDFs too
cleanall: clean
	rm -f $(PDFS)