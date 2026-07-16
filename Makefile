# =========================================================
# Root Makefile — EART 3017
#
# Targets:
#   make              — compile everything (booklet + workshops + problem sets)
#   make booklet      — compile the combined workshop booklet
#   make workshops    — compile all weekly workshop PDFs
#   make workshop WEEK=N  — compile one week's workshop
#   make PS           — compile all problem sets
#   make PS WEEK=N    — compile one week's problem set
#   make clean        — remove LaTeX auxiliary files
# =========================================================

SHELL   := /bin/bash
LATEX   := pdflatex
FLAGS   := -interaction=nonstopmode -halt-on-error
AUXEXTS := aux log out toc lof lot nav snm vrb bbl blg synctex.gz

GREEN := \033[0;32m
RED   := \033[0;31m
NC    := \033[0m

WORKSHOPS   := $(shell find . -type f -path './week_*_*/workshop/week*_workshop.tex' | sort)
PROBLEMSETS := $(shell find . -type f -path './week_*_*/assignment/week*_problem_set.tex' | sort)

# Shell function used inside recipe lines (must be defined at start of each recipe)
COMPILE_FN = compile() { \
	dir=$$(dirname "$$1"); base=$$(basename "$$1"); log=$$(mktemp); \
	printf "▶  %-62s" "$$1"; \
	if ( cd "$$dir" && \
	     $(LATEX) $(FLAGS) "$$base" >"$$log" 2>&1 && \
	     $(LATEX) $(FLAGS) "$$base" >>"$$log" 2>&1 ); then \
		printf "$(GREEN)ok$(NC)\n"; rm -f "$$log"; \
	else \
		printf "$(RED)FAILED$(NC)\n"; \
		sed -n '/^! /,/^l\.[0-9]/p' "$$log" 2>/dev/null || tail -20 "$$log"; \
		rm -f "$$log"; return 1; \
	fi; \
}

.PHONY: all booklet workshops workshop PS clean

# ---- default: everything ----------------------------------
all: booklet workshops PS

# ---- combined booklet -------------------------------------
booklet:
	@$(COMPILE_FN); compile workshop_booklet.tex
	@$(MAKE) --no-print-directory clean

# ---- all workshops ----------------------------------------
workshops:
	@$(COMPILE_FN); \
	failed=(); \
	for tex in $(WORKSHOPS); do \
		compile "$$tex" || failed+=("$$tex"); \
	done; \
	$(MAKE) --no-print-directory clean; \
	if [ $${#failed[@]} -gt 0 ]; then \
		printf "\n$(RED)Workshops failed:$(NC)\n"; \
		printf '  %s\n' "$${failed[@]}"; exit 1; \
	fi

# ---- one workshop: make workshop WEEK=N -------------------
workshop:
	$(if $(WEEK),,$(error Specify WEEK, e.g.:  make workshop WEEK=4))
	@$(COMPILE_FN); \
	tex=$$(find . -type f -path './week_$(WEEK)_*/workshop/week$(WEEK)_workshop.tex'); \
	[ -n "$$tex" ] || { printf "$(RED)No workshop found for week $(WEEK)$(NC)\n"; exit 1; }; \
	compile "$$tex"
	@$(MAKE) --no-print-directory clean

# ---- problem sets: make PS [WEEK=N] -----------------------
PS:
ifdef WEEK
	@$(COMPILE_FN); \
	tex=$$(find . -type f -path './week_$(WEEK)_*/assignment/week$(WEEK)_problem_set.tex'); \
	[ -n "$$tex" ] || { printf "$(RED)No problem set found for week $(WEEK)$(NC)\n"; exit 1; }; \
	compile "$$tex"
	@$(MAKE) --no-print-directory clean
else
	@$(COMPILE_FN); \
	failed=(); \
	for tex in $(PROBLEMSETS); do \
		compile "$$tex" || failed+=("$$tex"); \
	done; \
	$(MAKE) --no-print-directory clean; \
	if [ $${#failed[@]} -gt 0 ]; then \
		printf "\n$(RED)Problem sets failed:$(NC)\n"; \
		printf '  %s\n' "$${failed[@]}"; exit 1; \
	fi
endif

# ---- remove auxiliary files --------------------------------
clean:
	@for ext in $(AUXEXTS); do \
		find . -type f -name "*.$$ext" -delete; \
	done
	@printf "Cleaned auxiliary files.\n"
