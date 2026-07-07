# =========================================================
# Root Makefile for recursive LaTeX compilation
# =========================================================

SHELL := /bin/bash

LATEX       := pdflatex
LATEXFLAGS  := -interaction=nonstopmode -halt-on-error
AUXEXTS     := aux log out toc lof lot nav snm vrb

GREEN := \033[0;32m
RED   := \033[0;31m
RESET := \033[0m

# ---------------------------------------------------------
# Find all tex files
# ---------------------------------------------------------
TEX_ALL := $(shell find . -type f -name '*.tex')

PDF_ALL := $(TEX_ALL:.tex=.pdf)

# ---------------------------------------------------------
# Default target
# ---------------------------------------------------------
.PHONY: all
all:
	@$(MAKE) compile FILES="$(TEX_ALL)"

# ---------------------------------------------------------
# Week-only build
# Usage: make week WEEK=4
# ---------------------------------------------------------
.PHONY: week
week:
	@$(MAKE) compile FILES="$(shell find . -type f -path './week_$(WEEK)_*/*.tex')"

# ---------------------------------------------------------
# Week + subfolder build
# Usage: make part WEEK=4 PART=workshop
# ---------------------------------------------------------
.PHONY: part
part:
	@$(MAKE) compile FILES="$(shell find . -type f -path './week_$(WEEK)_*/$(PART)/*.tex')"

# ---------------------------------------------------------
# Compilation driver
# ---------------------------------------------------------
.PHONY: compile
compile:
	@failures=(); \
	for tex in $(FILES); do \
		dir=$$(dirname "$$tex"); \
		base=$$(basename "$$tex"); \
		log=$$(mktemp); \
        printf "▶ Compiling $$tex ... "; \
        if ( cd "$$dir" && \
        	$(LATEX) $(LATEXFLAGS) "$$base" >"$$log" 2>&1 && \
        	$(LATEX) $(LATEXFLAGS) "$$base" >>"$$log" 2>&1 ); then \
			printf "$(GREEN)Success$(RESET)\n"; \
			rm -f "$$log"; \
		else \
			printf "$(RED)Failed$(RESET)\n"; \
        	echo "---- LaTeX error output ----"; \
			sed -n '/^! /,/^l\.[0-9]\+/p' "$$log" || tail -n 20 "$$log"; \
			echo "----------------------------"; \
			failures+=("$$tex"); \
			rm -f "$$log"; \
		fi; \
	done; \
	$(MAKE) clean; \
	if [ $${#failures[@]} -ne 0 ]; then \
		echo ""; \
		echo "==================================="; \
		echo " LaTeX compilation failures:"; \
		printf '  - %s\n' "$${failures[@]}"; \
		echo "==================================="; \
		exit 1; \
	else \
		echo ""; \
		echo "All files compiled successfully."; \
	fi

# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------
.PHONY: clean
clean:
	@for ext in $(AUXEXTS); do \
		find . -type f -name "*.$$ext" -delete; \
	done