# Build the Cascade Engine paper.
#   make           — build main.pdf
#   make all       — build main.pdf + theory.pdf + related_work.pdf
#   make clean     — remove aux artifacts
#   make distclean — also remove built PDFs

LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all clean distclean

main.pdf: main.tex refs.bib
	$(LATEX) main
	$(BIBTEX) main || true
	$(LATEX) main
	$(LATEX) main

theory.pdf: theory.tex
	$(LATEX) theory
	$(LATEX) theory

related_work.pdf: related_work.tex
	$(LATEX) related_work
	$(LATEX) related_work

all: main.pdf theory.pdf related_work.pdf

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f main.pdf theory.pdf related_work.pdf
