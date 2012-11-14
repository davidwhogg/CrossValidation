.SUFFIXES: .tex .pdf
all: cv_technical.pdf

LATEX = pdflatex
CHECK_RERUN = grep Rerun $*.log

figures:
	python ../generate_figs.py

%.pdf: %.tex figures
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
