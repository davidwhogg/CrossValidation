.SUFFIXES: .tex .pdf
all: cv.pdf figs

LATEX = pdflatex
CHECK_RERUN = grep Rerun $*.log

figs:
	python ../generate_figs.py

%.pdf: %.tex
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
