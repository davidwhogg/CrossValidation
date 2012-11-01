.SUFFIXES: .tex .pdf
all: cv.pdf figures

LATEX = pdflatex
CHECK_RERUN = grep Rerun $*.log

figures:
	python ../generate_figs.py

%.pdf: %.tex
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
