.SUFFIXES: .tex .pdf
all: cv.pdf

LATEX = pdflatex
CHECK_RERUN = grep Rerun $*.log

%.pdf: %.tex
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
