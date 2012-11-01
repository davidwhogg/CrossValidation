.SUFFIXES: .tex .pdf
all: cv.pdf cv_technical.pdf figures

LATEX = pdflatex
CHECK_RERUN = grep Rerun $*.log

figures:
	python ../generate_figs.py

%.pdf: %.tex figures
	${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
