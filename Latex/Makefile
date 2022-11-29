# file: Makefile	George B. Moody (george@mit.edu)

template.pdf:	template.tex
	pdflatex template
	bibtex template
	pdflatex template
	pdflatex template

template.ps:	template.dvi
	dvips -Ppdf -t letter -o template.ps template

template.dvi:	template.tex
	latex template	# creates template.aux, needed by bibtex
	bibtex template	# creates template.bbl, needed by latex
	latex template	# merges references
	latex template  # produces final copy with correct citation numbers

kit:
	make clean
	cd ..; tar cfvz latex.tar.gz latex
	cd ..; rm -f latex.zip; zip -r latex.zip latex

clean:
	cd example1; make clean
	cd example2; make clean
	rm -f template.aux template.bbl template.blg template.dvi template.log
	rm -f template.ps *~