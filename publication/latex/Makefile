.PHONY: all
all:
	mkdir -p ../../build/latex
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../../build/latex main.tex
	mkdir -p ../../output/pdf
	cp ../../build/latex/main.pdf ../../output/pdf/inside-the-llm-engine-tex.pdf
