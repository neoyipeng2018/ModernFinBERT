#!/bin/bash
# Build the paper PDF
# Requires: pdflatex, bibtex (from a TeX distribution like MacTeX or TeX Live)

set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Build complete: main.pdf"
