.PHONY: pdf clean

PDF := ../../output/pdf/securerag_bench_urtc_draft.pdf

pdf:
	mkdir -p $(dir $(PDF))
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf $(PDF)

clean:
	latexmk -C
	rm -f $(PDF)
