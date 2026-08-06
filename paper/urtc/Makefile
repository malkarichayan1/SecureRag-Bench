.PHONY: pdf clean check-generated

PDF := ../../output/pdf/securerag_bench_urtc_draft.pdf

# The manuscript ``\input``s tables, headline macros, and figure data from
# ../generated/ (see scripts/import_study_bundle.py). That directory is
# empty (only .gitkeep) until a real, validated Kaggle study bundle has been
# imported; without this guard a missing input would fail deep inside
# latexmk with an opaque "file not found" error instead of a clear one.
check-generated:
	python ../../scripts/check_generated_manuscript_inputs.py

pdf: check-generated
	mkdir -p $(dir $(PDF))
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf $(PDF)

clean:
	latexmk -C
	rm -f $(PDF)
