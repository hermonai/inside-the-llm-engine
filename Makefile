PYTHON ?= python3
export PATH := /Library/TeX/texbin:$(PATH)
.PHONY: textbook textbook-check
textbook:
	$(PYTHON) scripts/build-textbook.py
textbook-check:
	$(PYTHON) scripts/check-textbook.py
