PYTHON ?= python3
export PATH := /Library/TeX/texbin:$(PATH)
.PHONY: textbook textbook-check status
textbook:
	$(PYTHON) scripts/build-textbook.py
textbook-check:
	$(PYTHON) scripts/check-textbook.py
status:
	$(PYTHON) scripts/check-textbook.py --write-status
