PYTHON ?= python

.PHONY: verify locked

verify:
	$(PYTHON) reanalysis/scripts/verify_release.py

locked:
	$(PYTHON) reanalysis/scripts/evaluate_locked_retention_protocol.py
	$(PYTHON) reanalysis/scripts/evaluate_locked_retention_sensitivity.py
	$(PYTHON) reanalysis/scripts/write_locked_protocol_report.py
	$(PYTHON) reanalysis/scripts/verify_release.py
