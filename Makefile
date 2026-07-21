PYTHON ?= python

.PHONY: verify locked common-weighting candidate-evidence comparators

verify:
	$(PYTHON) reanalysis/scripts/verify_release.py

locked:
	$(PYTHON) reanalysis/scripts/evaluate_locked_retention_protocol.py
	$(PYTHON) reanalysis/scripts/evaluate_locked_retention_sensitivity.py
	$(PYTHON) reanalysis/scripts/evaluate_retention_comparators.py
	$(PYTHON) reanalysis/scripts/write_locked_protocol_report.py
	$(PYTHON) reanalysis/scripts/verify_release.py

common-weighting:
	$(PYTHON) reanalysis/scripts/compute_holdout_common_weighting.py

candidate-evidence:
	$(PYTHON) reanalysis/scripts/build_candidate_evidence.py

comparators:
	$(PYTHON) reanalysis/scripts/evaluate_retention_comparators.py
