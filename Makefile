PYTHON ?= python

.PHONY: verify staged-retention common-weighting candidate-evidence comparators

verify:
	$(PYTHON) analysis/scripts/verify_release.py

staged-retention:
	$(PYTHON) analysis/scripts/evaluate_staged_retention.py
	$(PYTHON) analysis/scripts/evaluate_staged_retention_sensitivity.py
	$(PYTHON) analysis/scripts/evaluate_retention_comparators.py
	$(PYTHON) analysis/scripts/write_staged_retention_report.py
	$(PYTHON) analysis/scripts/verify_release.py

common-weighting:
	$(PYTHON) analysis/scripts/compute_holdout_common_weighting.py

candidate-evidence:
	$(PYTHON) analysis/scripts/build_candidate_evidence.py

comparators:
	$(PYTHON) analysis/scripts/evaluate_retention_comparators.py
