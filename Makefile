PYTHON ?= python

.PHONY: verify staged-retention common-weighting candidate-evidence comparators

verify:
	$(PYTHON) code/verify_release.py

staged-retention:
	$(PYTHON) code/evaluate_staged_retention.py
	$(PYTHON) code/evaluate_staged_retention_sensitivity.py
	$(PYTHON) code/evaluate_retention_comparators.py
	$(PYTHON) code/write_staged_retention_report.py
	$(PYTHON) code/verify_release.py

common-weighting:
	$(PYTHON) code/compute_holdout_common_weighting.py

candidate-evidence:
	$(PYTHON) code/build_candidate_evidence.py

comparators:
	$(PYTHON) code/evaluate_retention_comparators.py
