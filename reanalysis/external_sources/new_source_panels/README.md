# Archived panel screen

This directory records the retrospective source screen used for the staged-retention analysis. The registry contains 63 metadata or repository records, not 63 datasets used in the analysis. Six records supplied the 14 primary panels; one additional record supplied two sensitivity-only panels and 56 were excluded.

## Eligibility

A primary panel required:

- at least three fixed, source-defined biochar candidates;
- at least three complete numerical condition strata shared by every candidate;
- directly tabulated responses with higher values defined as better;
- candidate and condition identities recoverable without inferring them from rounded material properties;
- no response-dependent choice of the two pilot conditions.

Panels were excluded when the candidate set was incomplete, fewer than three common condition strata were available, material provenance was insufficient, responses required endpoint reorientation, or the archive did not contain usable candidate-by-condition data. `postfreeze_unified_source_screening_registry.csv` records the decision and reason for every screened record.

## Included source records

| Data record | Associated article | Primary panels |
| --- | --- | ---: |
| `10.1016/j.jiec.2024.10.004` | `10.1016/j.jiec.2024.10.004` | 2 |
| `10.5061/dryad.3xsj3txf4` | `10.1098/rsos.201789` | 1 |
| `10.5061/dryad.pc866t1w7` | `10.1007/s42773-023-00263-5` | 1 |
| `10.17632/g86tgcy22j.1` | `10.1039/D3EM00246B` | 3 |
| `10.17632/2w4st83pch.2` | `10.1016/j.heliyon.2020.e05388` | 3 |
| `10.17632/w882jvwwfk.1` | `10.1016/j.mtcomm.2025.113636` | 4 |

The screen combined a frozen 16-article metadata batch, a complete Dryad API query for `biochar` on 16 July 2026, and a targeted non-exhaustive Mendeley Data search. It is an auditable source screen, not a systematic review or probability sample of the biochar literature.

## Files

- `postfreeze_unified_source_screening_registry.csv`: all 63 screened records and decisions.
- `postfreeze_v1_locked_panel_audit.csv`: panel-level inclusion and exclusion audit.
- `postfreeze_v1_locked_panels_combined.csv`: 488 directly tabulated candidate-condition responses used by the frozen primary evaluation.
