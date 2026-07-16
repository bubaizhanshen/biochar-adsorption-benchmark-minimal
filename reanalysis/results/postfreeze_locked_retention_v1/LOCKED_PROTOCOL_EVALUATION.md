# Locked candidate-retention protocol evaluation

Generated 2026-07-16 from the checked result tables in this directory.

## Scientific question

Can two boundary conditions selected without response information reduce a fixed candidate-by-condition experiment while retaining at least one observed-best candidate at the unmeasured shared conditions? The observed best is the candidate with the highest recorded mean response in the complete panel at that condition. The endpoint is a retained candidate set, not a unique winner or an absolute-response prediction.

## Protocol integrity

- Protocol: `BC-CANDIDATE-RETENTION-2026-07-16-v1`
- SHA-256: `dfb2029812af8a4a915aeb53b04412c308787544086f94bd3014684d673c345e`
- Frozen rule: assay every fixed candidate at two maximally separated condition strata and retain the union of candidates ranked in the top half at either anchor.
- Evaluation status: post-freeze retrospective external evaluation. It is not prospective validation.
- Analysis unit: source study. Panels and condition strata from the same source are not treated as independent studies.

## Locked evidence base

The primary evaluation contains 6 article-level source studies, 14 eligible panels, and 59 unmeasured query strata. It spans heavy metals, phosphate, urea, methylene blue, and 17beta-estradiol. Repository discovery was targeted rather than a probability sample.

| Source | Panels | Query strata | All best retained | Query coverage | Assay reduction | Maximum relative regret |
| --- | --- | --- | --- | --- | --- | --- |
| Ogbuagu2023 | 3 | 8 | no | 75.0% | 17.9% | 0.89% |
| Padilla2023 | 1 | 3 | yes | 100.0% | 26.7% | 0.00% |
| Soria2020 | 3 | 19 | yes | 100.0% | 30.4% | 0.00% |
| Vasconcelos2025 | 2 | 2 | yes | 100.0% | 11.1% | 0.00% |
| Wang2021 | 1 | 10 | yes | 100.0% | 16.7% | 0.00% |
| Wei2025 | 4 | 17 | yes | 100.0% | 16.0% | 0.00% |

## Primary result

The frozen rule retained every query-best candidate in 5 of 6 source studies. At the query level, 57 of 59 best candidates were retained (96.6%). Source-balanced assay-cell reduction was 19.8%, and pooled assay-cell reduction was 21.1%.

This result does not establish a safety guarantee. The exact-binomial 35.9%-99.6% interval is a small-sample reference only because the six sources were not sampled from a defined population.

## Locked failures

The frozen rule missed the observed-best candidate at two intermediate Pb concentrations in one Ogbuagu wheat-straw panel. These failures remain failures even though the response loss was small.

| Source | Panel | Condition | Excluded observed best | Raw regret | Regret / best | Range-normalized regret |
| --- | --- | --- | --- | --- | --- | --- |
| Ogbuagu2023 | Ogbuagu2023_Pb_wheat_straw_concentration | Ci=1022.934568 | WheatStraw-550C | 0.273725 | 0.89% | 0.108 |
| Ogbuagu2023 | Ogbuagu2023_Pb_wheat_straw_concentration | Ci=494.3788752 | WheatStraw-350C | 0.0211798 | 0.14% | 0.034 |

The maximum observed response loss was 0.89% of the exact-best response. A practical-equivalence margin cannot be introduced after seeing this result; it must be specified from replicate uncertainty or a minimum meaningful difference before a future evaluation.

## Panel difficulty

Panels were split descriptively according to whether the identity of the observed best candidate changed across recorded condition strata. No performance threshold was chosen from this split.

| Observed best identity | Sources | Panels | Query strata | Failed queries | Query coverage | Source-balanced assay-cell reduction |
| --- | --- | --- | --- | --- | --- | --- |
| constant_best | 3 | 5 | 19 | 0 | 100.0% | 18.8% |
| switching_best | 5 | 9 | 40 | 2 | 95.0% | 17.2% |

Nine panels from five sources contained a changing best candidate. The frozen rule retained 38 of 40 exact query-best candidates in this subset. Thus, the aggregate result is not based only on panels with one constant winner, although the only locked failure occurred in the switching-best subset.

## Measurement uncertainty

Reported cell-level standard deviations supported Monte Carlo perturbation for 7 of 14 panels. Other panels either did not report cell uncertainty or did not identify the archived error statistic as SD versus SE. Therefore, exact-best identity and regret are evaluated on reported cell means, and uncertainty-aware conclusions are restricted to the panels with identifiable SD values.

## Retention-savings sensitivity

Only the top-half row below is the frozen primary rule. All other rows were computed after the locked data were available and are exploratory.

| Per-anchor rule | Status | Sources retaining all best | Source-balanced assay-cell reduction | Mean panel coverage | Mean normalized regret |
| --- | --- | --- | --- | --- | --- |
| ever_top_one_third | exploratory | 3/6 | 29.4% | 84.5% | 0.0055 |
| ever_top_half_frozen_primary | primary | 5/6 | 19.8% | 95.2% | 0.0034 |
| ever_top_two_thirds | exploratory | 6/6 | 10.5% | 100.0% | 0.0000 |
| ever_top_three_quarters | exploratory | 6/6 | 6.7% | 100.0% | 0.0000 |
| retain_all | exploratory | 6/6 | 0.0% | 100.0% | 0.0000 |

The exploratory top-two-thirds rule retained all observed-best candidates in the current six sources but reduced candidate-condition assay cells by only 10.5%. It is not a prospectively validated replacement for protocol v1.

## Defensible application

The procedure is usable when an investigator already has a fixed physical panel of at least three biochars, a bounded numerical condition domain with at least three shared strata, and the ability to test every candidate at two boundary conditions. It can serve as an auditable pilot-assay baseline for deciding whether any candidates can be deferred from the remaining condition matrix.

The output must be one of the following:

1. a retained candidate set for continued testing;
2. no reduction when anchor responses do not separate candidates; or
3. an out-of-scope decision when material identity, shared conditions, or response direction is not auditable.

It must not be used to select one universal winner, optimize preparation settings, infer performance for unmeasured materials, replace confirmation experiments, or justify safety-critical elimination.

## Manuscript-level conclusion

The locked analysis supports a bounded application claim: two shared-condition pilot assays sometimes reduce the remaining candidate-by-condition workload, but exact-best retention and assay-cell savings trade off. The observed failure rules out language such as "safe screening" or "reliable elimination." The scientifically defensible contribution is an evidence audit plus a falsifiable decision baseline that reports retained-set coverage, regret, assay-cell reduction, abstention, and source-level uncertainty together.
