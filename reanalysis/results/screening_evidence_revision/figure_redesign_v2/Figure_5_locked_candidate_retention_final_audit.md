# Figure 5 audit

## Automated checks

| Check | Status | Detail |
|---|---|---|
| Panel B source summaries | PASS | Query counts, failures, coverage, and assay reduction reproduce the source table. |
| Locked evidence totals | PASS | 6 sources, 14 panels, 59 query strata; 57/59 observed query bests retained. |
| Panel D failure cases | PASS | The only misses occur in the Ogbuagu wheat-straw panel at the two plotted intermediate concentrations. |
| Panel E frozen strategy | PASS | The square is the predeclared top-half rule; all other points are post-freeze sensitivity analyses. |
| Panel F uncertainty subset | PASS | 7 panels with identifiable reported cell SD; 20,000 draws each. |
| Export completeness | PASS | SVG, PDF, PNG, and TIFF are present. |
| Editable SVG | PASS | 101 editable text nodes; no embedded raster image nodes. |
| Minimum text size | PASS | Minimum SVG font size: 7.8 pt-equivalent px. |
| Forbidden effects | PASS | No gradient or SVG filter/shadow definitions. |
| PNG final dimensions | PASS | 2161 x 2362 px at 183 x 200 mm and 300 dpi; mode RGB. |
| PDF final dimensions | PASS | 183.00 x 200.00 mm. |

## Manual visual review

- Panel letters, titles, axes, legends, and direct labels remain inside the 183 x 200 mm canvas.
- No text overlap or clipped legend was visible in the RGB PNG rendered at final dimensions.
- Panel D is intentionally dense because it displays all ten candidate rank trajectories in the only locked failure panel.
- Navy, teal, amber, rust, and gray encodings remain distinguishable by luminance and marker/line form in grayscale.
- No grid, gradient, shadow, 3D object, decorative icon, or raster icon is used.

## Files requiring manual review

- None after the final RGB render. Panel D should still be checked once after journal typesetting reduction.
