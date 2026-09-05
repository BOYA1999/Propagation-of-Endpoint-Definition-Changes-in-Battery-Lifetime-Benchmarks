# Figures

The current five figures are built by `06_figures/reviewer_v14_sync/build_jeet_figures_v14.py` from the supplied derived data or freshly recomputed controlled-analysis tables.

1. Source endpoint, archived labels and generator outputs on their respective coordinates.
2. Paired endpoint-label geometry.
3. Cross-contract prediction movement relative to ordinary refitting; the ratio reference is one.
4. Winner stability and full-rank agreement.
5. Conformal coverage, width and calibration-source sensitivity. The accompanying efficiency table also records interval scores.

The builder compares eight generated source tables with the frozen tables and saves SVG, PDF, EPS and 600-dpi PNG files. The layout preserves a minimum placed font size of 8 pt at a width of 165.1 mm. Source-table checks establish numerical correspondence; they do not replace visual inspection.
