# ROI source

Source repository: [sepengsu/in-mold-surface-impregnation-inspection](https://github.com/sepengsu/in-mold-surface-impregnation-inspection).

Pinned commit: `dcaafcd5a03903699161b2ded5d2aa1add1c8b8c`.

The following files were initially copied from `src/roi/`:

- `auto_roi_local_std.py`
- `auto_roi_group.py`
- `roi_boundary.py`
- `roi_boundary_quality.py`
- `roi_annotations.py`
- `__init__.py`

`config/roi_gabor_boundary.yaml` started from that commit. Its distances remain
native source pixels. Gabor parameters have since been calibrated on a local
2046×2046 camera sample, and linked low/medium/high validation presets were added.

Local adaptation: `fit_boundary_line` accepts a configurable `min_third_support`
(default 1 for existing upstream callers). `roi_boundary.fit_line` supplies the
active quality setting so RANSAC and final review use the same support requirement.
The relaxed preset permits missing thirds while retaining minimum point count,
overall span, gap, residual, and geometry checks. No absent boundary is fabricated.

The system adapter is `src/preprocess/auto_roi.py`. It runs the upstream
`features → detect_candidates → finalize → roi_pixel_mask` path on one current
camera frame. Single-frame analysis has no multi-exposure consensus bonus.
`NEEDS_REVIEW` and full-frame preservation are displayed as review results and
cannot produce an automatic inspection PASS.

Normal and certification inspection treat quality review findings, including `AMBIGUOUS_CANDIDATES`
and `FABRIC_OUTSIDE_TOP` / `FABRIC_OUTSIDE_BOTTOM`, as warnings when the captured
score meets its threshold and an ROI was detected. `INVALID_GEOMETRY` and
`LOW_BOUNDARY_SCORE` still block inference. It preserves `NEEDS_REVIEW`, records `inspection_allowed` and
`warning_reasons`, and still checks native geometry. Normal inspection checks complete
measurement-region containment before predicting six tiles. Certification applies
the same score gate, then detects and measures its three windows in the complete
original frame using the existing PCE geometry and scoring rules. Default
preprocessing and Cam_test keep the strict detector decision.

Camera images retain their native coordinates. The general preprocessing API can
still tile the mask's bounding box. The inspection GUI instead places three
256x512 native measurement regions on the per-column fabric midline, with 547px
horizontal center spacing. Every region must be fully inside the fabric mask.
It predicts only the six nonoverlapping 256x256 tiles in those regions and uses
131,072 pixels per point for impregnation percentages. Display overlays use the
same original-image coordinates; the summary is the mean of the three points.
ROI detection and CLAHE are unchanged, with no resizing or perspective transforms.
