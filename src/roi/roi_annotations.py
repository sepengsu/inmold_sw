"""Manual ROI line and quadrilateral parsing in original image coordinates."""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path


def line_annotation(spec: dict) -> dict:
    shape = spec.get('shape')
    if (not isinstance(shape, list) or len(shape) != 2 or
            any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in shape)):
        raise ValueError('annotation shape must be integer [height, width]')
    h, w = shape
    if 'corners' in spec:
        if set(spec) != {'shape', 'corners'}:
            raise ValueError('corner annotation requires shape and corners only')
        corners = spec['corners']
        if isinstance(corners, dict):
            if set(corners) != {'TL', 'TR', 'BR', 'BL'}:
                raise ValueError('corners require TL, TR, BR, BL')
            corners = [corners[k] for k in ('TL', 'TR', 'BR', 'BL')]
        points = np.asarray(corners, dtype=float)
        if points.shape != (4, 2):
            raise ValueError('corners must be four [x, y] points in TL, TR, BR, BL order')
        sides = {'top': points[[0, 1]], 'bottom': points[[3, 2]]}
    else:
        if set(spec) != {'shape', 'top_points', 'bottom_points'}:
            raise ValueError('line annotation requires shape, top_points, bottom_points')
        sides = {s: np.asarray(spec[s + '_points'], dtype=float) for s in ('top', 'bottom')}
    result = {'shape': shape}
    for side, points in sides.items():
        if (points.shape != (2, 2) or not np.isfinite(points).all()
                or np.any(points[:, 0] < 0) or np.any(points[:, 0] > w - 1)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] > h - 1)
                or points[1, 0] <= points[0, 0]):
            raise ValueError(f'invalid {side} two-point annotation')
        a = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        b = points[0, 1] - a * points[0, 0]
        values = [float(b), float(a * (w - 1) + b)]
        if min(values) < 0 or max(values) > h - 1:
            raise ValueError(f'{side} line extrapolates outside image')
        result[side + '_y'] = values
    if np.any(np.array(result['top_y']) >= result['bottom_y']):
        raise ValueError('manual top and bottom cross')
    return result


def load_json_ground_truth(path: Path, source: Path, reference_exposure: str | None) -> dict:
    with path.open(encoding='utf-8-sig') as handle:
        entries = json.load(handle)
    if not isinstance(entries, dict):
        raise ValueError('GT JSON must be keyed by reference-relative paths')
    result, seen, splits = {}, set(), {}
    for name, entry in entries.items():
        relative = Path(name)
        if (not name or relative.is_absolute() or relative.drive or '..' in relative.parts
                or not (source / relative).is_file()):
            raise ValueError(f'invalid GT reference: {name}')
        if reference_exposure and relative.parent.name.casefold() != reference_exposure.casefold():
            raise ValueError(f'GT must name the reference exposure: {name}')
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f'duplicate GT reference: {name}')
        seen.add(key)
        if not isinstance(entry, dict) or entry.get('split') not in {'tune', 'validation'} or entry.get('status') not in {'approved', 'pending'}:
            raise ValueError(f'GT requires split tune/validation and status approved/pending: {name}')
        group = relative.parts[0].split('-', 1)[0].casefold()
        if group in splits and splits[group] != entry['split']:
            raise ValueError(f'GT case appears in both splits: {group}')
        splits[group] = entry['split']
        if entry['status'] == 'pending':
            continue
        spec = line_annotation({k: v for k, v in entry.items() if k not in {'split', 'status'}})
        result[key] = {'shape': spec['shape'], 'split': entry['split'],
                       **{s: [[0., spec[s + '_y'][0]], [spec['shape'][1] - 1., spec[s + '_y'][1]]]
                          for s in ('top', 'bottom')}}
    return result
