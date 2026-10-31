"""Conservative native-reference dental detail for an unobstructed source mouth."""
from dataclasses import dataclass

import cv2
import numpy as np

from core import INNER_MOUTH_106_INDICES


@dataclass
class ToothRow:
    origin: np.ndarray
    anchor: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    width: float
    hull: np.ndarray
    cavity: np.ndarray


def measure_teeth(image, points, labels, matrix):
    """Find one substantial upper-enamel row, excluding lips and small glints."""
    if points is None or np.asarray(points).shape != (106, 2) or not np.isfinite(points).all():
        return None
    contour = np.asarray(points, np.float32)[INNER_MOUTH_106_INDICES]
    axis = contour[4] - contour[0]
    width = float(np.linalg.norm(axis))
    if width < 4:
        return None
    axis /= width
    normal = np.array([-axis[1], axis[0]], np.float32)
    height = float(np.ptp(contour @ normal))
    if height < width * .08:
        return None
    low = np.maximum(np.floor(contour.min(0) - width * .05 - 2).astype(int), 0)
    high = np.minimum(np.ceil(contour.max(0) + width * .05 + 2).astype(int), image.shape[1::-1])
    if np.any(high - low < 2):
        return None
    size = tuple(high - low)
    inverse = cv2.invertAffineTransform(matrix)
    inverse[:, 2] -= low
    cavity = cv2.warpAffine((labels == 11).astype(np.float32), inverse, size)
    opening = np.zeros(cavity.shape, np.uint8)
    cv2.fillConvexPoly(opening, cv2.convexHull(np.rint(contour - low).astype(np.int32)), 1)
    cavity *= opening
    region = image[low[1]:high[1], low[0]:high[0]].astype(np.float32) / 255
    lightness = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)[:, :, 0]
    chroma = region.max(2) - region.min(2)
    centre = contour.mean(0)
    enamel = ((cavity > .9) & (lightness > 65) & (chroma < .32)).astype(np.uint8)
    # Join neighbouring teeth along the row, without joining upper/lower rows.
    radius = max(round(width * .0125), 0)
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), np.uint8)
    cv2.line(kernel, tuple(np.rint(radius - axis * radius).astype(int)),
              tuple(np.rint(radius + axis * radius).astype(int)), 1)
    enamel = cv2.morphologyEx(enamel, cv2.MORPH_CLOSE, kernel)
    count, components, stats, centres = cv2.connectedComponentsWithStats(enamel)
    candidates = [i for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= max(width * width * .006, 8)
                  and np.dot(centres[i] + low - centre, normal) < height * .25]
    if not candidates:
        return None
    selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    coords = np.column_stack(np.where(components == selected)[::-1]).astype(np.float32)
    row_width = float(np.ptp(coords @ axis))
    row_height = float(np.ptp(coords @ normal))
    if row_width < width * .28 or not width * .025 <= row_height <= width * .35:
        return None
    hull = np.zeros(cavity.shape, np.float32)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(coords).astype(np.int32)), 1)
    anchor = coords.mean(0)
    anchor += normal * (np.percentile(coords @ normal, 5) - anchor @ normal)
    return ToothRow(low, anchor + low, axis, normal, row_width, hull, cavity)


def project_teeth(image, reference, source, placed):
    """Place authentic enamel without stretching teeth or leaving a second row."""
    if source is None or placed is None:
        return image, 0
    scale = placed.width / source.width
    transform = (np.column_stack([placed.axis, placed.normal]) * scale
                 @ np.column_stack([source.axis, source.normal]).T)
    translation = placed.anchor - transform @ source.anchor
    matrix = np.column_stack([transform, translation - placed.origin]).astype(np.float32)
    size = placed.cavity.shape[::-1]
    donor = cv2.warpAffine(reference, matrix, size, flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REPLICATE)
    local_matrix = matrix.copy()
    local_matrix[:, 2] += transform @ source.origin
    hull = cv2.warpAffine(source.hull, local_matrix, size)
    cavity = cv2.warpAffine(source.cavity, local_matrix, size)
    # Erase the neural tooth footprint too. Intersecting enamel-only masks
    # leaves its lower edge visible as a second, false tooth row.
    feather = max(placed.width * .035, 1)
    radius = int(np.ceil(feather)) + 1
    support = cv2.dilate((np.maximum(hull, placed.hull) > .1).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    mask = support * np.minimum(cavity, placed.cavity)
    coverage = float(((mask > .5) & (placed.hull > .5)).sum() / max(placed.hull.sum(), 1))
    if coverage < .85:
        return image, 0
    distance = cv2.distanceTransform((mask > .5).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = (np.clip(distance / feather, 0, 1) * mask)[:, :, None]
    if np.count_nonzero(alpha > .1) < 8:
        return image, 0
    low = placed.origin
    region = np.s_[low[1]:low[1] + size[1], low[0]:low[0] + size[0]]
    result = image.copy()
    result[region] = np.rint(alpha * donor + (1 - alpha) * image[region]).clip(0, 255).astype(np.uint8)
    return result, 1
