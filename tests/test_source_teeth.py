import unittest

import cv2
import numpy as np

import core
from source_details import measure_teeth, project_teeth


class ReferenceTeethTests(unittest.TestCase):
    def fixture(self):
        points = np.zeros((106, 2), np.float32)
        points[core.INNER_MOUTH_106_INDICES] = [[50, 120], [70, 106], [110, 102], [150, 106],
                                               [170, 120], [150, 146], [110, 150], [70, 146]]
        labels = np.zeros((200, 240), np.uint8)
        cv2.fillConvexPoly(labels, cv2.convexHull(points[core.INNER_MOUTH_106_INDICES].astype(np.int32)), 11)
        source = np.full((200, 240, 3), [100, 120, 150], np.uint8)
        source[labels == 11] = [20, 25, 30]
        source[112:131, 75:146] = [220, 230, 230]
        source[112:131, 105:108] = [20, 25, 30]  # authentic dental gap
        generated = source.copy()
        generated[112:144, 75:146] = [220, 230, 230]  # blurred/overlong enamel footprint
        return source, generated, points, labels

    def test_reference_pattern_replaces_the_neural_row_without_leaving_a_second_edge(self):
        source, generated, points, labels = self.fixture()
        matrix = np.eye(2, 3, dtype=np.float32)
        a = measure_teeth(source, points, labels, matrix)
        b = measure_teeth(generated, points, labels, matrix)
        self.assertIsNotNone(a); self.assertIsNotNone(b)
        result, count = project_teeth(generated, source, a, b)
        self.assertEqual(count, 1)
        self.assertLess(int(result[120, 106].max()), 50)
        self.assertLess(int(result[140, 110].max()), 50)
        np.testing.assert_array_equal(result[labels != 11], generated[labels != 11])

    def test_semantic_row_detection_survives_quarter_turns(self):
        source, generated, points, labels = self.fixture()
        matrix = np.eye(2, 3, dtype=np.float32)
        a = measure_teeth(source, points, labels, matrix)
        transform = np.array([[0, -1, 199], [1, 0, 0]], np.float32)
        rotated = cv2.rotate(generated, cv2.ROTATE_90_CLOCKWISE)
        rotated_labels = cv2.rotate(labels, cv2.ROTATE_90_CLOCKWISE)
        rotated_points = cv2.transform(points[None], transform)[0]
        b = measure_teeth(rotated, rotated_points, rotated_labels, matrix)
        self.assertIsNotNone(b)
        result, count = project_teeth(rotated, source, a, b)
        self.assertEqual(count, 1)
        self.assertLess(int(result[106, 199 - 120].max()), 50)

    def test_dark_cavities_closed_mouths_and_lip_glints_are_not_teeth(self):
        source, _, points, labels = self.fixture()
        source[labels == 11] = 40
        self.assertIsNone(measure_teeth(source, points, labels, np.eye(2, 3, dtype=np.float32)))
        source[100:103, 60:160] = 250  # outside the actual mouth cavity
        self.assertIsNone(measure_teeth(source, points, labels, np.eye(2, 3, dtype=np.float32)))
        flat = points.copy(); flat[core.INNER_MOUTH_106_INDICES, 1] = 120
        self.assertIsNone(measure_teeth(source, flat, labels, np.eye(2, 3, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
