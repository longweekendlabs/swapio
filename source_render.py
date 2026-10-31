"""Source-pixel face reenactment, measured refinement and native eye detail.

The source appearance is retained throughout. Driving images supply pose and
expression, never target-eye pixels. Models are downloaded independently; CPU
geometry and compositing do not depend on the neural output's native size.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

import core
from source_models import SOURCE_MODELS
from source_details import measure_teeth, project_teeth

ARC_TEMPLATE = core.SOURCE_FACE_TEMPLATE


def rotation(motion):
    pitch, yaw, roll = [float(np.asarray(motion[i]).ravel()[0]) * np.pi / 180 for i in range(3)]
    cx, cy, cz = np.cos([pitch, yaw, roll])
    sx, sy, sz = np.sin([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return rz @ ry @ rx


def eye_ratios(points):
    return np.array([
        np.linalg.norm(points[40] - points[33]) / max(np.linalg.norm(points[39] - points[35]), .001),
        np.linalg.norm(points[94] - points[87]) / max(np.linalg.norm(points[93] - points[89]), .001),
    ], np.float32)


def lip_ratio(points):
    width, opening = core.SwapEngine._feature_geometry(points[core.INNER_MOUTH_106_INDICES])
    return opening / max(width, .001)


def eye_frame(points):
    # Semantic corners retain the same orientation at every head roll. Sorting
    # by image x flips the donor vertically when a head crosses 90 degrees.
    a, b = points[1], points[4]
    width = float(np.linalg.norm(b - a))
    if width < 1:
        return None
    axis = (b - a) / width
    normal = np.array([-axis[1], axis[0]], np.float32)
    return points.mean(0), axis, normal, width, float(np.ptp(points @ normal))


def reference_eyes(base, reference, source_points, placed_points, strength=1.0):
    """Project real source lids/lashes, retaining iris proportions during blinking."""
    if source_points is None or placed_points is None or strength <= 0:
        return base, 0
    source_points, placed_points = np.asarray(source_points), np.asarray(placed_points)
    if (source_points.shape != (106, 2) or placed_points.shape != (106, 2)
            or not np.isfinite(source_points).all() or not np.isfinite(placed_points).all()):
        return base, 0
    result = base.copy()
    accepted = 0
    for indices, pupil in ((core.LEFT_EYE_106_INDICES, 34), (core.RIGHT_EYE_106_INDICES, 88)):
        a, b = source_points[indices], placed_points[indices]
        if (np.any(a < 0) or np.any(a[:, 0] >= reference.shape[1])
                or np.any(a[:, 1] >= reference.shape[0])):
            continue
        fa, fb = eye_frame(a), eye_frame(b)
        if fa is None or fb is None:
            continue
        ca, ax, ay, wa, ha = fa
        cb, bx, by, wb, hb = fb
        if min(wa, wb) < 4 or ha / wa < .08 or hb / wb < .08:
            continue
        # Use the reenacted source's contour. Its identity and expression have
        # already been established; destination eye shape is never grafted in.
        sx, sy = wb / wa, hb / ha
        transform = np.column_stack([bx * sx, by * sy]) @ np.column_stack([ax, ay]).T
        matrix = np.column_stack([transform, cb - transform @ ca]).astype(np.float32)
        radius = max(round(wb * .18), 1)
        margin = radius + max(round(wb * .12), 2) + 2
        low = np.maximum(np.floor(b.min(0) - margin).astype(int), 0)
        high = np.minimum(np.ceil(b.max(0) + margin).astype(int), base.shape[1::-1])
        if np.any(high - low < 2):
            continue
        size = tuple(high - low)
        local_matrix = matrix.copy()
        local_matrix[:, 2] -= low
        donor = cv2.warpAffine(reference, local_matrix, size, flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(b - low).astype(np.int32)), 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip((distance - .5) / max(wb * .1, 1), 0, 1)
        alpha = (alpha * alpha * (3 - 2 * alpha) * float(np.clip(strength, 0, 1)))[:, :, None]

        # The lid transform can close an eye vertically. Reproject the iris
        # isotropically behind that opening, avoiding a flattened pupil.
        ap, bp = source_points[pupil], placed_points[pupil]
        iris_transform = np.column_stack([bx * sx, by * sx]) @ np.column_stack([ax, ay]).T
        iris_matrix = np.column_stack([iris_transform, bp - iris_transform @ ap]).astype(np.float32)
        iris_matrix[:, 2] -= low
        iris = cv2.warpAffine(reference, iris_matrix, size, flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
        yy, xx = np.mgrid[:reference.shape[0], :reference.shape[1]]
        radial = np.sqrt((xx - ap[0]) ** 2 + (yy - ap[1]) ** 2)
        iris_alpha = np.clip((wa * .21 - radial) / max(wa * .025, 1), 0, 1).astype(np.float32)
        iris_alpha *= core.SwapEngine._feature_mask(reference.shape[:2], a, ha)
        iris_alpha = cv2.warpAffine(iris_alpha, iris_matrix, size)
        iris_alpha = np.minimum(iris_alpha,
            core.SwapEngine._feature_mask((size[1], size[0]), b - low, hb))[:, :, None]
        # The affine lid donor also contains an iris. Remove its entire eye
        # opening before placing the proportional iris, otherwise a vertically
        # stretched pupil remains above/below the smaller isotropic patch.
        # Unseen sclera/iris portions come from the reenacted SOURCE appearance.
        # Destination eye pixels are never used here.
        local = b - low
        centred = local - local.mean(0)
        order = np.argsort(np.arctan2(centred[:, 1], centred[:, 0]))
        opening = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(opening, [np.rint(local[order] * 16).astype(np.int32)],
                     255, lineType=cv2.LINE_AA, shift=4)
        opening = (opening.astype(np.float32) / 255)[:, :, None]
        region = np.s_[low[1]:high[1], low[0]:high[0]]
        donor = opening * base[region] + (1 - opening) * donor
        donor = iris_alpha * iris + (1 - iris_alpha) * donor
        result[region] = np.rint(alpha * donor + (1 - alpha) * result[region]).clip(0, 255).astype(np.uint8)
        accepted += 1
    return result, accepted


def fit_illumination(source, scene, skin):
    """Fit smooth illumination from skin samples, excluding facial features."""
    a = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(scene.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x, y = (xx - a.shape[1] / 2) / (a.shape[1] / 2), (yy - a.shape[0] / 2) / (a.shape[0] / 2)
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1).astype(np.float32)
    points = np.flatnonzero(skin.ravel())
    points = points[::max(len(points) // 12000, 1)]
    if len(points) < 50:
        return None
    design = basis.reshape(-1, 6)[points]
    difference = (b - a).reshape(-1, 3)[points]
    weight = np.ones(len(points), np.float32)
    for _ in range(6):
        w = np.sqrt(weight)[:, None]
        coefficients = np.linalg.lstsq(design * w, difference * w, rcond=None)[0]
        residual = np.linalg.norm(difference - design @ coefficients, axis=1)
        scale = max(float(np.median(residual)) * 1.5, 2)
        weight = 1 / (1 + (residual / scale) ** 2)
    bounds = np.percentile(difference, [1, 99], axis=0).astype(np.float32)
    if not np.isfinite(coefficients).all():
        return None
    return coefficients, bounds


def apply_illumination(image, matrix, field, visibility):
    """Apply a low-frequency field without resizing the high-resolution pixels."""
    if field is None:
        return image
    yy, xx = np.where(visibility > 0)
    if not len(xx):
        return image
    low = np.array([xx.min(), yy.min()])
    high = np.array([xx.max() + 1, yy.max() + 1])
    ys, xs = np.mgrid[low[1]:high[1], low[0]:high[0]].astype(np.float32)
    x = (matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2] - 256) / 256
    y = (matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2] - 256) / 256
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    coefficients, bounds = field
    delta = np.clip(basis @ coefficients, bounds[0] - 5, bounds[1] + 5).astype(np.float32)
    region = np.s_[low[1]:high[1], low[0]:high[0]]
    lab = cv2.cvtColor(image[region].astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab += delta
    lab[:, :, 0] = lab[:, :, 0].clip(0, 100)
    result = image.copy()
    result[region] = np.rint(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255).clip(0, 255).astype(np.uint8)
    return result


class SourceRenderer:
    def __init__(self, owner):
        self.owner = owner
        self.sessions = {}
        self.cpu_only = set()
        self.reference_key = None
        self.reference = None
        self.verified = {}
        self.reference_teeth = None
        self.reference_teeth_checked = False

    def _session(self, name, on_log):
        if name not in self.sessions:
            import onnxruntime as ort
            # The retargeting MLPs are tiny; inference transfers more data than
            # their computation. Appearance/motion/generation stay accelerated.
            cuda = self.owner.use_gpu and name not in self.cpu_only and name not in ("eyes", "lips")
            if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = [("CUDAExecutionProvider", core.HYPERSWAP_CUDA_OPTIONS), "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            filename, size, digest = SOURCE_MODELS[name]
            path = self.owner.models / filename
            stamp = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None
            if stamp is None or self.verified.get(name) != stamp:
                if not path.is_file() or path.stat().st_size != size:
                    raise RuntimeError(f"Missing or incomplete {filename}; run model setup.")
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(chunk)
                if checksum.hexdigest() != digest:
                    raise RuntimeError(f"Invalid checksum for {filename}; run model setup.")
                self.verified[name] = stamp
            try:
                self.sessions[name] = ort.InferenceSession(str(path), providers=providers)
            except Exception as exc:
                if not cuda or providers == ["CPUExecutionProvider"]:
                    raise
                on_log(f"Source {name} could not start on CUDA ({type(exc).__name__}); retrying on CPU.")
                self.cpu_only.add(name)
                self.sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = self.sessions[name].get_providers()[0]
            self.owner._refresh_provider()
            on_log(f"Source {name} model ready on {'CUDA' if actual == 'CUDAExecutionProvider' else 'CPU'}.")
        return self.sessions[name]

    def _run(self, name, feed, on_log):
        session = self._session(name, on_log)
        try:
            outputs = session.run(None, feed)
            expected = {
                "appearance": [(1, 32, 16, 64, 64)],
                "motion": [(), (), (), (1, 1), (1, 3), (1, 21, 3), (1, 21, 3)],
                "generator": [(1, 3, 512, 512)],
                "eyes": [(1, 21, 3)], "lips": [(1, 21, 3)],
            }[name]
            if len(outputs) != len(expected) or any(a.shape != shape for a, shape in zip(outputs, expected)):
                raise RuntimeError(f"Source {name} model returned an invalid shape.")
            if any(not np.isfinite(a).all() for a in outputs):
                raise RuntimeError(f"Source {name} model returned invalid values.")
            if name == "generator" and (outputs[0].min() < -.1 or outputs[0].max() > 1.1
                                           or outputs[0].std() < 1e-5):
                raise RuntimeError("Source generator returned invalid pixels.")
            return outputs
        except Exception as exc:
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise
            on_log(f"Source {name} failed on CUDA ({type(exc).__name__}); retrying safely on CPU.")
            self.cpu_only.add(name)
            del self.sessions[name]
            self.owner._refresh_provider()
            # One retry, now with the permanently retired GPU session excluded.
            return self._run(name, feed, on_log)

    @staticmethod
    def _prepare(image, face):
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(face.kps, np.float32), ARC_TEMPLATE * 512,
            method=cv2.RANSAC, ransacReprojThreshold=100)
        if matrix is None or not np.isfinite(matrix).all():
            raise RuntimeError("Could not align the source-guided face.")
        fitted = cv2.transform(np.asarray(face.kps, np.float32)[None], matrix)[0]
        if np.max(np.linalg.norm(fitted - ARC_TEMPLATE * 512, axis=1)) > 512 * .12:
            raise RuntimeError("Face alignment is unreliable; result was not saved.")
        crop = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
        blob = (cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                / 255).transpose(2, 0, 1)[None].astype(np.float32)
        return blob, matrix

    def _reference(self, image, face, on_log):
        blob, _ = self._prepare(image, face)
        checksum = hashlib.blake2b(digest_size=16)
        checksum.update(np.ascontiguousarray(image).tobytes())
        checksum.update(np.asarray(image.shape, np.int64).tobytes())
        checksum.update(np.asarray(face.kps, np.float32).tobytes())
        key = checksum.digest()
        if key != self.reference_key:
            feature = self._run("appearance", {"input": blob}, on_log)[0]
            motion = self._run("motion", {"input": blob}, on_log)
            original = motion[3] * (motion[6] @ rotation(motion).T + motion[5]) + motion[4]
            frame = self.owner._eye_landmark_frame(image, face)
            measured = self.owner._eye_landmarks(image, face, frame)
            points = getattr(measured, "landmark_2d_106", None)
            self.reference = (feature, motion, original, points)
            self.reference_key = key
            self.reference_teeth = None
            self.reference_teeth_checked = False
        return self.reference

    def render(self, scene, current, face, source_image, source_face,
               quality, detail_strength, visibility, on_log, reference_mouth=False,
               mouth_unobstructed=False):
        feature, source_motion, original_motion, reference_points = self._reference(source_image, source_face, on_log)
        blob, matrix = self._prepare(scene, face)
        driving = self._run("motion", {"input": blob}, on_log)
        expression = source_motion[5].copy()
        ids = [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        expression[:, ids] = driving[5][:, ids]
        expression[:, 3:5, 1] = driving[5][:, 3:5, 1]
        expression[:, 5, 2] = driving[5][:, 5, 2]
        expression[:, 8, 2] = driving[5][:, 8, 2]
        expression[:, 9, 1:] = driving[5][:, 9, 1:]
        translation = driving[4].copy()
        translation[..., 2] = 0
        desired = source_motion[3] * (source_motion[6] @ rotation(driving).T + expression) + translation
        frame = self.owner._eye_landmark_frame(scene, face)

        def generate(points):
            prediction = self._run("generator", {
                "feature_volume": feature, "source": points.astype(np.float32),
                "target": original_motion.astype(np.float32),
            }, on_log)[0][0]
            crop = (prediction.transpose(1, 2, 0)[:, :, ::-1] * 255).round().clip(0, 255).astype(np.uint8)
            return self.owner._paste_crop(current, crop, matrix)

        rendered = generate(desired)
        measured = self.owner._eye_landmarks(rendered, face, frame)
        target_measured = self.owner._eye_landmarks(scene, face, frame) if frame is not None else face
        target_points = getattr(target_measured, "landmark_2d_106", None)
        target_eyes = eye_ratios(target_points) if target_points is not None else None
        target_lip = lip_ratio(target_points) if target_points is not None else None
        # The MLP accepts one opening target for both eyes. Asymmetric blinks
        # retain the independently driven expression instead of opening a wink.
        symmetric = target_eyes is not None and abs(float(target_eyes[0] - target_eyes[1])) < .10
        eye_goal = float(np.clip(target_eyes.mean(), 0, .6)) if symmetric else None
        # The parser checks actual lip tissue and tongue protrusion. A purely
        # geometric mouth ring also contains visible skin around an obstruction.
        oral_visible = 1.0 if mouth_unobstructed else 0.0
        maximum = 2 if quality == "best" else 1
        passes = 1
        for _ in range(maximum):
            points = getattr(measured, "landmark_2d_106", None)
            if points is None or target_points is None:
                break
            eyes, lips = eye_ratios(points), lip_ratio(points)
            eye_error = abs(float(eyes.mean()) - eye_goal) if eye_goal is not None else 0
            lip_error = abs(lips - target_lip)
            if eye_error < .025 and lip_error < .025:
                break
            delta = np.zeros_like(desired)
            if eye_error >= .025:
                feed = np.concatenate([desired.ravel(), eyes, [eye_goal]]).reshape(1, -1).astype(np.float32)
                delta += self._run("eyes", {"input": feed}, on_log)[0].reshape(1, 21, 3)
            # Occluded mouths are preserved by the outer compositor; unreliable
            # mouth measurements must not distort the reconstructed source.
            if lip_error >= .025 and oral_visible >= .85:
                feed = np.concatenate([desired.ravel(), [lips, target_lip]]).reshape(1, -1).astype(np.float32)
                delta += self._run("lips", {"input": feed}, on_log)[0].reshape(1, 21, 3)
            if not np.any(delta):
                break
            candidate_points = desired + delta
            candidate = generate(candidate_points)
            placed = self.owner._eye_landmarks(candidate, face, frame)
            candidate_landmarks = getattr(placed, "landmark_2d_106", None)
            if candidate_landmarks is None:
                break
            new_eye_error = abs(float(eye_ratios(candidate_landmarks).mean()) - eye_goal) if eye_goal is not None else 0
            new_lip_error = abs(lip_ratio(candidate_landmarks) - target_lip) if oral_visible >= .85 else 0
            old_lip_error = lip_error if oral_visible >= .85 else 0
            if (new_eye_error > eye_error + .003 or new_lip_error > old_lip_error + .005
                    or new_eye_error + new_lip_error >= eye_error + old_lip_error):
                break
            desired, rendered, measured = candidate_points, candidate, placed
            passes += 1
        on_log(f"Source-guided face: {passes} generation pass(es), retaining reference appearance.")
        rendered, count = reference_eyes(rendered, source_image, reference_points,
            getattr(measured, "landmark_2d_106", None), detail_strength)
        on_log(f"Projected actual source eye detail into {count} eye region(s); target-eye copying is separate.")
        source_crop, source_labels, light_matrix = self.owner._parse_face(rendered, face)
        if reference_mouth:
            count = 0
            pose_delta = max(abs(float(source_motion[i]) - float(driving[i])) for i in (0, 1))
            if oral_visible >= .85 and pose_delta <= 25:
                placed_teeth = measure_teeth(rendered, getattr(measured, "landmark_2d_106", None),
                                             source_labels, light_matrix)
                if placed_teeth is not None:
                    if not self.reference_teeth_checked:
                        _, labels, source_matrix = self.owner._parse_face(source_image, source_face)
                        self.reference_teeth = measure_teeth(source_image, reference_points, labels, source_matrix)
                        self.reference_teeth_checked = True
                    rendered, count = project_teeth(rendered, source_image, self.reference_teeth, placed_teeth)
            on_log("Projected actual reference upper-tooth detail." if count else
                   "Reference tooth detail unavailable or unsafe; retained the generated source mouth.")
        scene_crop, scene_labels, _ = self.owner._parse_face(scene, face, light_matrix)
        field = fit_illumination(source_crop, scene_crop, (source_labels == 1) & (scene_labels == 1))
        rendered = apply_illumination(rendered, light_matrix, field, visibility)
        return rendered
