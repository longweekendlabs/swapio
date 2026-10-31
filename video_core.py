"""Streaming video transport and conservative person tracking around Careful.

No SimSwap code or weights are used. Frames stay on disk only inside the encoded
temporary movie; the existing image renderer receives one decoded frame at a time.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

import core
from media_tools import native_process_environment

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
PREVIEW_SECONDS = 2.


def tools() -> tuple[str, str]:
    paths = tuple(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if not all(paths):
        raise RuntimeError("Video needs FFmpeg and ffprobe on this computer. Install your system's FFmpeg package, then restart Swapio. Nothing has been downloaded.")
    return paths


def run_tool(command, timeout=30, should_stop=None):
    if should_stop is None:
        result = subprocess.run(command, capture_output=True, env=native_process_environment(), timeout=timeout)
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=native_process_environment())
        deadline = time.monotonic()+timeout
        try:
            while True:
                if should_stop():
                    raise core.ProcessingCancelled()
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command,timeout)
                try:
                    stdout,stderr = process.communicate(timeout=.2)
                    result = subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            _terminate(process)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2400:] or "Media tool failed.")
    return result.stdout


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    rate: Fraction
    duration: float
    audio: bool
    sar: Fraction = Fraction(1)

    @property
    def fps(self):
        return float(self.rate)


def probe(path: Path, count_frames=False, should_stop=None) -> tuple[VideoInfo, dict]:
    _, ffprobe = tools()
    command = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        command += ["-count_frames"]
    data = json.loads(run_tool([*command, str(path)], timeout=120 if count_frames else 30, should_stop=should_stop))
    streams = data.get("streams", [])
    stream = next((s for s in streams if s.get("codec_type") == "video" and
                   not s.get("disposition", {}).get("attached_pic")), None)
    if stream is None:
        raise RuntimeError("No video stream was found.")
    try:
        rate = Fraction(stream.get("avg_frame_rate", "0/1"))
        if rate <= 0:
            rate = Fraction(stream["r_frame_rate"])
        duration = float(stream.get("duration") or data["format"]["duration"])
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        raise RuntimeError("Could not read the video's dimensions, duration or frame rate.") from exc
    if not (0 < rate <= 240 and math.isfinite(duration) and duration > 0 and width > 0 and height > 0):
        raise RuntimeError("The video has unsupported dimensions, frame rate or duration.")
    rotation = float(next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s),
                          stream.get("tags", {}).get("rotate", 0)))
    if not math.isfinite(rotation) or abs(rotation) % 90:
        raise RuntimeError("This video's arbitrary rotation needs normalization before swapping. Export an upright copy first.")
    try:
        sar = Fraction(stream.get("sample_aspect_ratio","1:1").replace(":","/"))
        if sar <= 0: sar = Fraction(1)
    except (ValueError,ZeroDivisionError):
        sar = Fraction(1)
    if abs(rotation) % 180 == 90:
        width, height = height, width
        sar = 1/sar
    if stream.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise RuntimeError("HDR video needs tone mapping before this first video version. Export an SDR copy to use here.")
    return VideoInfo(Path(path), width, height, rate, duration,
                     any(s.get("codec_type") == "audio" for s in streams),sar), stream


def clip_range(info, start=0., end=0.):
    end = min(float(end), info.duration) if end else info.duration
    start = float(start)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or start >= info.duration:
        raise ValueError("Choose a start before the end of the video. End 0 means the end of the clip.")
    return start, end


def _decode_command(info, start, duration, rate=None):
    ffmpeg, _ = tools()
    return [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-ss", str(start), "-i", str(info.path),
            "-t", str(duration), "-map", "0:v:0", "-an", "-sn", "-vf", f"fps={rate or info.rate}",
            "-threads", "2", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


def _read_frame(pipe, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = pipe.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    if chunks and len(chunks) != size:
        raise RuntimeError("Video decoding ended in the middle of a frame.")
    return bytes(chunks)


def _terminate(process):
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass


def _embedding(face):
    value = np.asarray(face.normed_embedding, np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    if value.size < 2 or not np.isfinite(value).all() or norm < 1e-8:
        raise RuntimeError("The selected person's face could not be measured. Choose a clearer starting point.")
    return value / norm


@dataclass
class PersonSelection:
    path: str
    signature: tuple[int, int]
    start: float
    embedding: np.ndarray
    image: np.ndarray
    box: np.ndarray
    timestamp: float

    def valid(self, path, start):
        stat = Path(path).stat()
        return str(Path(path).resolve()) == self.path and (stat.st_size, stat.st_mtime_ns) == self.signature and start == self.start


def choose_person(engine, path, start=0., end=0., selector=None, should_stop=lambda: False):
    info, _ = probe(path)
    start, end = clip_range(info, start, end)
    engine.ensure_loaded()
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(_decode_command(info, start, min(end-start, 10.), "2"),
                                       stdout=subprocess.PIPE, stderr=errors, env=native_process_environment())
            for index in range(20):
                if should_stop():
                    raise core.ProcessingCancelled()
                raw = _read_frame(process.stdout, info.width*info.height*3)
                if not raw:
                    break
                image = np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3).copy()
                faces = sorted(engine.analyser.get(image), key=core._face_area, reverse=True)
                if not faces:
                    continue
                choice = selector(path, image, faces) if selector else (0 if len(faces) == 1 else None)
                if choice is None:
                    raise core.ProcessingCancelled()
                if not isinstance(choice, int) or not 0 <= choice < len(faces):
                    raise ValueError("The selected person is unavailable.")
                face = faces[choice]
                stat = Path(path).stat()
                return PersonSelection(str(Path(path).resolve()), (stat.st_size, stat.st_mtime_ns), start,
                                       _embedding(face), image, np.asarray(face.bbox).copy(), start+index/2)
            if process.wait(timeout=10):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace")[-2000:])
        finally:
            _terminate(process)
    raise RuntimeError("No selectable face in the first 10 seconds of this range. Move Start to a clearer face and try again.")


def _iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(hi-lo, 0)))
    area = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection / max(area, 1.)


class PersonTracker:
    """Match the chosen original person, with no largest-face fallback."""
    def __init__(self, selection):
        self.reference = selection.embedding
        self.last_box = None
        self.last_kps = None
        self.missed = 0
        self.signature = None
        self.cut = False

    def select(self, image, faces):
        thumbnail = cv2.resize(image, (32,32)).astype(np.float32)
        self.cut = self.signature is not None and float(np.mean(np.abs(thumbnail-self.signature))) > 42
        self.signature = thumbnail
        if self.cut:
            self.last_box = self.last_kps = None
        candidates = []
        for face in faces:
            try:
                score = float(np.dot(self.reference, _embedding(face)))
            except (ValueError, RuntimeError):
                continue
            overlap = _iou(self.last_box, face.bbox) if self.last_box is not None else 0.
            candidates.append((score, overlap, face))
        candidates.sort(key=lambda row: row[0], reverse=True)
        threshold = .43 if self.last_box is not None and self.missed < 3 else .58
        if (not candidates or candidates[0][0] < threshold or
                (len(candidates)>1 and candidates[0][0]-candidates[1][0] < .08)):
            self.missed += 1
            self.last_kps = None
            return None
        score, overlap, detected = candidates[0]
        # Identity plus continuity, rather than following a stranger at the old position.
        if self.last_box is not None and overlap < .05 and score < .65:
            self.missed += 1
            self.last_kps = None
            return None
        face = type(detected)(deepcopy(dict(detected))) if isinstance(detected, dict) else deepcopy(detected)
        kps = np.asarray(face.kps, np.float32)
        if self.last_kps is not None and overlap > .5:
            delta = kps-self.last_kps
            span = max(float(np.max(face.bbox[2:]-face.bbox[:2])), 1.)
            # Damp tiny detector jitter only. Fast motion, blinks and mouth contours
            # remain current-frame measurements; no old face pixels are averaged in.
            if float(np.max(np.linalg.norm(delta, axis=1))) < span*.035:
                shift = -.15*delta.mean(axis=0)
                shift /= max(float(np.linalg.norm(shift)),1.)
                # One common offset, capped at one pixel, preserves current
                # inter-eye and mouth geometry rather than smoothing each lid.
                face.kps = kps+shift
        self.last_kps = np.asarray(face.kps).copy()
        self.last_box = np.asarray(face.bbox).copy()
        self.missed = 0
        return face


class VideoSwapEngine(core.SwapEngine):
    """Careful with a model/sampling choice held through each continuous scene."""
    def begin_segment(self):
        self.video_choice = None
        self.video_boost = None

    def _identity_checked_face(self, target, target_face, source_face, boost_size,
                               quality, restoration_strength, on_log, finish=None):
        if getattr(self, "video_choice", None) is None:
            self.video_boost = boost_size
            result, self.video_choice = super()._identity_checked_face(
                target, target_face, source_face, boost_size, "careful", restoration_strength, on_log, finish)
            on_log("Video: keeping this face model and sampling size until a scene change.")
            return result, self.video_choice
        if self.video_choice:
            result = self._inswapper_face(target, target_face, source_face,
                                         256 if self.video_boost <= 512 else 512, on_log)
            if restoration_strength > 0:
                result = self._restore_face(result, target_face,
                    min(restoration_strength, core.IDENTITY_RESTORATION_LIMIT), on_log, target)
        else:
            result = self._hyperswap_face(target, target_face, source_face, self.video_boost, on_log)
        return (finish(result, self.video_choice) if finish else result), self.video_choice


def render_video(engine, source_path, path, output_dir, selection, *, start=0., end=0.,
                 preview=False, character_name="", restoration_strength=core.DEFAULT_RESTORATION_STRENGTH,
                 destination_eyes=False, preserve_mouth=True, should_stop=lambda: False,
                 on_log=lambda _: None, on_progress=lambda *_: None):
    info, _ = probe(path)
    start, end = clip_range(info, start, end)
    if preview:
        end = min(end, start+PREVIEW_SECONDS)
    if not selection.valid(path, start):
        raise RuntimeError("This video or its start changed. Choose its person again.")
    ffmpeg, _ = tools()
    if "libx264" not in run_tool([ffmpeg,"-v","error","-encoders"]).decode(errors="replace"):
        raise RuntimeError("This FFmpeg installation has no H.264 encoder (libx264). Install a full FFmpeg build to render video.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = core.unique_output_path(output_dir, Path(path), suffix=".mp4", name_prefix=character_name)
    if output.resolve() == Path(path).resolve():
        raise ValueError("Output must differ from the source video.")
    engine.begin_segment()
    source_face = engine.source_face(source_path, on_log)
    tracker = PersonTracker(selection)
    duration = end-start
    expected = max(1, round(duration*info.fps))
    frame_size = info.width*info.height*3
    decoded = swapped = unchanged = 0
    decoder = encoder = None
    first_image = None
    started = time.monotonic()
    messages = set()
    def log_once(message):
        if message not in messages:
            messages.add(message)
            on_log(message)
    with tempfile.TemporaryDirectory(prefix=".swapio-video-", dir=output_dir) as folder:
        temp = Path(folder)
        with tempfile.TemporaryFile() as decode_errors, tempfile.TemporaryFile() as encode_errors:
            try:
                decoder = subprocess.Popen(_decode_command(info,start,duration),stdout=subprocess.PIPE,
                    stderr=decode_errors,env=native_process_environment())
                pix_fmt = "yuv444p" if info.width%2 or info.height%2 else "yuv420p"
                encoder = subprocess.Popen([ffmpeg,"-v","error","-nostdin","-y","-f","rawvideo",
                    "-pix_fmt","bgr24","-s",f"{info.width}x{info.height}","-r",str(info.rate),"-i","pipe:0",
                    "-an","-vf",f"setsar={info.sar.numerator}/{info.sar.denominator}",
                    "-c:v","libx264","-crf","17","-preset","medium","-threads","2",
                    "-pix_fmt",pix_fmt,str(temp/'silent.mp4')],stdin=subprocess.PIPE,
                    stderr=encode_errors,env=native_process_environment())
                while True:
                    if should_stop():
                        raise core.ProcessingCancelled()
                    raw = _read_frame(decoder.stdout,frame_size)
                    if not raw:
                        break
                    frame = np.frombuffer(raw,np.uint8).reshape(info.height,info.width,3).copy()
                    faces = engine.analyser.get(frame)
                    face = tracker.select(frame,faces)
                    if tracker.cut:
                        engine.begin_segment()
                        log_once("Scene change: rechecking the selected person's identity and face model.")
                    if face is not None:
                        frame = engine.swap_array(frame,source_face,quality="careful",detected_faces=[face],
                            restoration_strength=restoration_strength,destination_eyes=destination_eyes,
                            preserve_mouth=preserve_mouth,on_log=log_once)[0]
                        swapped += 1
                        if first_image is None:
                            first_image = frame.copy()
                    else:
                        unchanged += 1
                    try:
                        encoder.stdin.write(frame.tobytes())
                    except BrokenPipeError as exc:
                        encode_errors.seek(0)
                        raise RuntimeError(encode_errors.read().decode(errors="replace")[-2000:] or "Video encoding failed.") from exc
                    decoded += 1
                    elapsed = time.monotonic()-started
                    on_progress(decoded,expected,{"swapped":swapped,"unchanged":unchanged,"elapsed":elapsed,
                        "eta":max(0,expected-decoded)*elapsed/decoded,"provider":engine.provider})
                if decoder.wait(timeout=15):
                    decode_errors.seek(0)
                    raise RuntimeError(decode_errors.read().decode(errors="replace")[-2000:])
                encoder.stdin.close()
                encoder.stdin = None
                if encoder.wait(timeout=60):
                    encode_errors.seek(0)
                    raise RuntimeError(encode_errors.read().decode(errors="replace")[-2000:])
                if not swapped:
                    raise RuntimeError("No confidently matched face was swapped. No output saved. Choose a clearer person or start point.")
                if abs(decoded-expected) > max(2,round(info.fps*.1)):
                    raise RuntimeError(f"Video decoding stopped early: expected about {expected} frames, got {decoded}. No output saved.")
                if should_stop():
                    raise core.ProcessingCancelled()
                length = decoded/info.fps
                run_tool([ffmpeg,"-v","error","-nostdin","-y","-i",str(temp/'silent.mp4'),
                    "-ss",str(start),"-i",str(path),"-map","0:v:0","-map","1:a:0?","-c:v","copy",
                    "-c:a","aac","-b:a","192k","-t",str(length),"-map_metadata","-1",
                    "-metadata:s:v:0","rotate=0","-movflags","+faststart",str(temp/'finished.mp4')],timeout=120,should_stop=should_stop)
                checked, stream = probe(temp/'finished.mp4',count_frames=True,should_stop=should_stop)
                if ((checked.width,checked.height)!=(info.width,info.height) or checked.rate!=info.rate or
                        int(stream.get("nb_read_frames",0))!=decoded or
                        abs(checked.duration-length)>max(.1,1/info.fps) or info.audio!=checked.audio or info.sar!=checked.sar):
                    raise RuntimeError("Encoded video failed frame count, timing, dimensions or audio validation. No output saved.")
                if should_stop():
                    raise core.ProcessingCancelled()
                # All verification precedes the public filename and success count.
                (temp/'finished.mp4').replace(output)
            finally:
                _terminate(decoder)
                _terminate(encoder)
    on_log(f"Saved: {output}\n{swapped} swapped frames; {unchanged} unchanged frames without a confident person match.")
    return {"output":str(output),"frames":decoded,"swapped":swapped,"unchanged":unchanged,
            "duration":decoded/info.fps,"provider":engine.provider,"image":first_image,
            "preview":preview,"source":str(path),"seconds":time.monotonic()-started}
