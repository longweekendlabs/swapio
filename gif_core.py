"""Frame-wise GIF transport around the shared Careful renderer.

Pillow composites input disposal/offsets; full-canvas local-palette output frames
are written incrementally, retaining delays, looping and binary transparency.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import GifImagePlugin, Image

import core
import video_core as video

GIF_EXTENSIONS = {'.gif'}


@dataclass(frozen=True)
class GifInfo(video.VideoInfo):
    delays: tuple[int, ...] = ()
    loop: int | None = None


def probe(path, should_stop=lambda: False):
    delays = []
    with Image.open(path) as image:
        if image.format != 'GIF':
            raise ValueError('This file is not a GIF.')
        width, height = image.size
        loop = image.info.get('loop')
        for index in range(image.n_frames):
            if should_stop():
                raise core.ProcessingCancelled()
            image.seek(index)
            image.load()
            # Zero/absent delays have no portable playback speed. Use 100 ms.
            delays.append(int(image.info.get('duration', 0)) or 100)
    total = sum(delays)
    return GifInfo(Path(path), width, height, Fraction(len(delays)*1000, total),
                   total/1000, False, delays=tuple(delays), loop=loop)


def _range(info, start, end):
    start, end = video.clip_range(info, start, end)
    # GIF timestamps have 10 ms precision. Retain exact frame delays otherwise.
    start, end = round(start*100), round(end*100)
    if end <= start:
        raise ValueError('Choose a GIF range of at least 0.01 seconds.')
    return start*10, end*10


def _frames(path, info, start, end, should_stop):
    timestamp = 0
    with Image.open(path) as image:
        for index, delay in enumerate(info.delays):
            if should_stop():
                raise core.ProcessingCancelled()
            next_time = timestamp+delay
            if timestamp >= end:
                break
            if next_time > start:
                image.seek(index)
                yield image.convert('RGBA'), min(next_time, end)-max(timestamp, start), max(timestamp, start)
            timestamp = next_time


def _bgr(rgba):
    opaque = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba)
    return cv2.cvtColor(np.array(opaque.convert('RGB')), cv2.COLOR_RGB2BGR)


def choose_person(engine, path, start=0., end=0., selector=None, should_stop=lambda: False):
    info = probe(path, should_stop)
    first, last = _range(info, start, end)
    engine.ensure_loaded()
    for rgba, _, timestamp in _frames(path, info, first, min(last, first+10000), should_stop):
        image = _bgr(rgba)
        faces = sorted(engine.analyser.get(image), key=core._face_area, reverse=True)
        if not faces:
            continue
        choice = selector(path, image, faces) if selector else (0 if len(faces) == 1 else None)
        if choice is None:
            raise core.ProcessingCancelled()
        if not isinstance(choice, int) or not 0 <= choice < len(faces):
            raise ValueError('The selected person is unavailable.')
        face = faces[choice]
        stat = Path(path).stat()
        return video.PersonSelection(str(Path(path).resolve()), (stat.st_size, stat.st_mtime_ns), start,
            video._embedding(face), image, np.asarray(face.bbox).copy(), timestamp/1000)
    raise RuntimeError('No selectable face in the first 10 seconds of this range. Move Start to a clearer face and try again.')


def _palette_frame(bgr, alpha):
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frame = rgb.quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                         dither=Image.Dither.FLOYDSTEINBERG)
    palette = (frame.getpalette() or [])[:765]
    frame.putpalette(palette+[0]*(768-len(palette)))
    frame.paste(255, mask=Image.fromarray(np.where(alpha < 128, 255, 0).astype(np.uint8)))
    return frame


def render_gif(engine, source_path, path, output_dir, selection, *, start=0., end=0.,
               preview=False, character_name='', restoration_strength=core.DEFAULT_RESTORATION_STRENGTH,
               destination_eyes=False, preserve_mouth=True, should_stop=lambda: False,
               on_log=lambda _: None, on_progress=lambda *_: None):
    info = probe(path, should_stop)
    if preview:
        end = min(end or info.duration, start+video.PREVIEW_SECONDS)
    first, last = _range(info, start, end)
    if not selection.valid(path, start):
        raise RuntimeError('This GIF or its start changed. Choose its person again.')
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = core.unique_output_path(output_dir, Path(path), suffix='.gif', name_prefix=character_name)
    if output.resolve() == Path(path).resolve():
        raise ValueError('Output must differ from the source GIF.')
    engine.begin_segment()
    source_face = engine.source_face(source_path, on_log)
    tracker = video.PersonTracker(selection)
    swapped = unchanged = 0
    first_image = None
    delays = []
    started = time.monotonic()
    messages = set()
    def log_once(message):
        if message not in messages:
            messages.add(message)
            on_log(message)
    # Output stays private until every frame and its timing have been verified.
    with tempfile.TemporaryDirectory(prefix='.swapio-gif-', dir=output_dir) as folder:
        temporary = Path(folder)/'finished.gif'
        with temporary.open('wb') as writer:
            for rgba, delay, _ in _frames(path, info, first, last, should_stop):
                frame = _bgr(rgba)
                face = tracker.select(frame, engine.analyser.get(frame))
                if tracker.cut:
                    engine.begin_segment()
                if face is not None:
                    frame = engine.swap_array(frame, source_face, quality='careful', detected_faces=[face],
                        restoration_strength=restoration_strength, destination_eyes=destination_eyes,
                        preserve_mouth=preserve_mouth, on_log=log_once)[0]
                    swapped += 1
                    if first_image is None:
                        first_image = frame.copy()
                else:
                    unchanged += 1
                indexed = _palette_frame(frame, np.array(rgba.getchannel('A')))
                if not delays:
                    header = dict(background=255, transparency=255, optimize=False)
                    if info.loop is not None:
                        header['loop'] = info.loop
                    for chunk in GifImagePlugin.getheader(indexed, info=header)[0]:
                        writer.write(chunk)
                # Full frames with disposal=2 clear previous opaque pixels, too.
                for chunk in GifImagePlugin.getdata(indexed, duration=delay, disposal=2,
                        transparency=255, include_color_table=True):
                    writer.write(chunk)
                delays.append(delay)
                elapsed = time.monotonic()-started
                done = sum(delays)
                on_progress(done, last-first, dict(swapped=swapped, unchanged=unchanged, elapsed=elapsed,
                    eta=max(0,last-first-done)*elapsed/done, provider=engine.provider, unit='time'))
            writer.write(b';')
        if not swapped:
            raise RuntimeError('No confidently matched face was swapped. No output saved. Choose a clearer person or start point.')
        checked = probe(temporary, should_stop)
        if ((checked.width, checked.height) != (info.width, info.height) or
                checked.delays != tuple(delays) or checked.loop != info.loop):
            raise RuntimeError('Encoded GIF failed frame count, timing, dimensions or loop validation. No output saved.')
        if should_stop():
            raise core.ProcessingCancelled()
        temporary.replace(output)
    on_log(f'Saved: {output}\n{swapped} swapped frames; {unchanged} unchanged frames without a confident person match.')
    return dict(output=str(output), frames=len(delays), swapped=swapped, unchanged=unchanged,
        duration=sum(delays)/1000, provider=engine.provider, image=first_image,
        preview=preview, source=str(path), seconds=time.monotonic()-started)
