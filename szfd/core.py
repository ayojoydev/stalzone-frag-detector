#!/usr/bin/env python3
"""Shared detection, event grouping, reporting and clip export engine."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_ROI = (0.36, 0.68, 0.64, 0.86)
DEFAULT_DEATH_ROI = (0.08, 0.07, 0.35, 0.28)
REFERENCE_HEIGHT = 1079
REFERENCE_WIDTH = 1920
SCALE_FACTORS = (0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5)
VERSION = "0.6.1"


@dataclass(frozen=True)
class Detection:
    frame: int
    seconds: float
    score: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Event:
    index: int
    seconds: float
    last_seconds: float
    score: float
    hits: int
    best_detection: Detection


@dataclass(frozen=True)
class ClipGroup:
    index: int
    events: tuple[Event, ...]
    start_seconds: float
    end_seconds: float


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        roi = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain four decimal values") from exc
    if len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError("ROI values must be normalized and ordered")
    return roi  # type: ignore[return-value]


def format_timecode(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class PlaqueDetector:
    def __init__(
        self,
        template_path: Path,
        roi: tuple[float, float, float, float] = DEFAULT_ROI,
        threshold: float = 0.72,
    ) -> None:
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Cannot read template: {template_path}")
        self.template = template
        self.roi = roi
        self.threshold = threshold
        self._template_cache: dict[tuple[int, int], list[np.ndarray]] = {}

    def scaled_templates(self, width: int, height: int) -> list[np.ndarray]:
        """Match both height-scaled HUDs and independently stretched recordings."""
        key = (width, height)
        if key not in self._template_cache:
            sizes = set()
            templates = []
            for factor in SCALE_FACTORS:
                sy = height / REFERENCE_HEIGHT * factor
                for sx in (sy, width / REFERENCE_WIDTH * factor,
                           sy * (width / height) / (4 / 3)):
                    size = (max(2, round(self.template.shape[1] * sx)),
                            max(2, round(self.template.shape[0] * sy)))
                    if size in sizes:
                        continue
                    sizes.add(size)
                    templates.append(cv2.resize(
                        self.template, size,
                        interpolation=cv2.INTER_AREA if max(sx, sy) < 1 else cv2.INTER_CUBIC,
                    ))
            # Videos normally have one size; bound memory if used on many images.
            self._template_cache.clear()
            self._template_cache[key] = templates
        return self._template_cache[key]

    def detect(self, frame: np.ndarray) -> Detection | None:
        height, width = frame.shape[:2]
        x1 = round(self.roi[0] * width)
        y1 = round(self.roi[1] * height)
        x2 = round(self.roi[2] * width)
        y2 = round(self.roi[3] * height)
        # HUDs may keep their height-based horizontal offsets at non-16:9 ratios.
        # Custom ROIs remain exact; broaden only the supplied defaults.
        if self.roi in (DEFAULT_ROI, DEFAULT_DEATH_ROI):
            anchor = 0.5 if self.roi == DEFAULT_ROI else 0.0
            hud_width = height * REFERENCE_WIDTH / REFERENCE_HEIGHT
            x1 = max(0, min(x1, round(anchor * width + (self.roi[0] - anchor) * hud_width)))
            x2 = min(width, max(x2, round(anchor * width + (self.roi[2] - anchor) * hud_width)))
        if x2 <= x1 or y2 <= y1:
            return None
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_location = (0, 0)
        best_size = (0, 0)
        for template in self.scaled_templates(width, height):
            template_height, template_width = template.shape
            if template_width > gray.shape[1] or template_height > gray.shape[0]:
                continue
            matches = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(matches)
            if score > best_score:
                best_score = float(score)
                best_location = location
                best_size = (template_width, template_height)

        if best_score < self.threshold:
            return None

        left = x1 + best_location[0]
        top = y1 + best_location[1]
        template_width, template_height = best_size
        patch = frame[top : top + template_height, left : left + template_width]
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        white_ratio = float(np.mean(patch_gray >= 210))
        dark_ratio = float(np.mean(patch_gray <= 110))

        # The icon is a small white skull on a dark translucent plaque.
        if white_ratio < 0.025 or dark_ratio < 0.20:
            return None

        return Detection(
            frame=0,
            seconds=0.0,
            score=best_score,
            bbox=(left, top, template_width, template_height),
        )


def cluster_detections(
    detections: Sequence[Detection],
    merge_gap: float,
    min_hits: int,
    sampling_interval: float = 0.0,
) -> list[Event]:
    if not detections:
        return []

    groups: list[list[Detection]] = []
    # Consecutive coarse samples must be able to confirm the same plaque.
    # Otherwise merge_gap <= frame_step/fps turns every real hit into a singleton.
    # A tiny tolerance handles e.g. 29.999989 FPS and floating point subtraction.
    effective_gap = max(merge_gap, sampling_interval)
    for detection in sorted(detections, key=lambda item: item.seconds):
        if not groups or detection.seconds - groups[-1][-1].seconds > effective_gap + 1e-6:
            groups.append([detection])
        else:
            groups[-1].append(detection)

    events: list[Event] = []
    for group in groups:
        if len(group) < min_hits:
            continue
        best = max(group, key=lambda item: item.score)
        events.append(
            Event(
                index=len(events) + 1,
                seconds=group[0].seconds,
                last_seconds=group[-1].seconds,
                score=best.score,
                hits=len(group),
                best_detection=best,
            )
        )
    return events


def cluster_frame_hits(frame_indices: Sequence[int], max_gap_frames: int) -> list[int]:
    starts: list[int] = []
    previous: int | None = None
    for frame_index in sorted(set(frame_indices)):
        if previous is None or frame_index - previous > max_gap_frames:
            starts.append(frame_index)
        previous = frame_index
    return starts


def rescan_before_deaths(
    path: Path,
    detector: PlaqueDetector,
    death_frames: Sequence[int],
    fps: float,
    seconds_before: float,
    frame_step: int,
    quiet: bool,
) -> list[Detection]:
    if not death_frames or seconds_before <= 0:
        return []

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen video for death-screen rescan: {path}")

    windows = [
        (max(0, death_frame - round(seconds_before * fps)), death_frame)
        for death_frame in death_frames
    ]
    progress = tqdm(
        total=sum(end - start for start, end in windows),
        desc="Death rewind",
        unit="frame",
        dynamic_ncols=True,
        mininterval=0.2,
        disable=quiet,
    )
    detections: list[Detection] = []
    scanned_frames: set[int] = set()

    try:
        for start_frame, death_frame in windows:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            while frame_index < death_frame:
                ok = capture.grab()
                if not ok:
                    break
                if (
                    frame_index not in scanned_frames
                    and (frame_index - start_frame) % frame_step == 0
                ):
                    ok, frame = capture.retrieve()
                    if not ok:
                        break
                    scanned_frames.add(frame_index)
                    result = detector.detect(frame)
                    if result is not None:
                        detections.append(
                            Detection(
                                frame=frame_index,
                                seconds=frame_index / fps,
                                score=result.score,
                                bbox=result.bbox,
                            )
                        )
                frame_index += 1
                progress.update(1)
    finally:
        progress.close()
        capture.release()

    return detections


def scan_video(
    path: Path,
    detector: PlaqueDetector,
    death_detector: PlaqueDetector,
    frame_step: int,
    death_rescan_seconds: float,
    death_frame_step: int,
    quiet: bool,
) -> tuple[list[Detection], float, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise RuntimeError("Video reports an invalid frame rate")

    duration = total_frames / fps if total_frames > 0 else 0.0
    detections: list[Detection] = []
    death_hits: list[int] = []
    frame_index = 0
    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="Scanning",
        unit="frame",
        dynamic_ncols=True,
        mininterval=0.2,
        disable=quiet,
    )

    try:
        while capture.grab():
            if frame_index % frame_step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                result = detector.detect(frame)
                if result is not None:
                    detections.append(
                        Detection(
                            frame=frame_index,
                            seconds=frame_index / fps,
                            score=result.score,
                            bbox=result.bbox,
                        )
                    )
                if death_rescan_seconds > 0:
                    death_result = death_detector.detect(frame)
                    if death_result is not None:
                        death_hits.append(frame_index)

            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    death_triggers = cluster_frame_hits(
        death_hits,
        max_gap_frames=max(frame_step * 2, round(fps * 2)),
    )
    detections.extend(
        rescan_before_deaths(
            path,
            detector,
            death_triggers,
            fps,
            death_rescan_seconds,
            death_frame_step,
            quiet,
        )
    )

    return detections, fps, duration, total_frames, len(death_triggers)


def scan_image(path: Path, detector: PlaqueDetector) -> tuple[list[Detection], np.ndarray]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot open image: {path}")
    result = detector.detect(frame)
    return ([result] if result is not None else []), frame


def save_debug_frames(
    video_path: Path,
    events: Sequence[Event],
    output_dir: Path,
) -> None:
    if not events:
        return
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        for event in events:
            detection = event.best_detection
            capture.set(cv2.CAP_PROP_POS_FRAMES, detection.frame)
            ok, frame = capture.read()
            if not ok:
                continue
            left, top, width, height = detection.bbox
            cv2.rectangle(
                frame,
                (left, top),
                (left + width, top + height),
                (0, 255, 0),
                2,
            )
            label = f"frag {event.index} score={event.score:.3f}"
            cv2.putText(
                frame,
                label,
                (left, max(24, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            filename = f"event_{event.index:03d}_{format_timecode(event.seconds).replace(':', '-')}.jpg"
            cv2.imwrite(str(debug_dir / filename), frame)
    finally:
        capture.release()


def save_debug_image(frame: np.ndarray, detection: Detection, output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    preview = frame.copy()
    left, top, width, height = detection.bbox
    cv2.rectangle(preview, (left, top), (left + width, top + height), (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"score={detection.score:.3f}",
        (left, max(24, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_dir / "image_check.jpg"), preview)


def write_csv_report(
    csv_path: Path,
    events: Sequence[Event],
    clip_groups: Sequence[ClipGroup],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    group_by_event = {
        event.index: (group.index, len(group.events), group.start_seconds, group.end_seconds)
        for group in clip_groups
        for event in group.events
    }
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event",
                "seconds",
                "timecode",
                "score",
                "hits",
                "last_seconds",
                "clip_group",
                "group_size",
                "clip_start",
                "clip_end",
                "visible_seconds",
            ]
        )
        for event in events:
            group_index, group_size, clip_start, clip_end = group_by_event[event.index]
            writer.writerow(
                [
                    event.index,
                    f"{event.seconds:.3f}",
                    format_timecode(event.seconds),
                    f"{event.score:.4f}",
                    event.hits,
                    f"{event.last_seconds:.3f}",
                    group_index,
                    group_size,
                    f"{clip_start:.3f}",
                    f"{clip_end:.3f}",
                    f"{event.last_seconds - event.seconds:.3f}",
                ]
            )


def group_events_for_clips(
    events: Sequence[Event],
    duration: float,
    before: float,
    after: float,
    multifrag_gap: float,
    multikill: bool = False,
    plaque_duration: float = 5.0,
    sampling_interval: float = 0.0,
) -> list[ClipGroup]:
    if not events:
        return []

    grouped_events: list[list[Event]] = []
    for event in sorted(events, key=lambda item: item.seconds):
        if (
            not grouped_events
            or event.seconds - (
                max(grouped_events[-1][-1].seconds,
                    grouped_events[-1][-1].last_seconds + sampling_interval - plaque_duration)
                if multikill else grouped_events[-1][-1].seconds
            ) > multifrag_gap
        ):
            grouped_events.append([event])
        else:
            grouped_events[-1].append(event)

    groups: list[ClipGroup] = []
    for index, group in enumerate(grouped_events, start=1):
        start = max(0.0, group[0].seconds - before)
        # A refreshed icon can remain on screen without disappearing between kills.
        # Preserve its complete visible interval, not just the first detection.
        end = max(event.last_seconds + sampling_interval if multikill else event.seconds for event in group) + after
        if duration > 0:
            end = min(duration, end)
        groups.append(
            ClipGroup(
                index=index,
                events=tuple(group),
                start_seconds=start,
                end_seconds=end,
            )
        )
    return groups


def find_ffmpeg(explicit_path: str | None) -> str:
    if explicit_path:
        executable = Path(explicit_path)
        if executable.is_file():
            return str(executable)
        resolved = shutil.which(explicit_path)
        if resolved:
            return resolved
        raise FileNotFoundError(f"FFmpeg not found: {explicit_path}")

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise FileNotFoundError(
            "FFmpeg was not found. Install imageio-ffmpeg or pass --ffmpeg PATH."
        ) from exc


def cut_clips(
    ffmpeg: str,
    input_path: Path,
    output_dir: Path,
    groups: Sequence[ClipGroup],
    cut_mode: str,
    multikill: bool = False,
    quiet: bool = False,
) -> list[Path]:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for group in groups:
        clip_duration = max(0.01, group.end_seconds - group.start_seconds)
        suffix = input_path.suffix if cut_mode == "copy" else ".mp4"
        first_timecode = format_timecode(group.events[0].seconds).replace(":", "-")
        if multikill:
            # Visibility episodes are not an exact kill counter.
            filename = f"moment_{group.index:03d}_{first_timecode}{suffix}"
        elif len(group.events) == 1:
            filename = f"frag_{group.index:03d}_{first_timecode}{suffix}"
        else:
            filename = (
                f"multifrag_{group.index:03d}_{len(group.events)}frags_"
                f"{first_timecode}{suffix}"
            )
        output_path = clips_dir / filename

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{group.start_seconds:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{clip_duration:.3f}",
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-map",
            "0:s?",
        ]
        if cut_mode == "copy":
            command.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-c:s",
                    "copy",
                ]
            )
        command.append(str(output_path))
        if not quiet:
            print(f"Cutting clip {group.index}/{len(groups)}")
        subprocess.run(command, check=True)
        outputs.append(output_path)
    return outputs


def resolve_output_paths(
    input_path: Path,
    output: Path | None,
    cut: bool,
) -> tuple[Path, Path]:
    output_root = output or input_path.parent
    artifacts_dir = output_root / f"{input_path.stem}_frags"
    csv_dir = artifacts_dir if cut else output_root
    csv_path = csv_dir / f"{input_path.stem}_fragtime.csv"
    return csv_path, artifacts_dir
