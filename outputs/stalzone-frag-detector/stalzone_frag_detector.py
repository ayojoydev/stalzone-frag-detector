#!/usr/bin/env python3
"""Detect Stalzone kill plaques and optionally cut clips around them."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_ROI = (0.36, 0.68, 0.64, 0.86)
REFERENCE_HEIGHT = 1079
SCALE_FACTORS = (0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5)


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

    def detect(self, frame: np.ndarray) -> Detection | None:
        height, width = frame.shape[:2]
        x1 = round(self.roi[0] * width)
        y1 = round(self.roi[1] * height)
        x2 = round(self.roi[2] * width)
        y2 = round(self.roi[3] * height)
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_location = (0, 0)
        best_size = (0, 0)
        base_scale = height / REFERENCE_HEIGHT

        for factor in SCALE_FACTORS:
            scale = base_scale * factor
            template_width = max(8, round(self.template.shape[1] * scale))
            template_height = max(8, round(self.template.shape[0] * scale))
            if template_width >= gray.shape[1] or template_height >= gray.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            template = cv2.resize(
                self.template,
                (template_width, template_height),
                interpolation=interpolation,
            )
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
) -> list[Event]:
    if not detections:
        return []

    groups: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.seconds):
        if not groups or detection.seconds - groups[-1][-1].seconds > merge_gap:
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


def scan_video(
    path: Path,
    detector: PlaqueDetector,
    frame_step: int,
    quiet: bool,
) -> tuple[list[Detection], float, float, int]:
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
    frame_index = 0
    last_progress = -1

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

            if not quiet and total_frames > 0:
                progress = min(100, int(frame_index * 100 / total_frames))
                if progress >= last_progress + 10:
                    print(f"Scanning: {progress}%")
                    last_progress = progress
            frame_index += 1
    finally:
        capture.release()

    return detections, fps, duration, total_frames


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


def write_reports(
    output_dir: Path,
    input_path: Path,
    events: Sequence[Event],
    config: dict[str, object],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "timecodes.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(f"{event.index:03d}  {format_timecode(event.seconds)}  score={event.score:.3f}\n")

    with (output_dir / "detections.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", "seconds", "timecode", "score", "hits", "last_seconds"])
        for event in events:
            writer.writerow(
                [
                    event.index,
                    f"{event.seconds:.3f}",
                    format_timecode(event.seconds),
                    f"{event.score:.4f}",
                    event.hits,
                    f"{event.last_seconds:.3f}",
                ]
            )

    payload = {
        "input": str(input_path.resolve()),
        "config": config,
        "metadata": metadata,
        "events": [
            {
                "index": event.index,
                "seconds": round(event.seconds, 3),
                "timecode": format_timecode(event.seconds),
                "last_seconds": round(event.last_seconds, 3),
                "score": round(event.score, 4),
                "hits": event.hits,
                "best_frame": event.best_detection.frame,
                "bbox": list(event.best_detection.bbox),
            }
            for event in events
        ],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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
    events: Iterable[Event],
    duration: float,
    before: float,
    after: float,
    cut_mode: str,
) -> None:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for event in events:
        start = max(0.0, event.seconds - before)
        end = event.seconds + after
        if duration > 0:
            end = min(duration, end)
        clip_duration = max(0.01, end - start)
        suffix = input_path.suffix if cut_mode == "copy" else ".mp4"
        output_path = clips_dir / f"frag_{event.index:03d}_{format_timecode(event.seconds).replace(':', '-')}{suffix}"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
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
        print(f"Cutting clip {event.index}/{len(events) if isinstance(events, Sequence) else '?'}")
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detect Stalzone kill plaques and optionally cut frag clips."
    )
    parser.add_argument("input", type=Path, help="Video file or a single image for calibration")
    parser.add_argument(
        "--template",
        type=Path,
        default=script_dir / "assets" / "skull_template.png",
        help="Path to the skull template image",
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--frame-step", type=int, default=15, help="Check every Nth frame")
    parser.add_argument("--threshold", type=float, default=0.72, help="Template score threshold")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=DEFAULT_ROI,
        metavar="X1,Y1,X2,Y2",
        help="Normalized search region",
    )
    parser.add_argument("--merge-gap", type=float, default=1.25, help="Seconds between hits in one event")
    parser.add_argument("--min-hits", type=int, default=2, help="Required sampled frames per event")
    parser.add_argument("--debug", action="store_true", help="Save frames with marked detections")
    parser.add_argument("--clips", action="store_true", help="Cut a clip for every event")
    parser.add_argument("--before", type=float, default=12.0, help="Seconds before a frag")
    parser.add_argument("--after", type=float, default=12.0, help="Seconds after a frag")
    parser.add_argument(
        "--cut-mode",
        choices=("copy", "exact"),
        default="copy",
        help="copy is fast and lossless; exact re-encodes for exact boundaries",
    )
    parser.add_argument("--ffmpeg", help="Path or command name for FFmpeg")
    parser.add_argument("--quiet", action="store_true", help="Hide scan progress")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.merge_gap < 0 or args.min_hits < 1:
        parser.error("--merge-gap must be non-negative and --min-hits at least 1")
    if args.before < 0 or args.after < 0:
        parser.error("--before and --after must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    output_dir = args.output or args.input.with_name(f"{args.input.stem}_frags")
    detector = PlaqueDetector(args.template, args.roi, args.threshold)
    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES

    config = {
        "template": str(args.template.resolve()),
        "frame_step": args.frame_step,
        "threshold": args.threshold,
        "roi": list(args.roi),
        "merge_gap": args.merge_gap,
        "min_hits": args.min_hits,
        "before": args.before,
        "after": args.after,
        "cut_mode": args.cut_mode,
    }

    try:
        if is_image:
            detections, frame = scan_image(args.input, detector)
            events = cluster_detections(detections, args.merge_gap, 1)
            metadata = {"kind": "image", "width": frame.shape[1], "height": frame.shape[0]}
            write_reports(output_dir, args.input, events, config, metadata)
            if args.debug and detections:
                save_debug_image(frame, detections[0], output_dir)
            if detections:
                print(f"Kill plaque found, score={detections[0].score:.3f}")
            else:
                print("Kill plaque not found")
            return 0 if detections else 2

        detections, fps, duration, total_frames = scan_video(
            args.input, detector, args.frame_step, args.quiet
        )
        events = cluster_detections(detections, args.merge_gap, args.min_hits)
        metadata = {
            "kind": "video",
            "fps": round(fps, 6),
            "duration": round(duration, 3),
            "total_frames": total_frames,
            "sampled_frames": (total_frames + args.frame_step - 1) // args.frame_step,
            "raw_hits": len(detections),
        }
        write_reports(output_dir, args.input, events, config, metadata)
        if args.debug:
            save_debug_frames(args.input, events, output_dir)
        if args.clips and events:
            ffmpeg = find_ffmpeg(args.ffmpeg)
            cut_clips(
                ffmpeg,
                args.input,
                output_dir,
                events,
                duration,
                args.before,
                args.after,
                args.cut_mode,
            )

        print(f"Found events: {len(events)}")
        print(f"Results: {output_dir.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
