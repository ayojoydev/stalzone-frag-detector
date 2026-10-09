#!/usr/bin/env python3
"""Detect Stalzone kill plaques and optionally cut clips around them."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
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
SCALE_FACTORS = (0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5)
VERSION = "0.4.1"


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
        event.index: (group.index, len(group.events))
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
            ]
        )
        for event in events:
            group_index, group_size = group_by_event[event.index]
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
                ]
            )


def group_events_for_clips(
    events: Sequence[Event],
    duration: float,
    before: float,
    after: float,
    multifrag_gap: float,
) -> list[ClipGroup]:
    if not events:
        return []

    grouped_events: list[list[Event]] = []
    for event in sorted(events, key=lambda item: item.seconds):
        if (
            not grouped_events
            or event.seconds - grouped_events[-1][-1].seconds > multifrag_gap
        ):
            grouped_events.append([event])
        else:
            grouped_events[-1].append(event)

    groups: list[ClipGroup] = []
    for index, group in enumerate(grouped_events, start=1):
        start = max(0.0, group[0].seconds - before)
        end = group[-1].seconds + after
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
) -> None:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        clip_duration = max(0.01, group.end_seconds - group.start_seconds)
        suffix = input_path.suffix if cut_mode == "copy" else ".mp4"
        first_timecode = format_timecode(group.events[0].seconds).replace(":", "-")
        if len(group.events) == 1:
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
        print(
            f"Cutting clip {group.index}/{len(groups)} "
            f"({len(group.events)} frag{'s' if len(group.events) != 1 else ''})"
        )
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="SZFD",
        description="Detect Stalzone kill plaques and optionally cut frag clips."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Video file (legacy positional form)",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_option",
        type=Path,
        help="Video file or a single image for calibration",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=script_dir / "assets" / "skull_template.png",
        help="Path to the skull template image",
    )
    parser.add_argument(
        "--death-template",
        type=Path,
        default=script_dir / "assets" / "death_template.png",
        help="Path to the death-screen template image",
    )
    parser.add_argument("-o", "--output", type=Path, help="Directory for the result")
    parser.add_argument("--frame-step", type=int, default=30, help="Check every Nth frame")
    parser.add_argument("--threshold", type=float, default=0.72, help="Template score threshold")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=DEFAULT_ROI,
        metavar="X1,Y1,X2,Y2",
        help="Normalized search region",
    )
    parser.add_argument(
        "--death-threshold",
        type=float,
        default=0.72,
        help="Death-screen template score threshold",
    )
    parser.add_argument(
        "--death-roi",
        type=parse_roi,
        default=DEFAULT_DEATH_ROI,
        metavar="X1,Y1,X2,Y2",
        help="Normalized death-screen search region",
    )
    parser.add_argument(
        "--death-rescan",
        type=float,
        default=3.0,
        help="Seconds to rescan before a death screen; 0 disables it",
    )
    parser.add_argument(
        "--death-frame-step",
        type=int,
        default=1,
        help="Check every Nth frame during a death-screen rescan",
    )
    parser.add_argument("--merge-gap", type=float, default=1.25, help="Seconds between hits in one event")
    parser.add_argument("--min-hits", type=int, default=2, help="Required sampled frames per event")
    parser.add_argument("--debug", action="store_true", help="Save frames with marked detections")
    parser.add_argument(
        "--cut",
        "--clips",
        dest="cut",
        action="store_true",
        help="Cut clips around detected events",
    )
    parser.add_argument(
        "--cut-before",
        "--before",
        dest="cut_before",
        type=float,
        default=10.0,
        help="Seconds before the first frag in a clip",
    )
    parser.add_argument(
        "--cut-after",
        "--after",
        dest="cut_after",
        type=float,
        default=10.0,
        help="Seconds after the last frag in a clip",
    )
    parser.add_argument(
        "--multifrag-gap",
        type=float,
        default=10.0,
        help="Maximum seconds between frags grouped into one clip",
    )
    parser.add_argument(
        "--cut-mode",
        choices=("copy", "exact"),
        default="copy",
        help="copy is fast and lossless; exact re-encodes for exact boundaries",
    )
    parser.add_argument("--ffmpeg", help="Path or command name for FFmpeg")
    parser.add_argument("--quiet", action="store_true", help="Hide scan progress")
    parser.add_argument("--version", action="version", version=f"SZFD {VERSION}")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.input_path is not None and args.input_option is not None:
        parser.error("Use either positional input or --input, not both")
    args.input = args.input_option or args.input_path
    if args.input is None:
        parser.error("--input is required")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.output is not None and args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if not 0 < args.death_threshold <= 1:
        parser.error("--death-threshold must be in (0, 1]")
    if args.death_rescan < 0:
        parser.error("--death-rescan must be non-negative")
    if args.death_frame_step < 1:
        parser.error("--death-frame-step must be at least 1")
    if args.merge_gap < 0 or args.min_hits < 1:
        parser.error("--merge-gap must be non-negative and --min-hits at least 1")
    if args.cut_before < 0 or args.cut_after < 0:
        parser.error("--cut-before and --cut-after must be non-negative")
    if args.multifrag_gap < 0:
        parser.error("--multifrag-gap must be non-negative")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    csv_path, artifacts_dir = resolve_output_paths(
        args.input,
        args.output,
        args.cut,
    )
    detector = PlaqueDetector(args.template, args.roi, args.threshold)
    death_detector = PlaqueDetector(
        args.death_template,
        args.death_roi,
        args.death_threshold,
    )
    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES

    try:
        if is_image:
            detections, frame = scan_image(args.input, detector)
            events = cluster_detections(detections, args.merge_gap, 1)
            clip_groups = group_events_for_clips(
                events,
                duration=0,
                before=args.cut_before,
                after=args.cut_after,
                multifrag_gap=args.multifrag_gap,
            )
            write_csv_report(csv_path, events, clip_groups)
            if args.debug and detections:
                save_debug_image(frame, detections[0], artifacts_dir)
            if detections:
                print(f"Kill plaque found, score={detections[0].score:.3f}")
            else:
                print("Kill plaque not found")
            print(f"Timecodes: {csv_path.resolve()}")
            return 0 if detections else 2

        detections, _fps, duration, _total_frames, death_rewinds = scan_video(
            args.input,
            detector,
            death_detector,
            args.frame_step,
            args.death_rescan,
            args.death_frame_step,
            args.quiet,
        )
        events = cluster_detections(detections, args.merge_gap, args.min_hits)
        clip_groups = group_events_for_clips(
            events,
            duration,
            args.cut_before,
            args.cut_after,
            args.multifrag_gap,
        )
        write_csv_report(csv_path, events, clip_groups)
        if args.debug:
            save_debug_frames(args.input, events, artifacts_dir)
        if args.cut and clip_groups:
            ffmpeg = find_ffmpeg(args.ffmpeg)
            cut_clips(
                ffmpeg,
                args.input,
                artifacts_dir,
                clip_groups,
                args.cut_mode,
            )

        print(f"Found events: {len(events)}")
        if death_rewinds:
            print(f"Death-screen rewinds: {death_rewinds}")
        print(f"Timecodes: {csv_path.resolve()}")
        return 0
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
