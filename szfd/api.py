"""Public Python API. Analysis never writes files or opens an interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from . import core
from .core import ClipGroup, Detection, Event, PlaqueDetector


@dataclass(frozen=True)
class DetectorOptions:
    frame_step: int = 30
    threshold: float = 0.72
    roi: tuple[float, float, float, float] = core.DEFAULT_ROI
    death_threshold: float = 0.72
    death_roi: tuple[float, float, float, float] = core.DEFAULT_DEATH_ROI
    death_rescan: float = 3.0
    death_frame_step: int = 1
    merge_gap: float = 1.25
    min_hits: int = 2
    cut: bool = False
    debug: bool = False
    cut_before: float = 10.0
    cut_after: float = 10.0
    multifrag_gap: float = 10.0
    multikill: bool = False
    plaque_duration: float = 5.0
    cut_mode: str = "copy"
    template: Path = Path(__file__).parent / "assets" / "skull_template.png"
    death_template: Path = Path(__file__).parent / "assets" / "death_template.png"
    ffmpeg: str | None = None
    quiet: bool = True

    @property
    def should_cut(self) -> bool:
        return self.cut or self.multikill

    def validate(self) -> None:
        for name in ("threshold", "death_threshold", "death_rescan", "merge_gap",
                     "cut_before", "cut_after", "multifrag_gap", "plaque_duration"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("frame_step", "death_frame_step", "min_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        for name in ("threshold", "death_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("death_rescan", "merge_gap", "cut_before", "cut_after", "multifrag_gap"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.plaque_duration <= 0:
            raise ValueError("plaque_duration must be positive")
        if self.cut_mode not in ("copy", "exact"):
            raise ValueError("cut_mode must be copy or exact")
        for name in ("roi", "death_roi"):
            roi = getattr(self, name)
            if len(roi) != 4 or not all(math.isfinite(value) for value in roi):
                raise ValueError(f"{name} must contain four finite values")
            x1, y1, x2, y2 = roi
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"{name} must be normalized and ordered")


@dataclass(frozen=True)
class AnalysisResult:
    input_path: Path
    options: DetectorOptions
    detections: tuple[Detection, ...]
    events: tuple[Event, ...]
    clip_groups: tuple[ClipGroup, ...]
    fps: float
    duration: float
    total_frames: int
    death_rewinds: int
    is_image: bool

    @property
    def sampling_interval(self) -> float:
        return self.options.frame_step / self.fps if self.fps > 0 else 0.0

    @property
    def effective_merge_gap(self) -> float:
        return max(self.options.merge_gap, self.sampling_interval)


@dataclass(frozen=True)
class ExportResult:
    analysis: AnalysisResult
    csv_path: Path
    artifacts_dir: Path
    clips: tuple[Path, ...]


def analyze(input_path: str | Path, options: DetectorOptions | None = None) -> AnalysisResult:
    """Detect visible plaque episodes and plan clips; no files are created.

    Counts describe visibility episodes, not an exact number of kills. Failures
    raise standard Python exceptions; this API never calls sys.exit().
    """
    options = options or DetectorOptions()
    options.validate()
    path = Path(input_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    detector = PlaqueDetector(Path(options.template), options.roi, options.threshold)
    is_image = path.suffix.lower() in core.IMAGE_SUFFIXES
    fps, duration, total_frames, rewinds = 0.0, 0.0, 1, 0
    if is_image:
        detections, _ = core.scan_image(path, detector)
    else:
        death_detector = PlaqueDetector(Path(options.death_template), options.death_roi, options.death_threshold)
        detections, fps, duration, total_frames, rewinds = core.scan_video(
            path, detector, death_detector, options.frame_step, options.death_rescan,
            options.death_frame_step, options.quiet,
        )
    # Preserve 0.5 scan evidence when applying min_hits. A coarse hit and its
    # confirmation during death rewind were both counted by the original CLI.
    # Deduplicating here silently rejects previously accepted short episodes.
    detections = sorted(detections, key=lambda item: item.frame)
    sampling_interval = options.frame_step / fps if fps > 0 else 0.0
    events = core.cluster_detections(detections, options.merge_gap,
                                     1 if is_image else options.min_hits, sampling_interval)
    groups = core.group_events_for_clips(
        events, duration, options.cut_before, options.cut_after, options.multifrag_gap,
        options.multikill, options.plaque_duration,
        sampling_interval,
    )
    return AnalysisResult(path, options, tuple(detections), tuple(events), tuple(groups),
                          fps, duration, total_frames, rewinds, is_image)


def export(result: AnalysisResult, output: str | Path | None = None) -> ExportResult:
    """Write CSV, optional debug frames and optional clips for an analyzed file."""
    options = result.options
    directory = Path(output).resolve() if output is not None else None
    if directory is not None and directory.exists() and not directory.is_dir():
        raise ValueError("output must be a directory")
    csv_path, artifacts = core.resolve_output_paths(result.input_path, directory, options.should_cut)
    core.write_csv_report(csv_path, result.events, result.clip_groups)
    if options.debug:
        if result.is_image and result.detections:
            frame = core.cv2.imread(str(result.input_path))
            if frame is None:
                raise RuntimeError(f"Cannot reopen image: {result.input_path}")
            core.save_debug_image(frame, result.detections[0], artifacts)
        elif not result.is_image:
            core.save_debug_frames(result.input_path, result.events, artifacts)
    clips = ()
    if options.should_cut and result.clip_groups and not result.is_image:
        ffmpeg = core.find_ffmpeg(options.ffmpeg)
        clips = tuple(core.cut_clips(ffmpeg, result.input_path, artifacts,
                      result.clip_groups, options.cut_mode, options.multikill, options.quiet))
    return ExportResult(result, csv_path, artifacts, clips)


def process(input_path: str | Path, output: str | Path | None = None,
            options: DetectorOptions | None = None) -> ExportResult:
    """Convenience operation: analyze a file and export its results."""
    return export(analyze(input_path, options), output)
