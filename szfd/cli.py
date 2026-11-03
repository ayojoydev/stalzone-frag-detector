"""Manual command-line adapter over the shared public API."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .api import DetectorOptions, process
# Preserve imports used by integrations written against earlier versions.
from .core import (
    VERSION, IMAGE_SUFFIXES, DEFAULT_ROI, DEFAULT_DEATH_ROI, REFERENCE_HEIGHT,
    SCALE_FACTORS, Detection, Event, ClipGroup, PlaqueDetector,
    cluster_detections, cluster_frame_hits, cut_clips, find_ffmpeg,
    format_timecode, group_events_for_clips, parse_roi, rescan_before_deaths,
    resolve_output_paths, save_debug_frames, save_debug_image, scan_image,
    scan_video, write_csv_report,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = DetectorOptions()
    parser = argparse.ArgumentParser(prog="SZFD", description="Detect Stalzone kill plaques and cut full moments.")
    parser.add_argument("input_path", nargs="?", type=Path, help="Video or image (positional form)")
    parser.add_argument("-i", "--input", dest="input_option", type=Path, help="Video or image")
    parser.add_argument("-o", "--output", type=Path, help="Directory for results")
    parser.add_argument("--template", type=Path, default=defaults.template, help="Skull template")
    parser.add_argument("--death-template", type=Path, default=defaults.death_template, help="Death-screen template")
    parser.add_argument("--roi", type=parse_roi, default=defaults.roi, help="Normalized search region X1,Y1,X2,Y2")
    parser.add_argument("--death-roi", type=parse_roi, default=defaults.death_roi, help="Death-screen search region")
    numbers = (
        ("frame_step", int, "Check every Nth frame"),
        ("threshold", float, "Template score threshold (0, 1]"),
        ("death_threshold", float, "Death-screen template threshold"),
        ("death_rescan", float, "Seconds to rescan before death; 0 disables it"),
        ("death_frame_step", int, "Frame step during death rescan"),
        ("merge_gap", float, "Seconds between hits in one visible plaque episode"),
        ("min_hits", int, "Required sampled frames per episode"),
        ("multifrag_gap", float, "Maximum seconds between kills in a series (default: 10)"),
        ("plaque_duration", float, "Single-kill icon lifetime, seconds (default: 5)"),
    )
    for name, kind, help_text in numbers:
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name), help=help_text)
    parser.add_argument("--cut", "--clips", action="store_true", help="Cut clips around detected episodes")
    parser.add_argument("--cut-before", "--before", type=float, default=defaults.cut_before, help="Seconds before first kill")
    parser.add_argument("--cut-after", "--after", type=float, default=defaults.cut_after, help="Seconds after kill (after visible series with --multikill)")
    parser.add_argument("--multikill", "--multikills", action="store_true",
                        help="Cut complete visible series and join kills within --multifrag-gap; implies --cut")
    parser.add_argument("--cut-mode", choices=("copy", "exact"), default=defaults.cut_mode, help="copy is fast; exact re-encodes")
    parser.add_argument("--debug", action="store_true", help="Save marked frames")
    parser.add_argument("--ffmpeg", help="FFmpeg executable path")
    parser.add_argument("--quiet", action="store_true", help="Hide processing progress")
    interface = parser.add_mutually_exclusive_group()
    interface.add_argument("--menu", action="store_true", help="Open the color console menu")
    interface.add_argument("--gui", action="store_true", help="Open the legacy graphical interface")
    parser.add_argument("--version", action="version", version=f"SZFD {VERSION}")
    return parser


def options_from_args(args: argparse.Namespace) -> DetectorOptions:
    return DetectorOptions(**{field.name: getattr(args, field.name) for field in fields(DetectorOptions)})


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
    try:
        options_from_args(args).validate()
    except ValueError as exc:
        parser.error(str(exc))
    if args.multikill:
        args.cut = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.gui:
        from .gui import launch

        return launch(args)
    if not arguments or args.menu:
        from .terminal import launch

        return launch(args)
    validate_args(args, parser)
    try:
        result = process(args.input, args.output, options_from_args(args))
        analysis = result.analysis
        if analysis.options.merge_gap < analysis.sampling_interval:
            print(f"Plaque merge gap adjusted: {analysis.options.merge_gap:g}s -> "
                  f"{analysis.effective_merge_gap:.6f}s (frame sampling interval). "
                  "For a shorter gap, reduce --frame-step. "
                  "The interval between kills is --multifrag-gap.")
        print(f"Found plaque episodes: {len(analysis.events)}")
        print(f"Moments: {len(analysis.clip_groups)}")
        if analysis.death_rewinds:
            print(f"Death-screen rewinds: {analysis.death_rewinds}")
        print(f"Timecodes: {result.csv_path}")
        return 2 if analysis.is_image and not analysis.detections else 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def manual_main(argv: Sequence[str] | None = None) -> int:
    """Dedicated CLI entry point: no arguments shows help, never opens a menu."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        build_parser().print_help()
        return 0
    return main(arguments)


def menu_main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return main(["--menu", *arguments])


if __name__ == "__main__":
    raise SystemExit(manual_main())
