from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from szfd import DetectorOptions, analyze, export, process
from szfd import core
from szfd.cli import build_parser, manual_main, validate_args


def episode(index, start, end):
    hit = core.Detection(index, start, 0.9, (0, 0, 10, 10))
    return core.Event(index, start, end, 0.9, 2, hit)


def test_continuous_icon_is_not_truncated():
    # Five seconds before + ten seconds of the refreshed icon + three after = 18.
    groups = core.group_events_for_clips([episode(1, 10, 20)], 100, 5, 3, 10, multikill=True)
    assert (groups[0].start_seconds, groups[0].end_seconds) == (5, 23)
    assert groups[0].end_seconds - groups[0].start_seconds == 18


def test_multikill_off_preserves_legacy_bounds():
    group = core.group_events_for_clips([episode(1, 10, 20)], 100, 5, 3, 10)[0]
    assert (group.start_seconds, group.end_seconds) == (5, 13)


def test_chain_extends_until_last_visible_episode():
    events = [episode(1, 10, 15), episode(2, 20, 25), episode(3, 30, 35)]
    groups = core.group_events_for_clips(events, 100, 5, 3, 10, multikill=True)
    assert len(groups) == 1
    assert (groups[0].start_seconds, groups[0].end_seconds) == (5, 38)


def test_long_continuous_series_then_another_kill():
    # Refreshed icon ends at 20 => inferred last kill at 15. Next at 25: gap 10.
    groups = core.group_events_for_clips([episode(1, 10, 20), episode(2, 25, 30)],
                                         100, 5, 3, 10, multikill=True)
    assert len(groups) == 1
    assert groups[0].end_seconds == 33


def test_gap_over_ten_splits_series():
    groups = core.group_events_for_clips([episode(1, 10, 15), episode(2, 20.01, 25.01)],
                                         100, 5, 3, 10, multikill=True)
    assert len(groups) == 2


def test_sampling_padding_and_video_boundaries():
    group = core.group_events_for_clips([episode(1, 2, 11.5)], 13, 5, 3, 10,
                                        multikill=True, sampling_interval=0.5)[0]
    assert (group.start_seconds, group.end_seconds) == (0, 13)
    group = core.group_events_for_clips([episode(1, 10, 19.5)], 100, 5, 3, 10,
                                        multikill=True, sampling_interval=0.5)[0]
    assert group.end_seconds == 23


def test_flag_implies_cut(tmp_path):
    path = tmp_path / "video.mp4"
    path.touch()
    parser = build_parser()
    args = parser.parse_args(["--input", str(path), "--multikill"])
    validate_args(args, parser)
    assert args.cut and args.multikill
    assert args.plaque_duration == 5 and args.multifrag_gap == 10


@pytest.mark.parametrize("changes", [
    {"plaque_duration": 0}, {"plaque_duration": float("inf")},
    {"frame_step": 1.5}, {"multifrag_gap": -1}, {"threshold": float("nan")},
    {"roi": (0, 0, 2, 1)}, {"cut_mode": "invalid"},
])
def test_library_rejects_invalid_options(changes):
    with pytest.raises(ValueError):
        replace(DetectorOptions(), **changes).validate()


def test_missing_file_raises_exception_not_system_exit(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "missing.mp4")


def test_library_has_no_cli_or_ui_imports():
    completed = subprocess.run([sys.executable, "-c",
        "import sys, szfd; assert 'szfd.cli' not in sys.modules; "
        "assert 'tkinter' not in sys.modules; assert 'szfd.terminal' not in sys.modules"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_manual_cli_no_args_shows_help(capsys):
    assert manual_main([]) == 0
    assert "--multikill" in capsys.readouterr().out


def test_analysis_and_export_are_separate(tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.touch()
    hits = [core.Detection(frame, frame / 10, 0.9, (0, 0, 10, 10)) for frame in range(100, 201, 10)]
    # Preserve legacy confirmation counting across the coarse pass and rewind.
    monkeypatch.setattr(core, "scan_video", lambda *args: (hits + [hits[0]], 10, 100, 1000, 1))
    options = DetectorOptions(multikill=True, frame_step=10, cut_before=5, cut_after=3)
    result = analyze(path, options)
    assert result.events[0].hits == len(hits) + 1
    assert not list(tmp_path.glob("*.csv"))
    assert not (tmp_path / "video_frags").exists()
    assert result.clip_groups[0].end_seconds == 24  # Last hit 20 + sampling interval 1 + after 3.
    calls = []
    monkeypatch.setattr(core, "find_ffmpeg", lambda _: "ffmpeg")
    monkeypatch.setattr(core.subprocess, "run", lambda command, **kwargs: calls.append(command))
    saved = export(result, tmp_path / "results")
    assert saved.csv_path.is_file()
    assert len(saved.clips) == 1
    assert saved.clips[0].name.startswith("moment_")
    assert calls[0][calls[0].index("-ss") + 1] == "5.000"
    assert calls[0][calls[0].index("-t") + 1] == "19.000"
    assert "clip_end" in saved.csv_path.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("multikill", [False, True])
def test_short_episodes_keep_legacy_rescan_confirmation(tmp_path, monkeypatch, multikill):
    path = tmp_path / "video.mp4"
    path.touch()
    # Twenty short plaques, each seen once in the coarse scan and confirmed
    # at the same frame by rewind. The 0.6 deduplication incorrectly lost all 20.
    coarse = [core.Detection(index * 300, index * 10, 0.9, (0, 0, 10, 10))
              for index in range(1, 21)]
    raw = coarse + coarse
    monkeypatch.setattr(core, "scan_video", lambda *args: (raw, 30, 300, 9000, 20))
    legacy_events = core.cluster_detections(raw, 1.25, 2)
    result = analyze(path, DetectorOptions(multikill=multikill))
    assert len(result.events) == len(legacy_events) == 20
    assert result.events == tuple(legacy_events)
    assert all(event.hits == 2 for event in result.events)


def test_real_video_multikill_and_exact_export(tmp_path):
    """End-to-end: refreshed icon, video decoding and actual FFmpeg export."""
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()
    detector = core.PlaqueDetector(Path(core.__file__).parent / "assets" / "skull_template.png")
    template = cv2.resize(detector.template,
        (round(detector.template.shape[1] * 480 / 1079), round(detector.template.shape[0] * 480 / 1079)),
        interpolation=cv2.INTER_AREA)
    th, tw = template.shape
    for index in range(220):
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        if 50 <= index < 150:
            frame[360:360 + th, 310:310 + tw] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    saved = process(video, tmp_path / "out", DetectorOptions(
        multikill=True, frame_step=5, death_rescan=0, cut_before=5, cut_after=3, cut_mode="exact"))
    assert len(saved.analysis.events) == 1
    group = saved.analysis.clip_groups[0]
    assert group.start_seconds == 0
    assert group.end_seconds == 18
    assert len(saved.clips) == 1 and saved.clips[0].is_file()
    capture = cv2.VideoCapture(str(saved.clips[0]))
    try:
        length = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        assert abs(length - 18) < 0.2
    finally:
        capture.release()
