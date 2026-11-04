from pathlib import Path

import cv2
import numpy as np
import pytest

from szfd.cli import DEFAULT_ROI, PlaqueDetector, build_parser, main, validate_args
from szfd.gui import Application, detector_arguments


ASSETS = Path(__file__).resolve().parents[1] / "szfd" / "assets"


@pytest.mark.parametrize("width,height", [
    (640, 480), (1024, 768), (1280, 960), (1280, 1024),
    (1280, 720), (1920, 1080), (1920, 1200), (2560, 1080), (3840, 2160),
])
@pytest.mark.parametrize("stretched", ["native", "from_16_9", "from_4_3"])
@pytest.mark.parametrize("filename,roi", [
    ("skull_template.png", DEFAULT_ROI),
    ("death_template.png", (0.08, 0.07, 0.35, 0.28)),
])
def test_detect_native_and_stretched_resolutions(width, height, stretched, filename, roi):
    detector = PlaqueDetector(ASSETS / filename, roi)
    original = detector.template
    sy = height / 1079
    sx = {"native": sy, "from_16_9": width / 1920,
          "from_4_3": sy * (width / height) / (4 / 3)}[stretched]
    tw, th = round(original.shape[1] * sx), round(original.shape[0] * sy)
    template = cv2.resize(original, (tw, th), interpolation=cv2.INTER_AREA if max(sx, sy) < 1 else cv2.INTER_CUBIC)
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    x = round(width * (roi[0] + roi[2]) / 2 - tw / 2)
    y = round(height * (roi[1] + roi[3]) / 2 - th / 2)
    frame[y:y+th, x:x+tw] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    result = detector.detect(frame)
    assert result is not None
    assert abs(result.bbox[0] - x) <= 2
    assert abs(result.bbox[1] - y) <= 2


def test_tiny_frame_returns_no_detection():
    detector = PlaqueDetector(ASSETS / "skull_template.png")
    assert detector.detect(np.zeros((1, 1, 3), dtype=np.uint8)) is None


def test_gui_arguments_round_trip(tmp_path):
    video = tmp_path / "video with spaces.mp4"
    video.touch()
    arguments = detector_arguments({"input": str(video), "cut_before": "15", "ffmpeg": ""}, True, True, True)
    parser = build_parser()
    args = parser.parse_args(arguments)
    validate_args(args, parser)
    assert args.input == video
    assert args.cut and args.debug and args.cut_mode == "exact"
    assert args.cut_before == 15


def test_no_arguments_launches_gui(monkeypatch):
    monkeypatch.setattr("szfd.gui.launch", lambda args: 42)
    monkeypatch.setattr("szfd.terminal.launch", lambda args: 43)
    assert main([]) == 43
    assert main(["--menu"]) == 43
    assert main(["--gui"]) == 42


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_reject_non_finite_setting(value):
    parser = build_parser()
    args = parser.parse_args(["--input", __file__, "--cut-before=" + value])
    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_window_smoke():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        app = Application(root, build_parser().parse_args([]))
        assert len(app.buttons) == 9
        assert app.move(1) == "break"
        assert app.selected == 1
        app.toggle_cut()
        assert not app.cut
        root.update_idletasks()
    finally:
        root.destroy()
