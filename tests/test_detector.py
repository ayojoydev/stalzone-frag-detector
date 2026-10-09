from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from szfd.cli import Detection, cluster_detections, format_timecode, parse_roi


def detection(seconds: float, score: float = 0.8) -> Detection:
    return Detection(frame=round(seconds * 60), seconds=seconds, score=score, bbox=(1, 2, 3, 4))


def test_format_timecode() -> None:
    assert format_timecode(0) == "00:00:00.000"
    assert format_timecode(65.4321) == "00:01:05.432"
    assert format_timecode(3661.9996) == "01:01:02.000"


def test_parse_roi() -> None:
    assert parse_roi("0.1,0.2,0.8,0.9") == (0.1, 0.2, 0.8, 0.9)


def test_cluster_deduplicates_one_plaque() -> None:
    hits = [detection(10.0, 0.80), detection(10.25, 0.91), detection(10.5, 0.85)]
    events = cluster_detections(hits, merge_gap=1.25, min_hits=2)
    assert len(events) == 1
    assert events[0].seconds == 10.0
    assert events[0].last_seconds == 10.5
    assert events[0].score == 0.91
    assert events[0].hits == 3


def test_cluster_splits_separate_plaques_and_filters_noise() -> None:
    hits = [detection(5.0), detection(5.25), detection(12.0), detection(20.0), detection(20.2)]
    events = cluster_detections(hits, merge_gap=1.25, min_hits=2)
    assert [event.seconds for event in events] == [5.0, 20.0]
