from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from szfd.cli import (
    Detection,
    Event,
    cluster_frame_hits,
    cluster_detections,
    format_timecode,
    group_events_for_clips,
    parse_roi,
)


def detection(seconds: float, score: float = 0.8) -> Detection:
    return Detection(frame=round(seconds * 60), seconds=seconds, score=score, bbox=(1, 2, 3, 4))


def event(index: int, seconds: float) -> Event:
    best = detection(seconds, 0.9)
    return Event(
        index=index,
        seconds=seconds,
        last_seconds=seconds + 0.5,
        score=best.score,
        hits=2,
        best_detection=best,
    )


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


def test_death_screen_hits_trigger_one_rewind_per_screen() -> None:
    hits = [90, 120, 150, 600, 630]
    assert cluster_frame_hits(hits, max_gap_frames=60) == [90, 600]


def test_multifrag_groups_close_events_into_one_clip() -> None:
    groups = group_events_for_clips(
        [event(1, 15.0), event(2, 21.0), event(3, 40.0)],
        duration=45.0,
        before=10.0,
        after=10.0,
        multifrag_gap=10.0,
    )

    assert [len(group.events) for group in groups] == [2, 1]
    assert (groups[0].start_seconds, groups[0].end_seconds) == (5.0, 31.0)
    assert (groups[1].start_seconds, groups[1].end_seconds) == (30.0, 45.0)


def test_multifrag_supports_chained_events() -> None:
    groups = group_events_for_clips(
        [event(1, 10.0), event(2, 19.0), event(3, 28.0)],
        duration=60.0,
        before=10.0,
        after=10.0,
        multifrag_gap=10.0,
    )

    assert len(groups) == 1
    assert len(groups[0].events) == 3
    assert (groups[0].start_seconds, groups[0].end_seconds) == (0.0, 38.0)
