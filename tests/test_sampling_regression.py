"""Regression from a real 1440x1080 / 29.999989 FPS recording.

Only hit frame numbers are retained, not the video, images or player data.
The previous pipeline produced 32 episodes at gap=1.25 but only one at gap=1.
"""

import pytest

from szfd import DetectorOptions, analyze, core


FPS = 29.999989058476256
HIT_FRAMES = [
    360, 2250, 5430, 5460, 5490, 5520, 5760, 5850, 5880, 5910, 5940, 6060,
    8370, 9060, 9360, 9390, 9420, 9450, 9750, 10200, 10350, 10890, 11070, 13500,
    14550, 14580, 14610, 14640, 14670, 14700, 15450, 15480, 15510, 15540, 16020,
    16050, 16080, 16110, 16200, 16230, 16260, 16980, 18360, 18390, 18420, 18450,
    18510, 18540, 18570, 18630, 20040, 21030, 21300, 21540, 21720, 22350, 22440,
    23100, 23130, 23160, 23190, 23310, 23340, 23370, 23400, 23850, 23880, 23910,
    23940, 24780, 25050, 25080, 25140, 25170, 25200, 25230, 25260, 25350, 25530,
    26340, 26370, 26400, 26430, 26970, 27000, 27030, 27060, 27240, 27885, 28920,
    29340, 29370, 29400, 29430, 29580, 29820, 30180, 30690, 30900, 30930, 30960,
    30990, 31320, 31350, 31380, 31410, 31500, 31830, 31860, 31890, 32040, 32070,
    32100, 32130, 32160, 32190, 32220, 32250, 32880, 33330, 33360, 33390, 33420,
    33450, 33480, 33510, 33540, 34350, 34380, 34410, 34440, 35610, 36570, 36810,
    36840, 36870, 37065, 37066, 37073, 37075, 37076, 37077, 37078, 37098, 37980,
    38010, 38040, 38070, 38310, 38340, 38370, 38400, 40050, 40080, 40200, 40260,
    40290, 40320, 40350, 40380, 40410, 40440, 40470, 40500, 40530, 40560, 42360,
    42630, 44040, 49980, 50880, 51990, 52470, 52500, 52530, 52560, 52770, 52800,
    52830, 52860, 52980, 53010, 53040, 53070, 53430,
]


@pytest.mark.parametrize("merge_gap", [0.25, 0.5, 0.75, 1.0, 1.25])
@pytest.mark.parametrize("multikill", [False, True])
def test_real_recording_keeps_32_episodes(tmp_path, monkeypatch, merge_gap, multikill):
    path = tmp_path / "sample.mp4"
    path.touch()
    hits = [core.Detection(frame, frame / FPS, 0.9, (0, 0, 10, 10)) for frame in HIT_FRAMES]
    legacy = core.cluster_detections(hits, 1.25, 2)
    assert len(legacy) == 32
    monkeypatch.setattr(core, "scan_video", lambda *args: (hits, FPS, 53923 / FPS, 53923, 8))
    result = analyze(path, DetectorOptions(merge_gap=merge_gap, multikill=multikill))
    assert result.events == tuple(legacy)
    assert result.sampling_interval == 30 / FPS
    assert result.effective_merge_gap >= result.sampling_interval


def test_rounding_does_not_split_consecutive_samples():
    hits = [core.Detection(frame, frame / FPS, .9, (0, 0, 10, 10))
            for frame in (5430, 5460, 5490, 5520)]
    assert len(core.cluster_detections(hits, 1, 2)) == 1


def test_true_gap_with_missing_sample_still_splits():
    hits = [core.Detection(frame, frame / FPS, .9, (0, 0, 10, 10))
            for frame in (5430, 5460, 5520, 5550)]
    groups = core.cluster_detections(hits, .25, 2, sampling_interval=30 / FPS)
    assert len(groups) == 2


def test_high_frequency_scan_can_use_short_merge_gap():
    hits = [core.Detection(frame, frame / 30, .9, (0, 0, 10, 10))
            for frame in (0, 1, 15, 16)]
    groups = core.cluster_detections(hits, .25, 2, sampling_interval=1 / 30)
    assert len(groups) == 2


def test_cli_explains_sampling_adjustment(tmp_path, monkeypatch, capsys):
    from szfd.cli import manual_main

    path = tmp_path / "sample.mp4"
    path.touch()
    hits = [core.Detection(frame, frame / FPS, .9, (0, 0, 10, 10))
            for frame in (5430, 5460, 5490, 5520)]
    monkeypatch.setattr(core, "scan_video", lambda *args: (hits, FPS, 200, 6000, 0))
    assert manual_main(["--input", str(path), "--merge-gap", "0.5"]) == 0
    output = capsys.readouterr().out
    assert "merge gap adjusted" in output
    assert "Found plaque episodes: 1" in output
